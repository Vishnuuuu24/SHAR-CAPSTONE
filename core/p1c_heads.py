"""P3A: four visual heads on the SAME cached DINOv2 input, matched budgets.

Why this module exists
-----------------------
Every classification number on record so far (`core/p1c_head_training.py`)
comes from one head architecture: `nn.Linear` on the pooled `[CLS ++
mean(patch)]` vector. P3A asks a narrower question -- given the exact same
frozen features, does a richer head extract more from them? -- so the head is
the ONE thing allowed to differ across the four arms defined here. Everything
about training (optimizer, schedule, seeding, scoring) lives in the sibling
trainer, `core/p1c_head_variants.py`; this module is pure `nn.Module`
definitions with no I/O, matching `docs/ARCHITECTURE.md` section 3's
same-feature-head-ablation convention.

Registered deviation from the phase plan
------------------------------------------
`docs/ARCHITECTURE.md` section 4 specifies the SEMSCNN-inspired head over a
ResNet50 C5 `[B, 2048, 7, 7]` map. The project's selected backbone changed to
DINOv2 (D-44/D-45) before this head was ever built, so `MultiScaleSeHead`
below substitutes DINOv2's own pre-pool spatial representation: the 16x16
grid of 768-D patch tokens the registered patch cache stores
(`results/p1c/p1c-patch-cache/manifest.json`'s `phase_plan_deviation` field
records this same substitution on the data side). The architectural rule that
substitution must respect -- never reshape an already-pooled vector into a
fake image -- is upheld because DINOv2's patch tokens are genuine pre-pool
spatial positions, not a GAP output.

Parameter budgets
-----------------
`GapLinearHead` is the minimal control and is exempt from budget matching by
design (docs/PROJECT_RULES.md rule 3 / the P3A brief): it is deliberately as small as a
linear map can be, because it stands in for "no head at all beyond the
validated baseline". `MlpHead` and `MultiScaleSeHead` are chosen to land in
the same order of magnitude (see each class's docstring for the exact
count). `StepHead` operates on the full 1536-D pooled representation per its
cited architecture (arXiv 2503.24298) rather than a small ablation-scale
input, so its parameter count is honestly much larger; shrinking its internal
width to chase the ~2x band would mean deviating from the paper's own design
to satisfy an unrelated bookkeeping convention, which docs/PROJECT_RULES.md's "never
invent equations" rule forbids. `scripts/run_p1c_head_comparison.py` reports
every count plainly rather than hiding the mismatch.
"""

from __future__ import annotations

import math

import torch
from torch import nn

DEFAULT_NUM_CLASSES = 14


class _CountableHead(nn.Module):
    """Shared `parameter_count()` so every head reports the same way (rule of two)."""

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class GapLinearHead(_CountableHead):
    """The control arm: `nn.Linear` on the already-pooled `[CLS ++ mean(patch)]` vector.

    Consumes the same 1536-D vector every earlier P1C number was trained on
    (`core.p1c_head_training.train_cached_head`'s `GlobalAverageLinearHead(...)
    .classifier`). This class does not itself compute the CLS/mean-patch
    pooling -- that is a data-preparation step
    (`core.p1c_head_variants.pooled_vector_from_tokens`), not a head
    responsibility -- so `forward` is a bare `nn.Linear` and is required by
    the P3A brief to be numerically identical to that existing control.
    """

    def __init__(self, in_dim: int = 1536, num_classes: int = DEFAULT_NUM_CLASSES) -> None:
        super().__init__()
        if in_dim <= 0 or num_classes <= 1:
            raise ValueError("in_dim must be positive and num_classes must exceed one")
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[1] != self.linear.in_features:
            raise ValueError(f"GapLinearHead expects [B, {self.linear.in_features}]; got {tuple(pooled.shape)}")
        return self.linear(pooled)


class MlpHead(_CountableHead):
    """Linear -> GELU -> Dropout -> Linear on the same pooled 1536-D vector as the control.

    `hidden=32` is chosen so this head's parameter count (32*1536 + 32 +
    32*14 + 14 = 49,646 to be exact: see `parameter_count()`) sits in the
    same order of magnitude as `MultiScaleSeHead` at its default width (see
    that class's docstring), rather than at some arbitrary size -- the two
    non-control arms are meant to differ in *structure*, not in how many
    parameters they are given to work with.
    """

    def __init__(self, in_dim: int = 1536, hidden: int = 32, num_classes: int = DEFAULT_NUM_CLASSES, dropout: float = 0.1) -> None:
        super().__init__()
        if min(in_dim, hidden, num_classes) <= 0 or num_classes <= 1:
            raise ValueError("in_dim/hidden/num_classes must be positive, num_classes must exceed one")
        if not 0 <= dropout < 1:
            raise ValueError("dropout must be in [0,1)")
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, num_classes),
        )

    def forward(self, pooled: torch.Tensor) -> torch.Tensor:
        if pooled.ndim != 2 or pooled.shape[1] != self.net[0].in_features:
            raise ValueError(f"MlpHead expects [B, {self.net[0].in_features}]; got {tuple(pooled.shape)}")
        return self.net(pooled)


class MultiScaleSeHead(_CountableHead):
    """SEMSCNN-inspired 2-D multiscale + SE head, adapted to DINOv2's patch grid.

    Source and scope (docs/PROJECT_RULES.md's proposal-name table): the source is
    SEMSCNN, an EEG channel x time architecture; nothing about EEG semantics
    is borrowed, only the 2-D multiscale-branch + SE-recalibration
    *structure* `docs/ARCHITECTURE.md` section 4 describes. That section's
    plan assumed ResNet50's `[B, 2048, 7, 7]` C5 map; this class instead
    consumes DINOv2's genuine pre-pool spatial tokens, `[B, 256, 768]` (a
    16x16 grid of 768-D tokens, registers already excluded by the cache) --
    the one documented substitution the P3A brief authorizes, because the
    project's selected backbone changed (D-44/D-45) and a 768-D GAP vector is
    never reshaped into a fake image here; these are real spatial positions.

    Structure, in order: reshape tokens to `[B, 768, 16, 16]` -> pointwise
    project to `width` channels -> three parallel true 2-D depthwise
    convolutions ({3x3, 5x5, 7x7}, padding preserved, `groups=width`) ->
    channel-concatenate -> pointwise fuse back to `width` -> GroupNorm + GELU
    -> squeeze-and-excite channel recalibration (global average -> reduce ->
    expand -> sigmoid gate, multiplied per-channel) -> global average pool ->
    `nn.Linear(width, num_classes)`.

    `width=32` (`se_reduction=8`) gives 31,282 parameters -- the same order
    of magnitude as `MlpHead` above; see `parameter_count()`.

    GroupNorm rather than BatchNorm2d: this head must also be exercised in
    tests with batch size 1 (the gradient-flow test), and `BatchNorm2d` in
    training mode raises on a batch of one; `GroupNorm` needs no batch
    statistics at all, so behaviour does not change with batch size.
    """

    def __init__(
        self,
        token_dim: int = 768,
        grid: tuple[int, int] = (16, 16),
        width: int = 32,
        se_reduction: int = 8,
        num_classes: int = DEFAULT_NUM_CLASSES,
    ) -> None:
        super().__init__()
        if token_dim <= 0 or width <= 0 or se_reduction <= 0 or num_classes <= 1:
            raise ValueError("token_dim/width/se_reduction must be positive, num_classes must exceed one")
        if len(grid) != 2 or grid[0] <= 0 or grid[1] <= 0:
            raise ValueError(f"grid must be a (height, width) pair of positive ints; got {grid}")
        self.token_dim = token_dim
        self.grid = (int(grid[0]), int(grid[1]))
        self.expected_token_count = self.grid[0] * self.grid[1]

        self.project = nn.Conv2d(token_dim, width, kernel_size=1)
        self.branch_3x3 = nn.Conv2d(width, width, kernel_size=3, padding=1, groups=width)
        self.branch_5x5 = nn.Conv2d(width, width, kernel_size=5, padding=2, groups=width)
        self.branch_7x7 = nn.Conv2d(width, width, kernel_size=7, padding=3, groups=width)
        self.fuse = nn.Conv2d(width * 3, width, kernel_size=1)
        self.norm = nn.GroupNorm(num_groups=min(8, width), num_channels=width)
        self.activation = nn.GELU()

        reduced = max(width // se_reduction, 1)
        self.se_pool = nn.AdaptiveAvgPool2d(1)
        self.se_reduce = nn.Linear(width, reduced)
        self.se_activation = nn.ReLU(inplace=False)
        self.se_expand = nn.Linear(reduced, width)
        self.se_gate = nn.Sigmoid()

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(width, num_classes)

    def forward(self, tokens: torch.Tensor, *, return_debug: bool = False):
        if tokens.ndim != 3 or tokens.shape[2] != self.token_dim:
            raise ValueError(
                f"MultiScaleSeHead expects [B, tokens, {self.token_dim}]; got {tuple(tokens.shape)}"
            )
        if tokens.shape[1] != self.expected_token_count:
            raise ValueError(
                f"token count {tokens.shape[1]} does not match grid {self.grid} "
                f"({self.expected_token_count} expected)"
            )
        batch = tokens.shape[0]
        spatial = tokens.permute(0, 2, 1).reshape(batch, self.token_dim, self.grid[0], self.grid[1])

        projected = self.project(spatial)
        branches = torch.cat(
            [self.branch_3x3(projected), self.branch_5x5(projected), self.branch_7x7(projected)], dim=1
        )
        fused = self.activation(self.norm(self.fuse(branches)))

        pooled = self.se_pool(fused).flatten(1)
        gate = self.se_gate(self.se_expand(self.se_activation(self.se_reduce(pooled))))
        recalibrated = fused * gate.unsqueeze(-1).unsqueeze(-1)

        logits = self.classifier(self.gap(recalibrated).flatten(1))
        if return_debug:
            return logits, {"se_gate": gate}
        return logits


class _SimplifiedSelfAttentionBlock(nn.Module):
    """STEP's stripped attention block: Q/K/V, softmax, weighted sum -- nothing else.

    Deliberately NOT a standard transformer block: no residual/skip
    connection, no layer norm, and no feed-forward sublayer. That absence is
    the paper's contribution (Ponbagavathi & Roitberg, arXiv 2503.24298),
    so it is not "completed" into a normal transformer layer here.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = dim
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        query = self.query(x)
        key = self.key(x)
        value = self.value(x)
        scores = query @ key.transpose(-1, -2) / math.sqrt(self.dim)
        if key_padding_mask is not None:
            # key_padding_mask: [B, S], True at positions to ignore as keys.
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1), float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        return weights @ value


class StepHead(_CountableHead):
    """STEP (arXiv 2503.24298, Ponbagavathi & Roitberg): CLS-token temporal classifier.

    Consumes a per-video sequence of `frames` pooled 1536-D frame features
    (the same `[CLS ++ mean(patch)]` vector the other three heads see one
    frame of at a time, but here as a temporal sequence for one video), and
    predicts 14 logits from the position of a learnable CLS token prepended
    to that sequence. Structure: a learnable frame-wise positional encoding
    (added elementwise, one row per sequence position up to `frames`), a
    learnable CLS token prepended to the sequence, then `depth` (default 1,
    the paper's own setting) simplified self-attention blocks
    (`_SimplifiedSelfAttentionBlock` -- no skip connection, no layer norm, no
    feed-forward), reading the CLS position's output through one linear
    classifier.

    `token_dim=1536` (the full pooled feature width, not a compressed
    ablation-scale projection) is the architecture's own input size per the
    cited paper; see this module's docstring for why its parameter count is
    not shrunk to match the other two non-control heads.

    Order sensitivity is the property the paper exists for: because the
    positional encoding is added by sequence position and the CLS token
    attends over the resulting sequence, permuting the frame axis changes
    which positional code lands on which frame's content, and therefore
    changes the output. A head that ignored order would defeat the point of
    building STEP into this comparison at all -- see
    `tests/test_p1c_heads.py`'s order-sensitivity test.

    `key_padding_mask`, when given, is `[B, frames]` with `True` at padded
    positions (a video sampled with fewer than `frames` frames); those
    positions are excluded as attention keys/values but a batch always
    presents exactly `frames` positions so it can be shaped as one tensor.
    """

    def __init__(self, token_dim: int = 1536, frames: int = 8, depth: int = 1, num_classes: int = DEFAULT_NUM_CLASSES) -> None:
        super().__init__()
        if token_dim <= 0 or frames <= 0 or depth <= 0 or num_classes <= 1:
            raise ValueError("token_dim/frames/depth must be positive, num_classes must exceed one")
        self.token_dim = token_dim
        self.frames = frames
        self.position_embedding = nn.Parameter(torch.zeros(frames, token_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, token_dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.blocks = nn.ModuleList([_SimplifiedSelfAttentionBlock(token_dim) for _ in range(depth)])
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(self, frame_features: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        if frame_features.ndim != 3 or frame_features.shape[2] != self.token_dim:
            raise ValueError(
                f"StepHead expects [B, sequence, {self.token_dim}]; got {tuple(frame_features.shape)}"
            )
        batch, length, _ = frame_features.shape
        if length > self.frames:
            raise ValueError(f"sequence length {length} exceeds the registered window {self.frames}")
        if key_padding_mask is not None and tuple(key_padding_mask.shape) != (batch, length):
            raise ValueError(f"key_padding_mask must be [{batch}, {length}]; got {tuple(key_padding_mask.shape)}")

        positioned = frame_features + self.position_embedding[:length].unsqueeze(0)
        cls = self.cls_token.expand(batch, -1, -1)
        sequence = torch.cat([cls, positioned], dim=1)

        full_mask = None
        if key_padding_mask is not None:
            cls_mask = torch.zeros(batch, 1, dtype=torch.bool, device=key_padding_mask.device)
            full_mask = torch.cat([cls_mask, key_padding_mask], dim=1)

        for block in self.blocks:
            sequence = block(sequence, key_padding_mask=full_mask)

        return self.classifier(sequence[:, 0, :])

class VideoMeanMlpHead(_CountableHead):
    """Capacity control for `StepHead`: mask-aware mean over frames, then an MLP.

    `StepHead` beat the GAP+linear control by +0.0287, and its own permutation
    control (`step_permuted`, frame order destroyed) beat it by a further
    +0.0197 -- so the gain is NOT temporal order. Two candidate explanations
    survive, and they differ by exactly one thing: attention pooling over a
    video's frames, versus simply having 7.1M parameters and predicting a video
    directly instead of voting over its frames.

    This head removes the attention and keeps everything else: the same input
    (`[B, frames, 1536]` pooled frame features with the same padding mask), the
    same video-level prediction, and a parameter count matched to `StepHead`'s
    by stacking three 1536x1536 layers (StepHead's 7.1M is dominated by its
    q/k/v/out projections at the same width). A gain that survives here is
    capacity plus video-level prediction; a gain only `StepHead`/`step_permuted`
    show is attention pooling.
    """

    def __init__(self, token_dim: int = 1536, *, depth: int = 3, num_classes: int = 14) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")
        self.token_dim = token_dim
        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.Linear(token_dim, token_dim), nn.GELU()])
        self.trunk = nn.Sequential(*layers)
        self.classifier = nn.Linear(token_dim, num_classes)

    def forward(
        self, frame_features: torch.Tensor, key_padding_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        if frame_features.dim() != 3 or frame_features.shape[2] != self.token_dim:
            raise ValueError(
                f"VideoMeanMlpHead expects [B, sequence, {self.token_dim}]; got {tuple(frame_features.shape)}"
            )
        if key_padding_mask is None:
            pooled = frame_features.mean(dim=1)
        else:
            batch, length, _ = frame_features.shape
            if tuple(key_padding_mask.shape) != (batch, length):
                raise ValueError(
                    f"key_padding_mask must be [{batch}, {length}]; got {tuple(key_padding_mask.shape)}"
                )
            # `True` marks padding, matching StepHead's convention, so the mean
            # must be taken over the real frames only -- averaging zero-padding
            # in would make a short video's vector systematically smaller.
            keep = (~key_padding_mask).unsqueeze(-1).to(frame_features.dtype)
            pooled = (frame_features * keep).sum(dim=1) / keep.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.trunk(pooled))


class ProjectedStepHead(_CountableHead):
    """D-65: a fixed `Linear(in_dim -> width)` input projection in front of an unchanged `StepHead(width)`.

    Lets backbones of different feature widths share one attention head, so a backbone comparison changes the
    backbone and the projection's input size only. Initialisation order makes the StepHead identical for a given
    seed whatever `in_dim` is: the StepHead is built from the global RNG first, the projection from its own
    generator seeded with `projection_seed`.
    """

    def __init__(self, in_dim: int, *, width: int = 1536, frames: int = 8, num_classes: int = DEFAULT_NUM_CLASSES,
                 projection_seed: int = 0) -> None:
        super().__init__()
        self.step = StepHead(token_dim=width, frames=frames, depth=1, num_classes=num_classes)
        self.projection = nn.Linear(in_dim, width)
        g = torch.Generator().manual_seed(projection_seed)
        bound = 1.0 / math.sqrt(in_dim)
        with torch.no_grad():
            self.projection.weight.uniform_(-bound, bound, generator=g)
            self.projection.bias.uniform_(-bound, bound, generator=g)

    def forward(self, sequences: torch.Tensor, key_padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        return self.step(self.projection(sequences), key_padding_mask=key_padding_mask)
