# SHAR documentation map

> v4.0 · 2026-08-25 · Rewritten under D-36 (governance relaxation).

## Read these to start

1. **`PROJECT_STATUS.md`** — where we are, what is next. The single live resume pointer.
2. **`KNOWN_PROBLEMS.md`** — what is currently broken and how to work around it.

That is enough to begin. Root `docs/PROJECT_RULES.md` holds the working rules.

## Read these when the task touches them

| Document | Use it for |
|---|---|
| `DECISIONS.md` | Why something was decided. Dated rows, frozen at their date. |
| `EVIDENCE_REGISTER.md` | Whether a claim is actually supported, and corrections to dated records. |
| `DATA_SPEC.md` | Label semantics, datasets, splits, provenance, metric separation. |
| `ARCHITECTURE.md` | Design hypotheses and the ablations that would test them. |
| `TRAINING_LOG.md` | Measured figures for runs, and the experiment ledger. |
| `COMPUTE_POLICY.md` | Run-estimate practice. Its M5 Pro specifics are now history; execution is on the DGX. |
| `RISKS.md` | Anticipated reviewer objections and failure modes (future tense — contrast `KNOWN_PROBLEMS.md`). |
| `memory/` | Durable context for agents: owner preferences, cluster facts. Symlinked as the AI assistant's project memory. |
| `TECH_STACK.md`, `ARCHITECTURE.md` | Implementation environment and technical boundaries. |
| `archive/` | Closed work items. History, not live state. |

## Superseded, retained pending archive

`PRD.md`, `REQUIREMENTS.md`, `PHASE_PLAN.md`, `PROJECT_CONTROL.md`, and
`VALIDATION_PLAN.md` (~470 lines between them) predate D-36 and overlap heavily with each
other and with `docs/PROJECT_RULES.md`. They are **not** live authority any more: where one of them
contradicts `docs/PROJECT_RULES.md` or `DECISIONS.md`, those win.

They are retained rather than archived for now because `VALIDATION_PLAN.md`'s claim-to-check
mapping still has real scientific value that should be folded somewhere before it is moved,
and because the repository has uncommitted work at the time of writing. Archive them once
that is resolved — do not treat their process gates (owner registration, layered DoDs,
one-active-sprint) as binding in the meantime; D-36 retired those.

## If documents disagree

Prefer, in order: direct measurement of the current repository and data → `DECISIONS.md`
and `EVIDENCE_REGISTER.md` → everything else. `SHAR_Research_Plan.docx` and the original
proposal terminology are preserved historical input, never current authority.

When you find a conflict, resolve it and update the affected document rather than leaving
both versions standing. A dated record that turns out to be wrong is corrected in
`EVIDENCE_REGISTER.md`, not rewritten in place.

## Current research position

- **Pixel source is the official UCF-Crime MP4 corpus** (D-34): 1,950 files, 97.7 GiB,
  uniformly 320×240 but *not* uniformly 30 fps (K-04). Frames are extracted once at native
  resolution with stride 5 (D-35); training resolution is a separate loader-side choice.
- The Kaggle 64×64 mirror is **retained as a validation oracle** for the extractor, not as
  the pixel source.
- The 14-label result uses weak/noisy video-inherited training labels and an
  official-interval-derived, event-only test protocol. Wording here must stay exact.
- No manual annotation is required. Independent published annotations may be used in their
  native tasks only: UCF-Crime2Local for surveillance boxes, COCO for masks, AVA v2.2 for
  movie action localization.
- Pseudo-labels are ROI proposals or training auxiliaries. Never evaluation ground truth.
- YOLO11 is the pinned stock ROI teacher. Custom detector modules are deferred.
- The SEMSCNN-inspired 2-D visual head is a core, outcome-gated hypothesis. Gaussian
  fusion, detector Slide Loss, and shallow MSDAM placement stay deferred research notes.

## Access and terms

Raw annotations, media, and crops may not be redistributed; upstream footage rights remain
qualified. The official 1,950 MP4 video corpus is the core pixel source under D-34, with source/version/terms
intake resolved (blocker B-002 resolved; see [`results/p0a/official-ucf-videos-access-verification-20260828.json`](../results/p0a/official-ucf-videos-access-verification-20260828.json))
authorizing internal non-commercial research use. UCF-Crime2Local annotations are locally visible but
conditional on native localization work and verified terms. VALU and FS-UCF-Crime remain watchlist items.
