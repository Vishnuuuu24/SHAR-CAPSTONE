"""Registered device-policy resolution and backend dispatch for SHAR.

SHAR never silently falls back between accelerators.  A caller names an exact
registered policy and this module either returns that device or refuses.  The
backend-specific helpers below exist so that no other module has to branch on
``torch.mps`` versus ``torch.cuda`` by hand; adding a backend is a change here
rather than a change in every runner, calibration, and smoke script.

Determinism behaviour is deliberately part of the policy rather than a runner
default: MPS and CUDA do not have the same deterministic-operator coverage, so
"strict" cannot mean one thing across backends.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from typing import Any

import torch


#: Every policy a registered SHAR configuration may name.  ``device_policy``
#: in an experiment YAML is validated against exactly this set.
DEVICE_POLICIES = (
    "cuda_strict",
    "mps_strict",
    "mps_warn_only_repeatability",
    "cpu_strict_deterministic",
)

_POLICY_BACKEND = {
    "cuda_strict": "cuda",
    "mps_strict": "mps",
    "mps_warn_only_repeatability": "mps",
    "cpu_strict_deterministic": "cpu",
}

#: ``torch.use_deterministic_algorithms(..., warn_only=)`` per policy.  Only the
#: explicitly named warn-only policy may downgrade a determinism failure to a
#: warning; see B-006 for why that policy exists at all.
_POLICY_WARN_ONLY = {
    "cuda_strict": False,
    "mps_strict": False,
    "mps_warn_only_repeatability": True,
    "cpu_strict_deterministic": False,
}

#: Host-to-device pinned staging buffers. PyTorch 2.13 accepts
#: ``pin_memory=True`` on MPS but warns and ignores it, so a configuration that
#: sets it is recording an expectation the backend will not honour.
_BACKEND_PIN_MEMORY = {"cuda": True, "mps": False, "cpu": False}

#: Explicit "the backend told us nothing" telemetry, so an artifact never
#: silently reports zero bytes where it actually has no measurement.
_UNAVAILABLE_MEMORY: dict[str, int | None] = {
    "current_allocated_memory": None,
    "driver_allocated_memory": None,
    "recommended_max_memory": None,
    "peak_allocated_memory": None,
}


@dataclass(frozen=True)
class DeviceReport:
    """Exactly what was selected, what was available, and what was refused."""

    selected: str
    mps_built: bool
    mps_available: bool
    cpu_fallback_allowed: bool
    mps_fallback_environment: str | None
    policy: str | None = None
    cuda_available: bool = False
    cuda_device_count: int = 0
    cuda_device_name: str | None = None
    supports_pin_memory: bool = False
    deterministic_warn_only: bool = False
    torch_threads: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _cuda_probe() -> tuple[bool, int, str | None]:
    """Probe CUDA without importing it into existence on unsupported builds."""

    if not torch.cuda.is_available():
        return False, 0, None
    count = torch.cuda.device_count()
    name = torch.cuda.get_device_name(0) if count else None
    return True, count, name


def _report(
    selected: str,
    *,
    policy: str | None,
    allow_cpu_fallback: bool,
    warn_only: bool,
    threads: int | None,
) -> DeviceReport:
    cuda_available, cuda_count, cuda_name = _cuda_probe()
    return DeviceReport(
        selected=selected,
        mps_built=torch.backends.mps.is_built(),
        mps_available=torch.backends.mps.is_available(),
        cpu_fallback_allowed=allow_cpu_fallback,
        mps_fallback_environment=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
        policy=policy,
        cuda_available=cuda_available,
        cuda_device_count=cuda_count,
        cuda_device_name=cuda_name,
        supports_pin_memory=_BACKEND_PIN_MEMORY.get(selected, False),
        deterministic_warn_only=warn_only,
        torch_threads=threads,
    )


def _refuse_mps_fallback_environment(allow_cpu_fallback: bool) -> None:
    if os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK") == "1" and not allow_cpu_fallback:
        raise RuntimeError("PYTORCH_ENABLE_MPS_FALLBACK=1 is prohibited unless CPU fallback is explicitly allowed")


def _enforce_determinism(warn_only: bool) -> None:
    """Apply the policy's determinism mode to the live torch process.

    A ``DeviceReport.deterministic_warn_only`` field that is never read by a
    caller is metadata, not enforcement.  ``resolve_device_policy`` is the one
    place every runner names a policy, so the actual
    ``torch.use_deterministic_algorithms`` call belongs here rather than in
    each runner's own seeding helper, where it can silently go missing (as it
    did prior to the P1C revalidation that found this gap).
    """

    torch.use_deterministic_algorithms(True, warn_only=warn_only)


def resolve_device_policy(policy: object, *, cpu_threads: int | None = None) -> tuple[torch.device, DeviceReport]:
    """Return the device a registered policy names, or refuse to run.

    ``cpu_threads`` is required by, and only meaningful for, the explicit
    CPU policy: an unbounded thread count would make CPU timings and
    determinism unreproducible between machines.
    """

    if policy not in DEVICE_POLICIES:
        raise ValueError(f"device_policy must be one of {sorted(DEVICE_POLICIES)}; got {policy!r}")
    backend = _POLICY_BACKEND[str(policy)]
    warn_only = _POLICY_WARN_ONLY[str(policy)]

    if backend == "cuda":
        cuda_available, count, _ = _cuda_probe()
        if not cuda_available or not count:
            raise RuntimeError(
                f"policy {policy} requires CUDA but torch reports available={cuda_available}, "
                "device_count=0; refusing silent fallback"
            )
        _enforce_determinism(warn_only)
        return torch.device("cuda"), _report(
            "cuda", policy=str(policy), allow_cpu_fallback=False, warn_only=warn_only,
            threads=torch.get_num_threads(),
        )

    if backend == "mps":
        _refuse_mps_fallback_environment(False)
        if not torch.backends.mps.is_available():
            raise RuntimeError(
                f"policy {policy} requires MPS but torch reports built={torch.backends.mps.is_built()}, "
                "available=False; refusing silent CPU fallback"
            )
        _enforce_determinism(warn_only)
        return torch.device("mps"), _report(
            "mps", policy=str(policy), allow_cpu_fallback=False, warn_only=warn_only,
            threads=torch.get_num_threads(),
        )

    if isinstance(cpu_threads, bool) or not isinstance(cpu_threads, int) or cpu_threads <= 0:
        raise ValueError(f"policy {policy} requires a positive explicit cpu_threads")
    torch.set_num_threads(cpu_threads)
    _enforce_determinism(warn_only)
    return torch.device("cpu"), _report(
        "cpu", policy=str(policy), allow_cpu_fallback=False, warn_only=warn_only,
        threads=torch.get_num_threads(),
    )


def select_device(*, prefer_mps: bool = True, allow_cpu_fallback: bool = False) -> tuple[torch.device, DeviceReport]:
    """Legacy Apple-silicon-only entry point retained for P0C-era callers.

    New code should name a registered policy through
    :func:`resolve_device_policy` instead; this wrapper keeps the environment,
    smoke, and synthetic-calibration scripts unchanged.
    """

    mps_available = torch.backends.mps.is_available()
    _refuse_mps_fallback_environment(allow_cpu_fallback)
    if prefer_mps and mps_available:
        selected = "mps"
    elif prefer_mps and not allow_cpu_fallback:
        raise RuntimeError(
            f"MPS requested but unavailable (built={torch.backends.mps.is_built()}, available={mps_available}); "
            "refusing silent CPU fallback"
        )
    else:
        selected = "cpu"
    return torch.device(selected), _report(
        selected, policy=None, allow_cpu_fallback=allow_cpu_fallback, warn_only=False,
        threads=torch.get_num_threads(),
    )


def assert_tensor_device(value: torch.Tensor, expected: torch.device) -> None:
    if value.device.type != expected.type:
        raise RuntimeError(f"tensor device drift: expected {expected.type}, found {value.device.type}")


def policy_backend(policy: object) -> str:
    """The accelerator backend a registered policy name resolves to.

    Exported so that validators can ask "which backend would this policy
    select?" without either importing the private map or re-deriving it by
    splitting the name -- a derivation that happens to work for today's four
    policies and would break silently on the fifth.
    """

    if policy not in DEVICE_POLICIES:
        raise ValueError(f"device_policy must be one of {sorted(DEVICE_POLICIES)}; got {policy!r}")
    return _POLICY_BACKEND[str(policy)]


def supports_pin_memory(device: torch.device | str) -> bool:
    """Whether ``pin_memory=True`` does anything on this backend."""

    return _BACKEND_PIN_MEMORY.get(torch.device(device).type, False)


def synchronize(device: torch.device | str) -> None:
    """Block until queued accelerator work completes; a no-op on CPU.

    Timing code must call this, otherwise an asynchronous backend reports the
    cost of enqueuing work rather than of doing it.
    """

    backend = torch.device(device).type
    if backend == "cuda":
        torch.cuda.synchronize()
    elif backend == "mps":
        torch.mps.synchronize()


def memory_report(device: torch.device | str) -> dict[str, int | None]:
    """Backend memory telemetry, or explicit ``None`` values when unavailable."""

    backend = torch.device(device).type
    if backend == "cuda":
        # Querying a CUDA allocator on a build without CUDA raises from torch's
        # lazy init, so probe availability first exactly as the MPS branch does.
        if not torch.cuda.is_available():
            return dict(_UNAVAILABLE_MEMORY)
        return {
            "current_allocated_memory": int(torch.cuda.memory_allocated()),
            "driver_allocated_memory": int(torch.cuda.memory_reserved()),
            "recommended_max_memory": int(torch.cuda.get_device_properties(0).total_memory),
            "peak_allocated_memory": int(torch.cuda.max_memory_allocated()),
        }
    if backend == "mps":
        # On this macOS 26 / PyTorch combination, calling an MPS allocator query
        # while MPS is unavailable can segfault in libtorch before Python
        # receives an exception (B-008). Never probe unless MPS is really there.
        if not torch.backends.mps.is_available():
            return dict(_UNAVAILABLE_MEMORY)
        result: dict[str, int | None] = {}
        for name in ("current_allocated_memory", "driver_allocated_memory", "recommended_max_memory"):
            function = getattr(torch.mps, name, None)
            try:
                result[name] = int(function()) if function else None
            except RuntimeError:
                result[name] = None
        # torch.mps exposes no peak tracker, so a peak must be sampled by the
        # caller over the run rather than read once at the end.
        result["peak_allocated_memory"] = None
        return result
    return dict(_UNAVAILABLE_MEMORY)


def reset_peak_memory(device: torch.device | str) -> None:
    """Start a fresh peak-memory window where the backend tracks one."""

    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def device_allocated_bytes(device: torch.device | str) -> int | None:
    """One instantaneous allocation sample, for callers maintaining their own peak.

    CUDA tracks a true peak itself, so this returns that peak there.  MPS has no
    peak tracker, so this returns the current driver allocation and the caller is
    responsible for keeping the maximum it observed.
    """

    backend = torch.device(device).type
    if backend == "cuda":
        return int(torch.cuda.max_memory_allocated())
    if backend == "mps" and torch.backends.mps.is_available():
        try:
            return int(torch.mps.driver_allocated_memory())
        except RuntimeError:
            return None
    return None


def accelerator_rng_state() -> dict[str, Any]:
    """Capture accelerator RNG state so a resumed run is bitwise reproducible.

    Every present accelerator is captured rather than only the selected one: a
    checkpoint that omits the RNG of the device it was produced on cannot
    honestly claim exact resume.
    """

    state: dict[str, Any] = {}
    if torch.cuda.is_available():
        state["torch_cuda"] = torch.cuda.get_rng_state_all()
    if torch.backends.mps.is_available():
        state["torch_mps"] = torch.mps.get_rng_state()
    return state


def restore_accelerator_rng_state(state: dict[str, Any]) -> None:
    """Restore captured accelerator RNG, refusing to silently drop any of it."""

    if "torch_cuda" in state:
        if not torch.cuda.is_available():
            raise RuntimeError("checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(state["torch_cuda"])
    if "torch_mps" in state:
        if not torch.backends.mps.is_available():
            raise RuntimeError("checkpoint contains MPS RNG state but MPS is unavailable")
        torch.mps.set_rng_state(state["torch_mps"])
