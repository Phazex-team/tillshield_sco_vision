"""System-memory guard for the inference pipeline.

The DGX Spark target has 121 GiB of unified memory shared by GPU and
CPU. Loading Qwen3-VL (30B MoE BF16) and Gemma BF16 simultaneously
would peak around 100+ GiB and risk OOM-killing the recorder. This
module tracks system memory usage and gates inference work behind
soft / hard / emergency thresholds defined in ``config.yaml``.

The guard is intentionally light: it polls ``psutil.virtual_memory()``,
exposes the current state as a dataclass, and a tiny module-level
``MemoryPolicy`` that callers consult before doing expensive work.
Heavy unload/cleanup callbacks register with the policy; the policy
fires them when a threshold is crossed.

This module has zero hard runtime dependencies on GPU / CUDA so it
remains testable without any model loaded.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import psutil


log = logging.getLogger(__name__)


STATE_NORMAL = "normal"
STATE_SOFT = "soft_limit"
STATE_HARD = "hard_limit"
STATE_EMERGENCY = "emergency_limit"


@dataclass
class MemoryStatus:
    total_gb: float
    used_gb: float
    available_gb: float
    state: str
    inference_allowed: bool
    admission_paused: bool = False
    loaded_providers: list[str] = field(default_factory=list)
    degraded_reason: str = ""
    pending_unloads: list[str] = field(default_factory=list)
    polled_at: float = 0.0


@dataclass
class MemoryPolicyConfig:
    soft_gb: float = 90.0
    hard_gb: float = 100.0
    emergency_gb: float = 110.0
    # Once admission pauses at ``hard_gb`` it stays paused until used
    # memory falls back below ``resume_gb`` — a hysteresis band so the
    # gate doesn't flap open/closed right at the boundary. Defaults to
    # the soft limit.
    resume_gb: float = 90.0
    # Upper bound on how long a single job waits for headroom before it
    # is admitted anyway (best-effort, so the live pipeline can never
    # deadlock permanently if memory stays pinned high). 0 = wait
    # indefinitely (only a shutdown abort preempts it).
    admission_max_wait_sec: float = 300.0
    poll_interval_sec: float = 5.0
    max_loaded_big_vlms: int = 1
    # "defer_jobs" (default): at the hard limit, hold NEW jobs in the
    # queue and let in-flight jobs finish untouched — no weight unload,
    # so accuracy is preserved. "unload_models": legacy behaviour that
    # drops non-active weights (degrades the analysis to REVIEW).
    on_hard_limit: str = "defer_jobs"
    on_emergency_limit: str = "stop_inference_workers"


class MemoryPolicy:
    """Module-singleton guard.

    Callers do three things:

    * ``register_unload_callback(name, fn)`` — when memory enters HARD,
      the policy calls ``fn()`` to unload the named provider/model.
    * ``register_stop_callback(name, fn)`` — when memory enters
      EMERGENCY, the policy stops the named inference worker.
    * ``status()`` — returns the current ``MemoryStatus``.
    * ``allow_new_inference()`` — returns ``True`` only when state is
      ``normal``. Recorder/API paths never call this; only inference
      workers do.
    """

    def __init__(self, cfg: Optional[MemoryPolicyConfig] = None,
                 *,
                 probe: Optional[Callable[[], tuple[float, float]]] = None):
        self.cfg = cfg or MemoryPolicyConfig()
        # probe returns (total_gb, used_gb). Default uses psutil; tests
        # inject a stub.
        self._probe = probe or self._psutil_probe
        self._state = STATE_NORMAL
        # Admission gate (hysteresis-latched in ``poll``). When True a
        # new job must wait before it starts consuming RAM.
        self._admission_paused = False
        self._lock = threading.RLock()
        self._unload_callbacks: dict[str, Callable[[], None]] = {}
        self._stop_callbacks: dict[str, Callable[[], None]] = {}
        self._loaded_providers: set[str] = set()
        self._degraded_reason: str = ""
        self._pending_unloads: set[str] = set()
        self._last_status: Optional[MemoryStatus] = None
        self._poll_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    # ------------------------------------------------------------------
    # Hooks for providers/workers
    # ------------------------------------------------------------------

    def register_unload_callback(self, name: str,
                                 fn: Callable[[], None]) -> None:
        with self._lock:
            self._unload_callbacks[name] = fn

    def register_stop_callback(self, name: str,
                               fn: Callable[[], None]) -> None:
        with self._lock:
            self._stop_callbacks[name] = fn

    def mark_loaded(self, name: str) -> None:
        with self._lock:
            self._loaded_providers.add(name)

    def mark_unloaded(self, name: str) -> None:
        with self._lock:
            self._loaded_providers.discard(name)
            self._pending_unloads.discard(name)

    def loaded_providers(self) -> list[str]:
        with self._lock:
            return sorted(self._loaded_providers)

    # ------------------------------------------------------------------
    # State + decisions
    # ------------------------------------------------------------------

    def poll(self) -> MemoryStatus:
        total_gb, used_gb = self._probe()
        available_gb = max(0.0, total_gb - used_gb)
        prev_state = self._state
        new_state = self._classify(used_gb)
        with self._lock:
            self._state = new_state
            # Admission hysteresis: pause new jobs once we reach the hard
            # limit, stay paused until memory falls back below the resume
            # threshold. This is what keeps queued jobs waiting instead of
            # piling on and pushing the box into swap/OOM (the freeze).
            if used_gb >= self.cfg.hard_gb:
                self._admission_paused = True
            elif used_gb < self.cfg.resume_gb:
                self._admission_paused = False
            # else: inside the hysteresis band — keep previous paused state.
            if new_state != prev_state:
                log.info("memory state %s -> %s  used=%.1fG total=%.1fG",
                         prev_state, new_state, used_gb, total_gb)
                self._on_state_change(prev_state, new_state, used_gb)
            status = MemoryStatus(
                total_gb=round(total_gb, 2),
                used_gb=round(used_gb, 2),
                available_gb=round(available_gb, 2),
                state=new_state,
                inference_allowed=(new_state == STATE_NORMAL),
                admission_paused=self._admission_paused,
                loaded_providers=sorted(self._loaded_providers),
                degraded_reason=self._degraded_reason,
                pending_unloads=sorted(self._pending_unloads),
                polled_at=time.time(),
            )
            self._last_status = status
        return status

    def status(self) -> MemoryStatus:
        if self._last_status is None:
            return self.poll()
        return self._last_status

    def allow_new_inference(self) -> bool:
        return self.status().state == STATE_NORMAL

    def admission_open(self) -> bool:
        """True when a *new* job may start. False while paused at the
        hard limit (in-flight jobs are never gated by this)."""
        return not self.poll().admission_paused

    def wait_for_headroom(self,
                          *,
                          should_abort: Optional[Callable[[], bool]] = None,
                          log_every_sec: float = 30.0) -> bool:
        """Block until system memory has headroom below the hard limit,
        so a queued job waits its turn instead of running and pushing the
        box past its RAM ceiling. In-flight jobs are never touched, and
        no weights are unloaded — accuracy is preserved.

        Returns ``True`` once there is headroom (or immediately if the
        gate is already open). Returns ``False`` if ``should_abort`` fires
        first (e.g. pipeline shutdown) or the ``admission_max_wait_sec``
        best-effort cap is hit — in which case the caller proceeds anyway
        so the live pipeline can never deadlock permanently.
        """
        interval = max(0.5, float(self.cfg.poll_interval_sec))
        max_wait = float(self.cfg.admission_max_wait_sec)
        st = self.poll()
        if not st.admission_paused:
            return True
        waited = 0.0
        next_log = 0.0
        while True:
            if should_abort is not None and should_abort():
                return False
            if waited >= next_log:
                log.warning(
                    "[memory-gate] holding new job: used=%.1fG >= hard=%.1fG; "
                    "waiting for <%.1fG before admitting (in-flight work and "
                    "loaded models untouched)",
                    st.used_gb, self.cfg.hard_gb, self.cfg.resume_gb)
                next_log = waited + log_every_sec
            self._stop_event.wait(interval)
            if self._stop_event.is_set():
                return False
            waited += interval
            st = self.poll()
            if not st.admission_paused:
                log.info("[memory-gate] headroom restored: used=%.1fG < "
                         "resume=%.1fG; admitting job", st.used_gb,
                         self.cfg.resume_gb)
                return True
            if max_wait > 0 and waited >= max_wait:
                log.warning(
                    "[memory-gate] still over hard limit after %.0fs "
                    "(used=%.1fG); admitting job anyway to avoid a stalled "
                    "pipeline — check for a memory leak or lower the model "
                    "footprint", waited, st.used_gb)
                return False

    def reset_for_test(self) -> None:
        """Test-only hook to clear callbacks/state between tests."""
        with self._lock:
            self._state = STATE_NORMAL
            self._admission_paused = False
            self._unload_callbacks.clear()
            self._stop_callbacks.clear()
            self._loaded_providers.clear()
            self._degraded_reason = ""
            self._pending_unloads.clear()
            self._last_status = None

    # ------------------------------------------------------------------
    # Background polling
    # ------------------------------------------------------------------

    def start_polling(self) -> None:
        if self._poll_thread and self._poll_thread.is_alive():
            return
        self._stop_event.clear()
        t = threading.Thread(target=self._poll_loop,
                             name="memory-guard", daemon=True)
        self._poll_thread = t
        t.start()

    def stop_polling(self) -> None:
        self._stop_event.set()
        t = self._poll_thread
        if t and t.is_alive():
            t.join(timeout=2)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                self.poll()
            except Exception:
                log.exception("memory poll failed")
            self._stop_event.wait(self.cfg.poll_interval_sec)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _classify(self, used_gb: float) -> str:
        if used_gb >= self.cfg.emergency_gb:
            return STATE_EMERGENCY
        if used_gb >= self.cfg.hard_gb:
            return STATE_HARD
        if used_gb >= self.cfg.soft_gb:
            return STATE_SOFT
        return STATE_NORMAL

    def _on_state_change(self, prev: str, new: str, used_gb: float) -> None:
        if new == STATE_NORMAL:
            self._degraded_reason = ""
            return
        if new == STATE_SOFT:
            self._degraded_reason = (
                f"soft memory limit crossed (used={used_gb:.1f}G >= "
                f"{self.cfg.soft_gb:.1f}G); deferring new inference jobs"
            )
            return
        if new == STATE_HARD:
            if self.cfg.on_hard_limit != "unload_models":
                # Default "defer_jobs": hold new jobs at the queue (via
                # the admission gate) and leave loaded weights + in-flight
                # work alone. No accuracy cost.
                self._degraded_reason = (
                    f"hard memory limit crossed (used={used_gb:.1f}G >= "
                    f"{self.cfg.hard_gb:.1f}G); holding new jobs in queue "
                    f"until <{self.cfg.resume_gb:.1f}G (in-flight untouched)"
                )
                return
            self._degraded_reason = (
                f"hard memory limit crossed (used={used_gb:.1f}G >= "
                f"{self.cfg.hard_gb:.1f}G); unloading non-active models"
            )
            if self.cfg.on_hard_limit == "unload_models":
                for name, fn in list(self._unload_callbacks.items()):
                    self._pending_unloads.add(name)
                    try:
                        fn()
                    except Exception:
                        log.exception("unload callback %s failed", name)
                    self._pending_unloads.discard(name)
                    self._loaded_providers.discard(name)
            return
        if new == STATE_EMERGENCY:
            self._degraded_reason = (
                f"EMERGENCY memory limit crossed (used={used_gb:.1f}G >= "
                f"{self.cfg.emergency_gb:.1f}G); stopping inference workers"
            )
            if self.cfg.on_emergency_limit == "stop_inference_workers":
                for name, fn in list(self._stop_callbacks.items()):
                    try:
                        fn()
                    except Exception:
                        log.exception("stop callback %s failed", name)

    @staticmethod
    def _psutil_probe() -> tuple[float, float]:
        vm = psutil.virtual_memory()
        total_gb = vm.total / (1024 ** 3)
        used_gb = (vm.total - vm.available) / (1024 ** 3)
        return total_gb, used_gb


# ----------------------------------------------------------------------
# Module-level singleton + helpers
# ----------------------------------------------------------------------

_POLICY: Optional[MemoryPolicy] = None
_POLICY_LOCK = threading.Lock()


def get_policy() -> MemoryPolicy:
    global _POLICY
    if _POLICY is not None:
        return _POLICY
    with _POLICY_LOCK:
        if _POLICY is None:
            cfg = _load_policy_config()
            _POLICY = MemoryPolicy(cfg)
    return _POLICY


def set_policy_for_test(policy: MemoryPolicy) -> None:
    """Tests inject a policy with a stub probe so memory thresholds can
    be exercised without consuming actual RAM."""
    global _POLICY
    with _POLICY_LOCK:
        _POLICY = policy


def _load_policy_config() -> MemoryPolicyConfig:
    try:
        from app.config import load_config
        cfg = load_config()
        gpu = cfg.raw.get("gpu") or {}
        soft = float(gpu.get("soft_memory_limit_gb", 90))
        hard = float(gpu.get("hard_memory_limit_gb", 100))
        return MemoryPolicyConfig(
            soft_gb=soft,
            hard_gb=hard,
            emergency_gb=float(gpu.get("emergency_memory_limit_gb", 110)),
            resume_gb=float(gpu.get("resume_memory_limit_gb", soft)),
            admission_max_wait_sec=float(
                gpu.get("admission_max_wait_sec", 300)),
            poll_interval_sec=float(gpu.get("poll_interval_sec", 5)),
            max_loaded_big_vlms=int(gpu.get("max_loaded_big_vlms", 1)),
            on_hard_limit=str(gpu.get("on_hard_limit", "defer_jobs")),
            on_emergency_limit=str(
                gpu.get("on_emergency_limit", "stop_inference_workers")),
        )
    except Exception:
        log.exception("failed to load policy config; using defaults")
        return MemoryPolicyConfig()
