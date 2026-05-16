"""Optional device-synchronized profiler for tt-boltz.

Enabled by setting ``TT_BOLTZ_PROFILE=1``. When enabled, this module
patches the public ``forward`` methods on the tt-boltz Tenstorrent
wrappers (and a few Boltz2 entry points) so each call is bracketed by
``ttnn.synchronize_device`` and per-call wall-clock time is accumulated.

A summary is printed on process exit. Stats are scoped per-process so
both the launcher and worker subprocesses are profiled.
"""

from __future__ import annotations

import atexit
import functools
import os
import time
from typing import Callable

_RAW = os.environ.get("TT_BOLTZ_PROFILE", "")
ENABLED = bool(_RAW)
DETAILED = _RAW.lower() in ("2", "detailed", "deep", "all")

_stats: dict[str, list[float]] = {}
_phase_stack: list[str] = []
_device = None


def _get_device():
    global _device
    if _device is not None:
        return _device
    try:
        from tt_boltz.tenstorrent import _device as cached
    except Exception:
        return None
    _device = cached
    return _device


def _sync() -> None:
    dev = _get_device()
    if dev is None:
        return
    try:
        import ttnn

        ttnn.synchronize_device(dev)
    except Exception:
        pass


def _record(name: str, dt_ms: float) -> None:
    entry = _stats.setdefault(name, [0, 0.0, 0.0, 1e18])
    entry[0] += 1
    entry[1] += dt_ms
    if dt_ms > entry[2]:
        entry[2] = dt_ms
    if dt_ms < entry[3]:
        entry[3] = dt_ms


def timed(name: str) -> Callable:
    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if not ENABLED:
                return fn(*args, **kwargs)
            _sync()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            _sync()
            _record(name, (time.perf_counter() - t0) * 1000.0)
            return out

        return wrapper

    return decorator


def dump_stats() -> None:
    if not _stats:
        return
    items = sorted(_stats.items(), key=lambda x: x[1][1], reverse=True)
    total_ms = sum(v[1] for _, v in items)
    pid = os.getpid()
    print(f"\n========== TT-BOLTZ PROFILE (pid={pid}) ==========", flush=True)
    print(f"  {'name':45s} {'calls':>6} {'total_s':>9} {'mean_ms':>9} {'min_ms':>9} {'max_ms':>9} {'pct':>6}", flush=True)
    for name, (calls, tot, mx, mn) in items:
        mean = tot / max(calls, 1)
        pct = tot / max(total_ms, 1e-9) * 100.0
        print(
            f"  {name:45s} {calls:6d} {tot/1000:9.3f} {mean:9.2f} {mn:9.2f} {mx:9.2f} {pct:5.1f}%",
            flush=True,
        )
    print(f"  {'TOTAL':45s} {'':6s} {total_ms/1000:9.3f}", flush=True)
    print(f"=====================================================\n", flush=True)


_INSTALLED = False


def install() -> None:
    """Install timed wrappers around hot forward methods.

    Safe to call multiple times; no-op when ``TT_BOLTZ_PROFILE`` is unset.
    Must be called after the worker has applied its TT environment so that
    importing ``tenstorrent`` (and ``ttnn``) sees ``TT_VISIBLE_DEVICES``.
    """
    global _INSTALLED
    if not ENABLED or _INSTALLED:
        return
    try:
        from tt_boltz import tenstorrent as tt
    except Exception:
        return

    tt.PairformerModule.forward = timed("PairformerModule.forward")(tt.PairformerModule.forward)
    tt.MSAModule.forward = timed("MSAModule.forward")(tt.MSAModule.forward)
    tt.DiffusionModule.forward = timed("DiffusionModule.forward")(tt.DiffusionModule.forward)

    try:
        from tt_boltz import boltz2 as b2
    except Exception:
        b2 = None
    if b2 is not None and hasattr(b2, "Boltz2"):
        b2.Boltz2.forward = timed("Boltz2.forward")(b2.Boltz2.forward)

    if DETAILED:
        _install_detailed(tt)

    atexit.register(dump_stats)
    _INSTALLED = True


def _install_detailed(tt) -> None:
    """Wrap the ttnn-level building blocks with per-call timers.

    Adds sync overhead per micro-op call; use sparingly (one or two runs).
    """
    targets = [
        ("TriangleMultiplication", "TriangleMultiplication.__call__"),
        ("TriangleAttention", "TriangleAttention.__call__"),
        ("AttentionPairBias", "AttentionPairBias.__call__"),
        ("Transition", "Transition.__call__"),
        ("PairWeightedAveraging", "PairWeightedAveraging.__call__"),
        ("OuterProductMean", "OuterProductMean.__call__"),
        ("AdaLN", "AdaLN.__call__"),
        ("ConditionedTransitionBlock", "ConditionedTransitionBlock.__call__"),
        ("DiffusionTransformerLayer", "DiffusionTransformerLayer.__call__"),
        ("PairformerLayer", "PairformerLayer.__call__"),
        ("MSALayer", "MSALayer.__call__"),
    ]
    for cls_name, label in targets:
        cls = getattr(tt, cls_name, None)
        if cls is None or not hasattr(cls, "__call__"):
            continue
        cls.__call__ = timed(label)(cls.__call__)
