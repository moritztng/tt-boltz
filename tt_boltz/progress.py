"""Real-time terminal progress display for tt-boltz predict.

Uses Rich Live to show per-device status, stage progress bars,
and a rolling log of completed structures. Communicates with
worker processes via a multiprocessing Queue.
"""

import threading
import time
from dataclasses import dataclass
from queue import Empty

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text

# ── Stage boundaries (cumulative fraction of per-structure work) ──────────
#
#   MSA → Prep → Trunk(recycling) → Diffusion(steps) → Confidence → Save
#
STAGE_START = {
    "loading":    0.00,
    "msa":        0.00,
    "prep":       0.12,
    "trunk":      0.18,   # recycling fills 0.18 → 0.48
    "diffusion":  0.48,   # steps fill     0.48 → 0.88
    "confidence": 0.88,
    "saving":     0.96,
    "done":       1.00,
}
STAGE_END = {
    "msa":        0.12,
    "prep":       0.18,
    "trunk":      0.48,
    "diffusion":  0.88,
    "confidence": 0.96,
    "saving":     1.00,
}

BAR_WIDTH = 20
RECENT_MAX = 8


@dataclass
class DeviceState:
    device_id: int
    name: str = ""
    stage: str = "idle"
    step: int = 0
    total_steps: int = 0
    done: int = 0
    assigned: int = 0


class ProgressDisplay:
    """Drives a Rich Live display from a multiprocessing.Queue of events.

    Events (dicts sent by workers):
        {"dev": int, "event": "init",    "assigned": int}
        {"dev": int, "event": "loading"}
        {"dev": int, "event": "start",   "name": str}
        {"dev": int, "event": "stage",   "stage": str, "step": int, "total": int}
        {"dev": int, "event": "done",    "name": str, "time": float, "status": str}
    """

    def __init__(self, queue, total: int, n_devices: int):
        self.queue = queue
        self.total = total
        self.n_devices = n_devices

        self.devices: dict[int, DeviceState] = {}
        self.completed = 0
        self.failed = 0
        self.recent: list[dict] = []
        self.start_time = time.time()

        self._lock = threading.Lock()
        self._live = None
        self._thread = None
        self._stop = threading.Event()

    # ── lifecycle ─────────────────────────────────────────────────────────

    def start(self):
        self._live = Live(
            self._render(), console=Console(stderr=True),
            refresh_per_second=4, transient=False,
        )
        self._live.start()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._live:
            self._drain()
            self._live.update(self._render())
            self._live.stop()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, *_):
        self.stop()

    # ── background thread ─────────────────────────────────────────────────

    def _loop(self):
        while not self._stop.is_set():
            self._drain()
            if self._live:
                self._live.update(self._render())
            self._stop.wait(0.25)

    def _drain(self):
        while True:
            try:
                ev = self.queue.get_nowait()
            except (Empty, EOFError):
                break
            with self._lock:
                self._handle(ev)

    def _handle(self, ev: dict):
        dev = ev.get("dev", 0)
        kind = ev["event"]

        if kind == "init":
            self.devices[dev] = DeviceState(
                device_id=dev, assigned=ev.get("assigned", 0),
            )
        elif kind == "loading":
            d = self.devices.setdefault(dev, DeviceState(device_id=dev))
            d.stage = "loading"
            d.name = ""
        elif kind == "start":
            d = self.devices.get(dev)
            if d:
                d.name = ev["name"]
                d.stage = "msa"
                d.step = 0
                d.total_steps = 0
        elif kind == "stage":
            d = self.devices.get(dev)
            if d:
                d.stage = ev.get("stage", d.stage)
                d.step = ev.get("step", 0)
                d.total_steps = ev.get("total", 0)
        elif kind == "done":
            self.completed += 1
            if ev.get("status") != "ok":
                self.failed += 1
            d = self.devices.get(dev)
            if d:
                d.done += 1
                d.stage = "idle"
                d.name = ""
            self.recent.append(ev)
            if len(self.recent) > RECENT_MAX:
                self.recent.pop(0)

    # ── rendering ─────────────────────────────────────────────────────────

    @staticmethod
    def _frac(d: DeviceState) -> float:
        s = d.stage
        if s in ("idle", "loading"):
            return 0.0
        if s == "done":
            return 1.0
        base = STAGE_START.get(s, 0.0)
        end = STAGE_END.get(s, base)
        if d.total_steps > 0:
            return base + (end - base) * min(d.step / d.total_steps, 1.0)
        return base

    @staticmethod
    def _bar(frac: float) -> Text:
        filled = int(frac * BAR_WIDTH)
        txt = Text()
        txt.append("█" * filled, style="green")
        txt.append("░" * (BAR_WIDTH - filled), style="bright_black")
        return txt

    @staticmethod
    def _stage_label(d: DeviceState) -> str:
        if d.stage == "idle":
            return "·"
        if d.stage == "loading":
            return "Loading model…"
        if d.stage == "trunk":
            return f"Trunk {d.step}/{d.total_steps}" if d.total_steps else "Trunk"
        if d.stage == "diffusion":
            return f"Diffusion {d.step}/{d.total_steps}" if d.total_steps else "Diffusion"
        return {"msa": "MSA", "prep": "Featurize", "confidence": "Confidence",
                "saving": "Saving", "done": "Done"}.get(d.stage, d.stage)

    def _render(self) -> Group:
        with self._lock:
            return self._build()

    def _build(self) -> Group:
        elapsed = time.time() - self.start_time
        h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)

        # ── header ────────────────────────────────────────────────────────
        pct = f"{self.completed * 100 // self.total}%" if self.total else "–"
        hdr = Text("  ")
        hdr.append("tt-boltz", style="bold cyan")
        hdr.append(f"  {self.n_devices} device{'s' if self.n_devices != 1 else ''}", style="dim")
        hdr.append(f"  {self.completed}/{self.total}", style="bold")
        hdr.append(f" ({pct})", style="dim")
        if self.failed:
            hdr.append(f"  {self.failed} failed", style="red")
        hdr.append(f"  {h:02d}:{m:02d}:{s:02d}", style="dim")
        if self.completed > 0:
            eta = elapsed / self.completed * (self.total - self.completed)
            hdr.append(f"  ~{int(eta)//3600:02d}:{int(eta)%3600//60:02d}:{int(eta)%60:02d} left",
                       style="dim italic")

        sep = Text("  " + "─" * 68, style="bright_black")

        # ── device rows ───────────────────────────────────────────────────
        tbl = Table(show_header=False, box=None, padding=(0, 1),
                    pad_edge=False, expand=False)
        tbl.add_column("d", style="dim", width=9, justify="right")
        tbl.add_column("name", width=18, no_wrap=True)
        tbl.add_column("bar", width=BAR_WIDTH, no_wrap=True)
        tbl.add_column("stage", width=18, no_wrap=True)
        tbl.add_column("cnt", style="dim", width=6, justify="right")

        for dev_id in sorted(self.devices):
            d = self.devices[dev_id]
            frac = self._frac(d)
            active = d.stage not in ("idle", "loading")
            tbl.add_row(
                f"device {dev_id}",
                Text(d.name[:18] if d.name else "·",
                     style="bold" if active else "dim"),
                self._bar(frac),
                Text(self._stage_label(d),
                     style="bold cyan" if active else "dim"),
                f"{d.done}/{d.assigned}" if d.assigned else "",
            )

        # ── recent log ────────────────────────────────────────────────────
        log_lines = []
        for r in self.recent[-self.n_devices:]:
            ln = Text("  ")
            if r.get("status") == "ok":
                ln.append("✓ ", style="green")
                ln.append(r.get("name", "?"))
                ln.append(f"  {r.get('time', 0):.0f}s", style="dim")
            else:
                ln.append("✗ ", style="red")
                ln.append(r.get("name", "?"), style="red")
                ln.append(f"  {r.get('error', 'failed')[:36]}", style="dim red")
            log_lines.append(ln)

        parts = [hdr, Text(""), sep, Text(""), tbl]
        if log_lines:
            parts.append(Text(""))
            parts.append(sep)
            parts.extend(log_lines)

        return Group(*parts)


class NullDisplay:
    """No-op display — used when neither Rich nor debug logging is wanted."""
    def __init__(self, queue, **_kw): self.queue = queue
    def start(self): pass
    def stop(self):
        # drain silently so the queue doesn't block
        while True:
            try: self.queue.get_nowait()
            except Exception: break

    def __enter__(self): return self
    def __exit__(self, *_): self.stop()


class DebugDisplay:
    """Drop-in replacement for ProgressDisplay that prints simple text lines.

    Same start()/stop() interface so callers don't need to branch.
    Runs a background thread that drains the queue, just like ProgressDisplay.
    """

    def __init__(self, queue, **_kw):
        self.queue = queue
        self._stop = threading.Event()
        self._thread = None

    def start(self):
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        self._drain()

    def _loop(self):
        while not self._stop.is_set():
            self._drain()
            self._stop.wait(0.25)

    def _drain(self):
        while True:
            try:
                ev = self.queue.get_nowait()
            except (Empty, EOFError):
                break
            dev = ev.get("dev", 0)
            kind = ev["event"]
            if kind == "loading":
                print(f"[dev {dev}] loading model…", flush=True)
            elif kind == "start":
                print(f"[dev {dev}] {ev.get('name', '?')}", flush=True)
            elif kind == "stage":
                s, step, total = ev.get("stage", ""), ev.get("step", 0), ev.get("total", 0)
                print(f"[dev {dev}]   {s} {step}/{total}" if total else f"[dev {dev}]   {s}", flush=True)
            elif kind == "done":
                sym = "✓" if ev.get("status") == "ok" else "✗"
                print(f"[dev {dev}] {sym} {ev.get('name', '?')} — {ev.get('time', 0):.0f}s", flush=True)


def make_progress_fn(queue, device_id: int):
    """Return a lightweight callback for Boltz2.progress_fn.

    Workers call: model.progress_fn = make_progress_fn(queue, dev_id)
    The model then calls progress_fn("stage", step=N, total=M) at key points.
    """
    def _fn(stage: str, step: int = 0, total: int = 0):
        try:
            queue.put_nowait({
                "dev": device_id, "event": "stage",
                "stage": stage, "step": step, "total": total,
            })
        except Exception:
            pass  # never block the model
    return _fn
