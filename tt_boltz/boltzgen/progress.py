"""Real-time terminal progress display for ``tt-boltz gen``.

One unified layout for single- and multi-card runs: a header plus one row per
device, each with a **total progress bar** (overall pipeline = stages done +
fraction of the current stage) and a **current-stage bar** with its sub-step
(e.g. ``trunk 2/4``, ``diff 340/500``). Single card = one row; N cards = N rows.

Default (normal) display modes:
  * ``SingleDeviceDisplay`` — in-process single-card run; fed by the
    :func:`progress` singleton (tasks emit stage + sub-stage events).
  * ``MultiDeviceDisplay``  — parent of a multi-card run; tails each shard's
    JSONL events (written by :class:`FileDisplay`) and renders all rows.
Both share :func:`render_devices`, so the look is identical.

Debug modes:
  * ``SilentDisplay`` — ``--debug``: all model/task stdout passes through.
  * ``DebugDisplay``  — ``--debug --log``: one text line per event.

The CLI installs the chosen display in ``execute_command``. In normal mode the
model's fd-level chatter is routed to ``<output>/run.log`` via
:func:`redirect_low_level_io` so it can't scroll over the live display.
"""
from __future__ import annotations

import json
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text


BAR_WIDTH = 22

# Map progress "kinds" (emitted by tasks via ``progress(kind, step, total)``)
# to the label shown in the active stage row.
_KIND_LABELS = {
    "batch":     "batch",
    "diffusion": "diff",
    "trunk":     "trunk",
    "msa":       "msa",
    "writing":   "write",
}


@dataclass
class _DeviceProgress:
    """Unified per-device state, shared by the single- and multi-card views.

    A run is one row per device: an overall bar (completed stages), plus the
    current stage's progress — how many proteins/designs of this stage are
    done (batch i/N) and the current protein's sub-phase (trunk i/N, diff
    step/500).
    """
    label: str = "device"
    total_stages: int = 0
    done_stages: int = 0
    stage_idx: int = 0          # 1-based current stage
    stage_name: str = ""
    batch_step: int = 0         # proteins/designs done in the current stage
    batch_total: int = 0
    sub_kind: str = ""          # within-protein phase: trunk / diffusion
    sub_step: int = 0
    sub_total: int = 0
    status: str = "starting"    # starting | running | done | failed
    offset: int = 0             # (multi-device) JSONL read offset


# ── singleton plumbing ──────────────────────────────────────────────────────
#
# Tasks running inside the same Python process emit progress via the module
# function below, which forwards to whichever display the CLI installed.

class _Display:
    """Base interface; default is a no-op so tasks importing :func:`progress`
    work fine when the CLI hasn't installed a display."""
    def start(self): pass
    def stop(self): pass
    def on_stage_start(self, name: str, idx: int, total: int) -> None: pass
    def on_stage_progress(self, kind: str, step: int, total: int) -> None: pass
    def on_stage_done(self, name: str, elapsed: float, ok: bool = True) -> None: pass
    def log(self, message: str) -> None: pass

    def __enter__(self): self.start(); return self
    def __exit__(self, *_): self.stop()


_active: _Display = _Display()


def set_display(display: _Display) -> None:
    """Install ``display`` as the active sink for :func:`progress`."""
    global _active
    _active = display


def progress(kind: str, step: int = 0, total: int = 0) -> None:
    """Emit a sub-stage progress event from a running task.

    Safe to call when no display is installed — falls back to a no-op base.
    """
    _active.on_stage_progress(kind, step, total)


# ── displays ────────────────────────────────────────────────────────────────


def _bar(fraction: float, *, width: int = BAR_WIDTH, done: bool = False) -> Text:
    filled = int(max(0.0, min(1.0, fraction)) * width)
    txt = Text()
    txt.append("█" * filled, style="green" if done else "cyan")
    txt.append("░" * (width - filled), style="bright_black")
    return txt


def _fmt_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def _stage_fraction(st: "_DeviceProgress") -> float:
    # Stage bar = proteins/designs done in this stage (batch i/N); fall back to
    # the within-protein sub-phase if no batch counter has arrived yet.
    if st.status == "done":
        return 1.0
    if st.batch_total > 0:
        return min(1.0, st.batch_step / st.batch_total)
    if st.sub_total > 0:
        return min(1.0, st.sub_step / st.sub_total)
    return 0.0


def _total_fraction(st: "_DeviceProgress") -> float:
    # Completed-stage fraction only — strictly monotonic. The current stage's
    # sub-progress is NOT folded in: one stage emits several sub-phases
    # (batch → trunk → diffusion, repeating per design), each resetting 0→1, so
    # mixing them into the total made it jump backwards (e.g. 50% → 33%). The
    # per-stage bar carries that fine-grained progress instead.
    if st.status == "done":
        return 1.0
    if not st.total_stages:
        return 0.0
    return min(1.0, max(0, st.done_stages) / st.total_stages)


def _device_icon(st: "_DeviceProgress") -> Text:
    return {
        "done": Text("✓", style="green"),
        "failed": Text("✗", style="red"),
        "running": Text("▸", style="cyan bold"),
    }.get(st.status, Text("·", style="bright_black"))


def _stage_name_text(st: "_DeviceProgress") -> Text:
    if st.status == "starting":
        return Text("loading…", style="dim")
    if st.status == "done":
        return Text("done", style="green")
    if st.status == "failed":
        return Text("failed", style="red")
    return Text(st.stage_name or "running", style="bold")


def _stage_step_text(st: "_DeviceProgress") -> Text:
    if st.status in ("starting", "done", "failed"):
        return Text("")
    parts = []
    if st.batch_total > 0:
        parts.append(f"{st.batch_step}/{st.batch_total}")  # proteins done / total
    if st.sub_total > 0:
        kind = _KIND_LABELS.get(st.sub_kind, st.sub_kind or "")
        parts.append(f"{kind} {st.sub_step}/{st.sub_total}".strip())
    return Text("  ".join(parts) if parts else "…",
                style="cyan" if parts else "dim")


def render_devices(states: List["_DeviceProgress"], start: float, *, multi: bool) -> Group:
    """One row per device: [icon] label [total bar] pct │ stage [stage bar] step.

    The single shared layout for both single- and multi-card runs.
    """
    elapsed = time.time() - start if start else 0.0
    n = len(states)

    hdr = Text("  ")
    hdr.append("tt-boltz gen", style="bold cyan")
    if multi:
        finished = sum(1 for s in states if s.status in ("done", "failed"))
        hdr.append(f" · {n} devices", style="bold")
        hdr.append(f"   {finished}/{n} done", style="dim")
    hdr.append(f"   {_fmt_time(elapsed)}", style="dim")

    sep = Text("  " + "─" * 74, style="bright_black")

    tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    tbl.add_column("icon", width=2)
    tbl.add_column("device", width=9, no_wrap=True)
    tbl.add_column("total", width=12, no_wrap=True)
    tbl.add_column("pct", width=4, justify="right", no_wrap=True)
    tbl.add_column("sep", width=1)
    tbl.add_column("stage", width=15, no_wrap=True)
    tbl.add_column("stagebar", width=8, no_wrap=True)
    tbl.add_column("step", width=19, no_wrap=True)

    for st in states:
        done = st.status == "done"
        running = st.status == "running"
        total_frac = _total_fraction(st)
        stage_bar = (Text(" " * 8) if st.status == "starting"
                     else _bar(_stage_fraction(st), width=8, done=done))
        tbl.add_row(
            _device_icon(st),
            Text(st.label, style="bold" if running else ("" if done else "dim")),
            _bar(total_frac, width=12, done=done),
            Text(f"{int(total_frac * 100)}%", style="dim"),
            Text("│", style="bright_black"),
            _stage_name_text(st),
            stage_bar,
            _stage_step_text(st),
        )

    return Group(hdr, sep, Text(""), tbl)


class SilentDisplay(_Display):
    """No UI. Selected with ``--debug``; lets task stdout/stderr pass through."""


class FileDisplay(_Display):
    """Writes structured progress events as JSONL to a file.

    Used by multi-device shard workers: each worker emits machine-readable
    events that the parent process tails to render a combined per-device view
    (see :class:`MultiDeviceDisplay`). The worker's own stdout is suppressed
    (normal mode) or sent to its shard log (debug), so this is the only signal
    the parent needs.
    """

    def __init__(self, path: str):
        self._f = open(path, "a", buffering=1)  # line-buffered

    def _emit(self, event: str, **fields) -> None:
        try:
            self._f.write(json.dumps({"type": event, "t": time.time(), **fields}) + "\n")
        except (ValueError, OSError):
            pass

    def on_stage_start(self, name: str, idx: int, total: int) -> None:
        self._emit("start", name=name, idx=idx, total=total)

    def on_stage_progress(self, kind: str, step: int, total: int) -> None:
        self._emit("progress", kind=kind, step=step, total=total)

    def on_stage_done(self, name: str, elapsed: float, ok: bool = True) -> None:
        self._emit("done", name=name, elapsed=elapsed, ok=ok)

    def stop(self):
        try:
            self._f.close()
        except OSError:
            pass


class DebugDisplay(_Display):
    """Plain-text per-event lines. Selected with ``--debug --log``."""

    def __init__(self, *, stages: List[str]):
        self._idx = 0
        self._total = len(stages)
        self._start = 0.0

    def on_stage_start(self, name: str, idx: int, total: int) -> None:
        self._idx = idx
        self._total = total
        self._start = time.time()
        print(f"\n>>> [{idx}/{total}] {name}", flush=True)

    def on_stage_progress(self, kind: str, step: int, total: int) -> None:
        label = _KIND_LABELS.get(kind, kind)
        if total > 0:
            print(f"    {label} {step}/{total}", flush=True)
        else:
            print(f"    {label}", flush=True)

    def on_stage_done(self, name: str, elapsed: float, ok: bool = True) -> None:
        sym = "✓" if ok else "✗"
        print(f"<<< {sym} {name}   {elapsed:.1f}s", flush=True)

    def log(self, message: str) -> None:
        print(message, flush=True)


class SingleDeviceDisplay(_Display):
    """Live view for an in-process single-card run — one device row with a
    total bar + current-stage bar, fed by the ``progress()`` events. Same
    layout as the multi-card view (just one row)."""

    def __init__(self, *, stages: List[str], device_label: str = "device",
                 console_file=None):
        self._st = _DeviceProgress(label=device_label, total_stages=len(stages))
        self._start = 0.0
        # When the model's fd-level output is redirected to a log, the display
        # is handed a preserved terminal stream so it still renders live.
        self._console = (Console(file=console_file, force_terminal=True)
                         if console_file is not None
                         else Console(stderr=True, force_terminal=True))
        self._live: Optional[Live] = None

    def start(self):
        self._start = time.time()
        # Don't let Live hijack sys.stdout/stderr — redirect_low_level_io already
        # routes the model's chatter to the log; Live grabbing the streams would
        # pull it back onto the terminal and clobber the display.
        self._live = Live(self._build(), console=self._console,
                          refresh_per_second=8, transient=False,
                          redirect_stdout=False, redirect_stderr=False)
        self._live.start()

    def stop(self):
        if self._live is not None:
            self._live.update(self._build())
            self._live.stop()
            self._live = None

    def on_stage_start(self, name: str, idx: int, total: int) -> None:
        st = self._st
        st.status = "running"
        st.stage_idx = idx
        st.total_stages = total or st.total_stages
        st.stage_name = name
        st.done_stages = idx - 1
        st.batch_step = st.batch_total = 0
        st.sub_kind = ""
        st.sub_step = st.sub_total = 0
        self._refresh()

    def on_stage_progress(self, kind: str, step: int, total: int) -> None:
        st = self._st
        if kind == "batch":           # proteins/designs done in this stage
            st.batch_step, st.batch_total = step, total
        else:                         # within-protein phase (trunk / diffusion)
            st.sub_kind, st.sub_step, st.sub_total = kind, step, total
        self._refresh()

    def on_stage_done(self, name: str, elapsed: float, ok: bool = True) -> None:
        st = self._st
        st.done_stages = st.stage_idx
        if not ok:
            st.status = "failed"
        elif st.done_stages >= st.total_stages:
            st.status = "done"
        self._refresh()

    def _refresh(self):
        if self._live is not None:
            self._live.update(self._build())

    def _build(self) -> Group:
        return render_devices([self._st], self._start, multi=False)


# ── multi-device parent view ────────────────────────────────────────────────


class MultiDeviceDisplay:
    """Rich Live view for a multi-card run — one row per device, identical
    layout to the single-card view.

    The parent spawns one worker per card, each writing JSONL progress events
    (via :class:`FileDisplay`) to ``files[device]``. This tails those files and
    renders the combined per-device table.
    """

    def __init__(self, devices: List[int], files: Dict[int, Path], counts: Dict[int, int]):
        self._dev = {d: _DeviceProgress(label=f"device {d}") for d in devices}
        self._files = files
        self._start = time.time()
        self._console = Console(stderr=True, force_terminal=True)
        self._live: Optional[Live] = None

    def start(self):
        self._start = time.time()
        self._live = Live(self._build(), console=self._console,
                          refresh_per_second=6, transient=False,
                          redirect_stdout=False, redirect_stderr=False)
        self._live.start()

    def stop(self):
        if self._live is not None:
            self._ingest()
            self._live.update(self._build())
            self._live.stop()
            self._live = None

    def update(self, proc_status: Dict[int, Optional[int]]) -> None:
        """Refresh from the JSONL files and per-device process exit codes
        (``None`` while running, else the return code)."""
        self._ingest()
        for dev, rc in proc_status.items():
            st = self._dev.get(dev)
            if st is None or rc is None:
                continue
            if rc == 0:
                st.status = "done"
                if st.total_stages:
                    st.done_stages = st.total_stages
            elif st.status != "done":
                st.status = "failed"
        if self._live is not None:
            self._live.update(self._build())

    def _ingest(self) -> None:
        for dev, st in self._dev.items():
            path = self._files[dev]
            if not path.exists():
                continue
            try:
                with open(path) as f:
                    f.seek(st.offset)
                    lines = f.readlines()
                    st.offset = f.tell()
            except OSError:
                continue
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue  # ignore a partially-written trailing line
                kind = ev.get("type")
                if kind == "start":
                    st.status = "running"
                    st.stage_idx = ev.get("idx", st.stage_idx)
                    st.total_stages = ev.get("total", st.total_stages)
                    st.stage_name = ev.get("name", "")
                    st.batch_step = st.batch_total = 0
                    st.sub_kind = ""
                    st.sub_step = st.sub_total = 0
                elif kind == "progress":
                    if ev.get("kind") == "batch":
                        st.batch_step = ev.get("step", 0)
                        st.batch_total = ev.get("total", 0)
                    else:
                        st.sub_kind = ev.get("kind", "")
                        st.sub_step = ev.get("step", 0)
                        st.sub_total = ev.get("total", 0)
                elif kind == "done":
                    st.done_stages = max(st.done_stages, st.stage_idx)
                    if not ev.get("ok", True):
                        st.status = "failed"

    def _build(self) -> Group:
        return render_devices(list(self._dev.values()), self._start, multi=True)


# ── framed header / summary ─────────────────────────────────────────────────


def _kv_table(rows: List[tuple[str, str]]) -> Table:
    tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    tbl.add_column("k", style="dim", no_wrap=True)
    tbl.add_column("v", style="bold")
    for k, v in rows:
        tbl.add_row(k, v)
    return tbl


def print_header(*, specs: List[str], protocol: str, num_designs: int,
                 batches: int, budget: int, devices: List[int],
                 output: str) -> None:
    """One-time framed summary of what this run will do."""
    from rich.panel import Panel

    if len(devices) > 1:
        dev_str = f"{','.join(map(str, devices))}  ({len(devices)} cards)"
    elif devices:
        dev_str = f"{devices[0]}  (1 card)"
    else:
        dev_str = "default"
    rows = [
        ("design", ", ".join(specs)),
        ("protocol", protocol),
        ("designs", f"{num_designs:,}  ({batches} batch{'es' if batches != 1 else ''})"),
        ("keep", f"top {budget} after filtering"),
        ("devices", dev_str),
        ("output", output),
    ]
    Console(stderr=True).print(
        Panel(_kv_table(rows), title="[bold cyan]tt-boltz gen[/]",
              title_align="left", border_style="cyan", padding=(0, 1),
              expand=False)
    )


def print_summary(*, output: str, elapsed: float, ok: bool = True) -> None:
    """Framed final summary; reads the result dir for design/ranked counts."""
    from rich.panel import Panel

    out = Path(output)
    ranked_dir = out / "final_ranked_designs"

    def _csv_rows(path: Path) -> Optional[int]:
        try:
            with open(path) as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return None

    total = _csv_rows(ranked_dir / "all_designs_metrics.csv")
    ranked = _csv_rows(ranked_dir / "final_designs_metrics_30.csv")
    if ranked is None:
        for p in sorted(ranked_dir.glob("final_designs_metrics_*.csv")):
            ranked = _csv_rows(p)
            break

    rows: List[tuple[str, str]] = []
    if total is not None and ranked is not None:
        rows.append(("designs", f"{total:,} scored → {ranked} ranked"))
    elif total is not None:
        rows.append(("designs", f"{total:,} scored"))
    if ranked_dir.exists():
        rows.append(("results", str(ranked_dir)))
    else:
        rows.append(("output", str(out)))

    title = (f"[bold green]✓ done[/] in {_fmt_time(elapsed)}" if ok
             else f"[bold red]✗ failed[/] after {_fmt_time(elapsed)}")
    Console(stderr=True).print(
        Panel(_kv_table(rows), title=title, title_align="left",
              border_style="green" if ok else "red", padding=(0, 1), expand=False)
    )


# ── stdout/stderr suppression ───────────────────────────────────────────────


@contextmanager
def redirect_low_level_io(log_path):
    """Send OS-level fd 1 & 2 to ``log_path`` for the duration, and yield a
    stream still attached to the real terminal.

    Two leak paths must be closed so nothing scrolls over the live display:
    (1) ttnn/tt-metal write straight to file descriptors 1/2, and (2) Python
    code (prints, tqdm, loguru) writes to ``sys.stdout``/``sys.stderr`` — which
    the host CLI may have decoupled from fd 2. So this both dup2's fd 1/2 to the
    log *and* swaps the Python stream objects to it, after dup'ing the current
    terminal for the display to keep.
    """
    import sys
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    term = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1,
                     encoding=getattr(sys.stderr, "encoding", None) or "utf-8")
    logf = open(log_path, "w", buffering=1)
    saved_out_fd, saved_err_fd = os.dup(1), os.dup(2)
    saved_stdout, saved_stderr = sys.stdout, sys.stderr
    os.dup2(logf.fileno(), 1)
    os.dup2(logf.fileno(), 2)
    sys.stdout = logf
    sys.stderr = logf
    try:
        yield term
    finally:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
        except (ValueError, OSError):
            pass
        sys.stdout = saved_stdout
        sys.stderr = saved_stderr
        os.dup2(saved_out_fd, 1)
        os.dup2(saved_err_fd, 2)
        os.close(saved_out_fd)
        os.close(saved_err_fd)
        try:
            term.close()
        except OSError:
            pass
        logf.close()


@contextmanager
def suppress_output(active: bool, streams: tuple = ("stdout", "stderr")):
    """Redirect the named ``streams`` to ``os.devnull`` when ``active``.

    Used by the CLI to keep BoltzGen's internal print statements from
    fighting with the Rich display. Suppressing only ``stdout`` lets a
    stderr-based Rich spinner/status remain visible during a quiet phase.
    """
    if not active:
        yield
        return
    import sys
    saved = {name: getattr(sys, name) for name in streams}
    devnull = open(os.devnull, "w")
    for name in streams:
        setattr(sys, name, devnull)
    try:
        yield
    finally:
        for name, stream in saved.items():
            setattr(sys, name, stream)
        devnull.close()
