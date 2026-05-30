"""Terminal progress for ``tt-boltz gen`` — one row per device.

Each row shows overall pipeline progress (completed stages) plus the current
stage: how many proteins/designs are done (``i/N``) and the current protein's
sub-phase (``trunk i/N``, ``diff step/total``).

A single-card run feeds the view in-process: tasks call :func:`progress` and
the CLI calls the reporter's ``stage_start``/``stage_done``. A multi-card run
spawns one worker per card; each worker writes the same events as JSONL
(:class:`FileReporter`) and the parent tails them into one :class:`View`.

The only thing model code touches is :func:`progress`; everything else is the
CLI's.
"""
from __future__ import annotations

import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

_LABEL = {"trunk": "trunk", "diffusion": "diff", "batch": "batch", "msa": "msa"}
_ICON = {"run": ("▸", "cyan bold"), "ok": ("✓", "green"), "fail": ("✗", "red")}


# ── per-device state ─────────────────────────────────────────────────────────


@dataclass
class Device:
    label: str
    stages: int = 0          # total pipeline stages
    done: int = 0            # completed stages
    cur: int = 0             # current stage (1-based)
    name: str = ""           # current stage name
    pi: int = 0              # proteins done in the current stage
    pn: int = 0              # proteins total in the current stage
    kind: str = ""           # current sub-phase (trunk / diffusion)
    ki: int = 0
    kn: int = 0
    status: str = "wait"     # wait | run | ok | fail
    pos: int = 0             # JSONL byte offset (multi-card tailing)


def _apply(d: Device, ev: dict) -> None:
    """Apply one progress event — the single source of truth for transitions."""
    t = ev["t"]
    if t == "start":
        d.status, d.name = "run", ev["name"]
        d.stages, d.cur, d.done = ev["total"], ev["idx"], ev["idx"] - 1
        d.pi = d.pn = d.ki = d.kn = 0
        d.kind = ""
    elif t == "step":
        if ev["kind"] == "batch":
            d.pi, d.pn = ev["n"], ev["total"]
        else:
            d.kind, d.ki, d.kn = ev["kind"], ev["n"], ev["total"]
    elif t == "done":
        d.done = d.cur
        if not ev.get("ok", True):
            d.status = "fail"
        elif d.done >= d.stages:
            d.status = "ok"


def finish(d: Device, ok: bool) -> None:
    """Mark a device finished from the outside (e.g. a worker's exit code)."""
    if ok:
        d.status, d.done = "ok", (d.stages or d.done)
    elif d.status != "ok":
        d.status = "fail"


# ── rendering ────────────────────────────────────────────────────────────────


def _bar(frac: float, width: int, ok: bool = False) -> Text:
    n = int(max(0.0, min(1.0, frac)) * width)
    t = Text()
    t.append("█" * n, style="green" if ok else "cyan")
    t.append("░" * (width - n), style="bright_black")
    return t


def _fmt(sec: float) -> str:
    sec = max(0, int(sec))
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:d}:{s:02d}"


def render(devices: list[Device], started: float, multi: bool) -> Group:
    head = Text("  ")
    head.append("tt-boltz gen", style="bold cyan")
    if multi:
        ndone = sum(d.status in ("ok", "fail") for d in devices)
        head.append(f" · {len(devices)} devices  {ndone}/{len(devices)} done", style="dim")
    head.append(f"   {_fmt(time.time() - started)}", style="dim")

    tbl = Table(show_header=False, box=None, padding=(0, 1), pad_edge=False)
    # Last column (stage detail) is flexible: when the terminal is narrow, Rich
    # shrinks it rather than cropping the percentage into "10…".
    for w in (2, 9, 12, 4, 1, 15, 8, None):
        tbl.add_column(width=w, no_wrap=True)

    for d in devices:
        ok = d.status == "ok"
        total = d.done / d.stages if d.stages else 0.0
        # Stage bar tracks the CURRENT protein's progress (its trunk/diffusion),
        # so it fills per protein and restarts for the next one. Fall back to the
        # proteins-done count for stages with no per-protein sub-steps.
        stage_frac = d.ki / d.kn if d.kn else (d.pi / d.pn if d.pn else 0.0)
        icon, istyle = _ICON.get(d.status, ("·", "bright_black"))
        name = {"run": d.name, "ok": "done", "fail": "failed"}.get(d.status, "")
        detail = []
        if d.pn:
            detail.append(f"{d.pi}/{d.pn}")
        if d.kn:
            detail.append(f"{_LABEL.get(d.kind, d.kind)} {d.ki}/{d.kn}")
        tbl.add_row(
            Text(icon, style=istyle),
            Text(d.label, style="bold" if d.status == "run" else ("" if ok else "dim")),
            _bar(total, 12, ok),
            Text(f"{int(total * 100)}%", style="dim", justify="right"),
            Text("│", style="bright_black"),
            Text(name, style=istyle if d.status != "run" else "bold"),
            _bar(stage_frac, 8, ok) if d.status == "run" and (d.kn or d.pn)
            else Text(" " * 8),
            Text("  ".join(detail), style="cyan"),
        )
    return Group(Text(""), head, Text("  " + "─" * 74, style="bright_black"),
                 Text(""), tbl)


class View:
    """A Rich Live over a list of :class:`Device` rows."""

    def __init__(self, devices: list[Device]):
        self.devices = devices
        self._t0 = time.time()
        self._live: Live | None = None

    @contextmanager
    def show(self, file=None):
        self._t0 = time.time()
        console = Console(file=file, stderr=file is None, force_terminal=True)
        # redirect_stdout/stderr=False: the model's output is already routed to a
        # log; letting Live grab the streams would pull it back onto the screen.
        self._live = Live(self._group(), console=console, refresh_per_second=8,
                          redirect_stdout=False, redirect_stderr=False)
        with self._live:
            yield self
        self._live = None

    def refresh(self) -> None:
        if self._live is not None:
            self._live.update(self._group())

    def _group(self) -> Group:
        return render(self.devices, self._t0, multi=len(self.devices) > 1)


def tail(d: Device, path: Path) -> None:
    """Read new JSONL events from a worker's progress file into its state."""
    if not path or not path.exists():
        return
    try:
        with open(path) as f:
            f.seek(d.pos)
            lines = f.readlines()
            d.pos = f.tell()
    except OSError:
        return
    for line in lines:
        line = line.strip()
        if line:
            try:
                _apply(d, json.loads(line))
            except ValueError:
                pass  # ignore a partially-written trailing line


# ── reporters (event sinks) ──────────────────────────────────────────────────


class Reporter:
    """Sink for pipeline progress. Base is a no-op (``--debug`` raw / headless)."""
    def stage_start(self, name: str, idx: int, total: int) -> None: ...
    def step(self, kind: str, n: int, total: int) -> None: ...
    def stage_done(self, ok: bool = True) -> None: ...
    def __enter__(self): return self
    def __exit__(self, *exc): ...


_active = Reporter()


def set_reporter(r: Reporter) -> None:
    global _active
    _active = r


def progress(kind: str, step: int = 0, total: int = 0) -> None:
    """Emit a sub-stage event from a running task (safe when no reporter is set)."""
    _active.step(kind, step, total)


class DebugReporter(Reporter):
    """``--debug --log``: one plain text line per event."""
    def stage_start(self, name, idx, total): print(f"\n>>> [{idx}/{total}] {name}", flush=True)
    def step(self, kind, n, total): print(f"    {_LABEL.get(kind, kind)} {n}/{total}", flush=True)
    def stage_done(self, ok=True): print(f"<<< {'✓' if ok else '✗'}", flush=True)


class FileReporter(Reporter):
    """Multi-card worker: write events as JSONL for the parent to tail."""
    def __init__(self, path):
        self._f = open(path, "a", buffering=1)
    def stage_start(self, name, idx, total): self._w(t="start", name=name, idx=idx, total=total)
    def step(self, kind, n, total): self._w(t="step", kind=kind, n=n, total=total)
    def stage_done(self, ok=True): self._w(t="done", ok=ok)
    def _w(self, **ev):
        try:
            self._f.write(json.dumps(ev) + "\n")
        except (ValueError, OSError):
            pass
    def __exit__(self, *exc):
        try:
            self._f.close()
        except OSError:
            pass


class LiveReporter(Reporter):
    """Single-card: drive a one-row :class:`View` in-process."""
    def __init__(self, view: View):
        self._view, self._dev = view, view.devices[0]
    def stage_start(self, name, idx, total):
        _apply(self._dev, dict(t="start", name=name, idx=idx, total=total)); self._view.refresh()
    def step(self, kind, n, total):
        _apply(self._dev, dict(t="step", kind=kind, n=n, total=total)); self._view.refresh()
    def stage_done(self, ok=True):
        _apply(self._dev, dict(t="done", ok=ok)); self._view.refresh()


# ── output capture & framed panels ───────────────────────────────────────────


@contextmanager
def redirect_output(log_path):
    """Send fd 1/2 *and* ``sys.stdout``/``stderr`` to ``log_path`` and yield a
    handle to the real terminal (for the live display to keep).

    Both paths must be closed: ttnn/tt-metal write to the fds directly, while
    tqdm/prints use the Python streams (which the host CLI may decouple from fd 2).
    """
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()
    term = os.fdopen(os.dup(sys.stderr.fileno()), "w", buffering=1,
                     encoding=getattr(sys.stderr, "encoding", None) or "utf-8")
    log = open(log_path, "w", buffering=1)
    saved_fds = (os.dup(1), os.dup(2))
    saved_streams = (sys.stdout, sys.stderr)
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    sys.stdout = sys.stderr = log
    try:
        yield term
    finally:
        sys.stdout, sys.stderr = saved_streams
        os.dup2(saved_fds[0], 1)
        os.dup2(saved_fds[1], 2)
        os.close(saved_fds[0])
        os.close(saved_fds[1])
        for f in (term, log):
            try:
                f.close()
            except OSError:
                pass


def _panel(title: str, rows: list[tuple[str, str]], color: str) -> None:
    grid = Table.grid(padding=(0, 1))
    grid.add_column(style="dim", no_wrap=True)
    grid.add_column(style="bold")
    for k, v in rows:
        grid.add_row(k, v)
    console = Console(stderr=True)
    console.print()  # breathing room above the box
    console.print(
        Panel(grid, title=title, title_align="left", border_style=color,
              padding=(0, 1), expand=False))


def header(*, specs, protocol, num_designs, batches, budget, devices, output) -> None:
    if len(devices) > 1:
        dev = f"{','.join(map(str, devices))}  ({len(devices)} cards)"
    elif devices:
        dev = str(devices[0])
    else:
        dev = "default"
    _panel("[bold cyan]tt-boltz gen[/]", [
        ("design", ", ".join(specs)),
        ("protocol", protocol),
        ("designs", f"{num_designs:,}  ({batches} batch{'es' if batches != 1 else ''})"),
        ("keep", f"top {budget} after filtering"),
        ("devices", dev),
        ("output", output),
    ], "cyan")


def summary(*, output, elapsed, ok=True) -> None:
    out = Path(output)
    ranked = out / "final_ranked_designs"

    def _rows(path):
        try:
            with open(path) as f:
                return max(0, sum(1 for _ in f) - 1)
        except OSError:
            return None

    total = _rows(ranked / "all_designs_metrics.csv")
    kept = next((_rows(p) for p in sorted(ranked.glob("final_designs_metrics_*.csv"))), None)
    rows = []
    if total is not None:
        rows.append(("designs", f"{total:,} scored → {kept} kept" if kept is not None
                     else f"{total:,} scored"))
    rows.append(("results", str(ranked)) if ranked.exists() else ("output", str(out)))
    _panel(f"[bold green]✓ done[/] in {_fmt(elapsed)}" if ok
           else f"[bold red]✗ failed[/] after {_fmt(elapsed)}",
           rows, "green" if ok else "red")
