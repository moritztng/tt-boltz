"""Real-time terminal progress display for ``tt-boltz gen``.

Three display modes, matching tt-boltz predict's pattern:

  * ``RichDisplay``  — default. Rich Live with per-stage rows, status icons,
    progress bars, elapsed time, and ETA. Looks clean while suppressing the
    underlying BoltzGen task output.
  * ``DebugDisplay`` — text lines per event. Selected with ``--debug --log``.
  * ``SilentDisplay`` — no UI; all model/task stdout passes through unfiltered.
    Selected with ``--debug``.

BoltzGen runs everything in-process, so unlike tt-boltz predict there's no
multiprocessing queue — tasks call into a module-level singleton via
:func:`progress` to emit step-level events. The CLI installs the chosen
display in ``execute_command`` and routes lifecycle calls to it.
"""
from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import List, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.table import Table
from rich.text import Text


BAR_WIDTH = 22

# Map progress "kinds" (emitted by tasks via ``progress(kind, step, total)``)
# to the label shown in the active stage row.
_KIND_LABELS = {
    "batch":     "Batch",
    "diffusion": "Diffusion",
    "trunk":     "Trunk",
    "msa":       "MSA",
    "writing":   "Writing",
}


@dataclass
class _StageState:
    name: str
    status: str = "pending"  # pending | running | done | failed
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    elapsed: float = 0.0
    # Sub-progress within an active stage.
    sub_kind: str = ""
    sub_step: int = 0
    sub_total: int = 0


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


class SilentDisplay(_Display):
    """No UI. Selected with ``--debug``; lets task stdout/stderr pass through."""


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


class RichDisplay(_Display):
    """Rich Live UI — default mode. Renders a header + per-stage table."""

    def __init__(self, *, stages: List[str]):
        self._stages: List[_StageState] = [_StageState(name=s) for s in stages]
        self._active_idx: Optional[int] = None
        self._start = 0.0
        self._console = Console(stderr=True)
        self._live: Optional[Live] = None
        self._log_lines: List[Text] = []

    # ── lifecycle ────────────────────────────────────────────────────────
    def start(self):
        self._start = time.time()
        self._live = Live(
            self._build(),
            console=self._console,
            refresh_per_second=8,
            transient=False,
        )
        self._live.start()

    def stop(self):
        if self._live is not None:
            self._live.update(self._build())
            self._live.stop()
            self._live = None

    # ── events ───────────────────────────────────────────────────────────
    def on_stage_start(self, name: str, idx: int, total: int) -> None:
        # ``idx`` is 1-based.
        i = idx - 1
        if 0 <= i < len(self._stages):
            s = self._stages[i]
            s.status = "running"
            s.started_at = time.time()
            s.sub_kind = ""
            s.sub_step = 0
            s.sub_total = 0
            self._active_idx = i
            self._refresh()

    def on_stage_progress(self, kind: str, step: int, total: int) -> None:
        if self._active_idx is None:
            return
        s = self._stages[self._active_idx]
        s.sub_kind = kind
        s.sub_step = step
        s.sub_total = total
        self._refresh()

    def on_stage_done(self, name: str, elapsed: float, ok: bool = True) -> None:
        if self._active_idx is None:
            return
        s = self._stages[self._active_idx]
        s.status = "done" if ok else "failed"
        s.finished_at = time.time()
        s.elapsed = elapsed
        self._active_idx = None
        self._refresh()

    def log(self, message: str) -> None:
        self._log_lines.append(Text(f"  {message}", style="dim"))
        if len(self._log_lines) > 5:
            self._log_lines = self._log_lines[-5:]
        self._refresh()

    # ── rendering ────────────────────────────────────────────────────────
    def _refresh(self):
        if self._live is not None:
            self._live.update(self._build())

    @staticmethod
    def _bar(fraction: float, *, done: bool = False) -> Text:
        filled = int(max(0.0, min(1.0, fraction)) * BAR_WIDTH)
        txt = Text()
        style = "green" if done else "cyan"
        txt.append("█" * filled, style=style)
        txt.append("░" * (BAR_WIDTH - filled), style="bright_black")
        return txt

    @staticmethod
    def _fmt_time(seconds: float) -> str:
        seconds = max(0, int(seconds))
        h, rem = divmod(seconds, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def _stage_fraction(self, s: _StageState) -> float:
        if s.status == "done":
            return 1.0
        if s.status in ("pending", "failed"):
            return 0.0
        if s.sub_total > 0:
            return s.sub_step / s.sub_total
        # Running but no inner-progress yet — leave bar empty.
        return 0.0

    def _stage_label(self, s: _StageState) -> str:
        if s.status == "pending":
            return "pending"
        if s.status == "failed":
            return "failed"
        if s.status == "done":
            return f"done    {self._fmt_time(s.elapsed)}"
        # running
        label = _KIND_LABELS.get(s.sub_kind, "running")
        if s.sub_total > 0:
            return f"{label} {s.sub_step}/{s.sub_total}"
        return label

    def _icon(self, s: _StageState) -> Text:
        if s.status == "done":
            return Text("✓", style="green")
        if s.status == "failed":
            return Text("✗", style="red")
        if s.status == "running":
            return Text("▸", style="cyan bold")
        return Text("·", style="bright_black")

    def _build(self) -> Group:
        elapsed = time.time() - self._start if self._start else 0.0
        done = sum(1 for s in self._stages if s.status == "done")
        total = len(self._stages)
        pct = (done * 100 // total) if total else 0

        hdr = Text("  ")
        hdr.append("tt-boltz gen", style="bold cyan")
        hdr.append(f"   {done}/{total} stages", style="bold")
        hdr.append(f" ({pct}%)", style="dim")
        hdr.append(f"   {self._fmt_time(elapsed)}", style="dim")
        if done > 0 and done < total:
            eta = elapsed / done * (total - done)
            hdr.append(f"   ~{self._fmt_time(eta)} left", style="dim italic")

        sep = Text("  " + "─" * 68, style="bright_black")

        tbl = Table(show_header=False, box=None, padding=(0, 1),
                    pad_edge=False, expand=False)
        tbl.add_column("icon", width=2)
        tbl.add_column("name", width=18, no_wrap=True)
        tbl.add_column("bar", width=BAR_WIDTH, no_wrap=True)
        tbl.add_column("status", no_wrap=True)

        for s in self._stages:
            active = s.status == "running"
            done_s = s.status == "done"
            tbl.add_row(
                self._icon(s),
                Text(s.name,
                     style="bold" if active else ("dim" if not done_s else "")),
                self._bar(self._stage_fraction(s), done=done_s),
                Text(self._stage_label(s),
                     style="bold cyan" if active else "dim"),
            )

        parts = [hdr, sep, Text(""), tbl]
        if self._log_lines:
            parts.append(Text(""))
            parts.append(sep)
            parts.extend(self._log_lines)
        return Group(*parts)


# ── stdout/stderr suppression ───────────────────────────────────────────────


@contextmanager
def suppress_output(active: bool):
    """Redirect stdout+stderr to ``os.devnull`` when ``active``.

    Used by the CLI to keep BoltzGen's internal print statements from
    fighting with the Rich Live display. Yields the original streams so
    callers can still emit to the real terminal via the display.
    """
    if not active:
        yield
        return
    import os
    import sys
    real_out, real_err = sys.stdout, sys.stderr
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    try:
        yield
    finally:
        sys.stdout = real_out
        sys.stderr = real_err
        devnull.close()
