"""Worker pool for the quad-card demo.

Owns the per-card subprocesses, a single drain thread that pulls events from
the shared ``multiprocessing.Queue``, a supervisor thread that respawns dead
workers with exponential backoff, and the SSE subscriber fan-out used by the
Flask app.

Threading model:

* main thread:     Flask handlers (multi-threaded via Werkzeug).
* drain thread:    sole consumer of ``mp.Queue`` → updates snapshots + fans
                   events out to subscribers.
* supervisor:      polls process liveness, respawns crashed workers.
* worker procs:    one per TT card, isolated by ``TT_VISIBLE_DEVICES``.

Subscribers each get a bounded ``queue.Queue``. If a subscriber falls behind
beyond the queue size, we drop the subscription rather than blocking the
broadcast or growing memory without bound; the browser's ``EventSource`` will
auto-reconnect and pick up a fresh snapshot.
"""

from __future__ import annotations

import importlib.util
import logging
import multiprocessing as mp
import os
import queue
import signal
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

from tt_boltz.runtime import detect_tenstorrent_devices

from demo_quad import worker as worker_module
from demo_quad.complexes import Complex


log = logging.getLogger("quad.pool")

# Tunables that affect production behavior. Kept at module scope so they're
# easy to find and patch in tests.
EVENT_QUEUE_SIZE = 4096           # main mp.Queue between workers and drain
SUBSCRIBER_QUEUE_SIZE = 1024      # per-SSE-client queue
SUPERVISOR_POLL_SECONDS = 2.0
RESTART_BACKOFF_BASE_SECONDS = 2.0
RESTART_BACKOFF_MAX_SECONDS = 30.0
SHUTDOWN_GRACE_SECONDS = 15.0     # SIGTERM → wait → SIGKILL


# Lone P300 chips need a custom 1×1 fabric mesh graph descriptor; without it,
# ttnn aborts at device open with TT_FATAL on tt_cluster.cpp. These helpers
# stay ttnn-free so the parent process never opens a TT device just to figure
# out which descriptor to point each worker at.

_P300_SUBSYSTEM_IDS = {"0x0044", "0x0045", "0x0046"}


def _detect_p300_devices() -> set[int]:
    devices: set[int] = set()
    for entry in Path("/sys/class/tenstorrent").glob("tenstorrent!*"):
        try:
            dev_id = int(entry.name.rsplit("!", 1)[1])
            sub = (entry / "device" / "subsystem_device").read_text().strip().lower()
        except Exception:
            continue
        if sub in _P300_SUBSYSTEM_IDS:
            devices.add(dev_id)
    return devices


def _find_p300_mesh_graph_descriptor() -> str | None:
    spec = importlib.util.find_spec("ttnn")
    if spec is None or not spec.submodule_search_locations:
        return None
    path = (
        Path(next(iter(spec.submodule_search_locations)))
        / "tt_metal" / "fabric" / "mesh_graph_descriptors"
        / "p150_mesh_graph_descriptor.textproto"
    )
    return str(path) if path.is_file() else None


# ── Snapshot ─────────────────────────────────────────────────────────────────


@dataclass
class CardSnapshot:
    """Latest known state for one card. Used for ``/status`` and snapshot
    replay on SSE reconnect — never mutated by anyone except the drain thread
    and the supervisor."""

    card_id: int
    device_id: int
    label: str
    alive: bool = False
    stage: str = "starting"       # high-level UI stage label
    protein: str = ""
    step: int = 0
    total: int = 0
    last_complete: dict | None = None
    error: str | None = None
    updated_at: float = field(default_factory=time.time)


@dataclass
class Card:
    """Bookkeeping for one worker subprocess."""

    snapshot: CardSnapshot
    mesh_graph_descriptor: str | None
    process: mp.Process | None = None
    consecutive_failures: int = 0
    next_restart_at: float = 0.0


# ── Pool ─────────────────────────────────────────────────────────────────────


class CardPool:
    """Owns the worker subprocesses and the SSE broadcast plane."""

    def __init__(
        self,
        complexes: list[Complex],
        *,
        fast: bool = True,
        sampling_steps: int = 200,
        linger_seconds: float = 2.0,
        log_dir: str | None = None,
    ):
        self._ctx = mp.get_context("spawn")
        device_ids = detect_tenstorrent_devices(None, 0, 32)
        if not device_ids:
            raise RuntimeError("No Tenstorrent devices detected under /dev/tenstorrent/")

        self.events: mp.Queue = self._ctx.Queue(maxsize=EVENT_QUEUE_SIZE)
        self.play: mp.Event = self._ctx.Event()
        self.shutdown: mp.Event = self._ctx.Event()

        self.complexes = complexes
        self.fast = fast
        self.sampling_steps = sampling_steps
        self.linger_seconds = linger_seconds
        self.log_dir = log_dir

        # Shared atomic counter for work dispatch. Pre-seeded so that the
        # first ``len(device_ids)`` iterations land cleanly on card N →
        # complex N — workers consume index N directly via ``start_offset``,
        # then this counter takes over from index ``len(device_ids)``.
        self._dispatch = self._ctx.Value("i", len(device_ids))

        p300 = _detect_p300_devices()
        mgd = _find_p300_mesh_graph_descriptor() if p300 else None

        self.cards: dict[int, Card] = {}
        for i, d in enumerate(device_ids):
            self.cards[i] = Card(
                snapshot=CardSnapshot(card_id=i, device_id=d, label=f"Processor {d}"),
                mesh_graph_descriptor=(mgd if d in p300 else None),
            )

        # SSE fan-out
        self._subscribers: set[queue.Queue] = set()
        self._subscribers_lock = threading.Lock()

        self._drain_thread = threading.Thread(
            target=self._drain_loop, name="quad-drain", daemon=True,
        )
        self._supervisor_thread = threading.Thread(
            target=self._supervisor_loop, name="quad-supervisor", daemon=True,
        )

    # ── lifecycle ───────────────────────────────────────────────────────

    def start(self) -> None:
        """Spawn the initial worker for every card, then start the helpers.

        Each initial worker is handed ``start_offset=card_id`` so the very
        first round is the deterministic processor↔complex mapping
        (W0→C0, W1→C1, …). Subsequent iterations and any respawned workers
        pull from the shared dispatcher.
        """
        for card in self.cards.values():
            self._spawn(card, start_offset=card.snapshot.card_id)
        self._drain_thread.start()
        self._supervisor_thread.start()
        log.info("pool started with %d cards, %d complexes in rotation",
                 len(self.cards), len(self.complexes))

    def stop(self) -> None:
        """Stop everything gracefully, then forcefully if anyone holds out.

        Idempotent. Safe to call from a signal handler.
        """
        if self.shutdown.is_set():
            return
        self.shutdown.set()
        self.play.set()  # wake any worker waiting on play

        # SIGTERM each worker so its installed handler triggers a graceful
        # loop exit. Then wait for the join with a deadline.
        for card in self.cards.values():
            proc = card.process
            if proc and proc.is_alive():
                try:
                    proc.terminate()
                except Exception:
                    pass

        deadline = time.monotonic() + SHUTDOWN_GRACE_SECONDS
        for card in self.cards.values():
            proc = card.process
            if not proc:
                continue
            remaining = max(0.0, deadline - time.monotonic())
            proc.join(timeout=remaining)
            if proc.is_alive():
                log.warning("card %d worker did not exit in %.0fs, SIGKILL",
                            card.snapshot.card_id, SHUTDOWN_GRACE_SECONDS)
                try:
                    os.kill(proc.pid, signal.SIGKILL)
                    proc.join(timeout=2.0)
                except Exception:
                    pass

        log.info("pool stopped")

    # ── control ─────────────────────────────────────────────────────────

    def play_workers(self) -> None:
        self.play.set()
        self._broadcast({"event": "control", "playing": True})

    def pause_workers(self) -> None:
        self.play.clear()
        self._broadcast({"event": "control", "playing": False})

    def status(self) -> dict:
        return {
            "playing": self.play.is_set(),
            "complexes": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "seq_len": c.seq_len,
                    "chain_count": c.chain_count,
                    "ligand_ccds": list(c.ligand_ccds),
                    "pdb": c.pdb,
                }
                for c in self.complexes
            ],
            "cards": [self._snapshot_to_dict(c.snapshot) for c in self.cards.values()],
        }

    # ── subscribers ─────────────────────────────────────────────────────

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=SUBSCRIBER_QUEUE_SIZE)
        with self._subscribers_lock:
            self._subscribers.add(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._subscribers_lock:
            self._subscribers.discard(q)

    def _broadcast(self, event: dict) -> None:
        """Send ``event`` to every subscriber. Slow subscribers get dropped."""
        with self._subscribers_lock:
            targets = list(self._subscribers)
        for q in targets:
            try:
                q.put_nowait(event)
            except queue.Full:
                # The browser will auto-reconnect and request a fresh
                # snapshot; that's a cleaner recovery than letting a frozen
                # client wedge our broadcast.
                self.unsubscribe(q)
                log.warning("dropped slow SSE subscriber")

    # ── drain ───────────────────────────────────────────────────────────

    def _drain_loop(self) -> None:
        """Sole consumer of the worker→main event queue."""
        while not self.shutdown.is_set():
            try:
                ev = self.events.get(timeout=0.5)
            except queue.Empty:
                continue
            self._apply(ev)
            self._broadcast(ev)
        # Drain any tail events emitted while shutting down — best effort.
        try:
            while True:
                ev = self.events.get_nowait()
                self._apply(ev)
                self._broadcast(ev)
        except queue.Empty:
            pass

    def _apply(self, ev: dict) -> None:
        card = self.cards.get(ev.get("card"))
        if card is None:
            return
        s = card.snapshot
        s.updated_at = time.time()
        kind = ev.get("event")
        if kind == worker_module.EVT_LOADING:
            s.stage = "loading"
            # ``ev.message`` is the loading sub-stage (``model``, ``featurizer``)
            # — keep it out of ``s.protein`` so it can never accidentally
            # show up as a complex name in the UI banner via snapshot replay.
            s.protein = ""
            s.alive = True
        elif kind == worker_module.EVT_READY:
            s.stage = "ready"
            s.alive = True
            # Successful boot resets the restart backoff.
            card.consecutive_failures = 0
        elif kind == worker_module.EVT_START:
            s.stage = "running"
            s.protein = ev.get("name", "")
            s.step = 0
            s.total = 0
            s.error = None
            s.alive = True
        elif kind == worker_module.EVT_STAGE:
            s.stage = ev.get("stage", s.stage)
            s.step = ev.get("step", 0)
            s.total = ev.get("total", 0)
            # Workers stamp every stage event with the active complex name
            # so a tab opened mid-prediction can populate the banner without
            # waiting for the next ``start`` event.
            if ev.get("name"):
                s.protein = ev["name"]
            s.alive = True
        elif kind == worker_module.EVT_COMPLETE:
            s.stage = "complete"
            s.last_complete = {
                "name": ev.get("name"),
                "elapsed": ev.get("elapsed"),
                "confidence": ev.get("confidence", {}),
            }
            s.alive = True
        elif kind == worker_module.EVT_ERROR:
            s.stage = "error"
            s.error = ev.get("message", "")
        elif kind == worker_module.EVT_STOPPED:
            s.stage = "stopped"
            s.alive = False

    @staticmethod
    def _snapshot_to_dict(s: CardSnapshot) -> dict:
        return {
            "card": s.card_id,
            "device": s.device_id,
            "label": s.label,
            "alive": s.alive,
            "stage": s.stage,
            "protein": s.protein,
            "step": s.step,
            "total": s.total,
            "error": s.error,
            "last_complete": s.last_complete,
        }

    # ── supervisor ──────────────────────────────────────────────────────

    def _supervisor_loop(self) -> None:
        """Detect crashed workers and respawn with exponential backoff."""
        while not self.shutdown.is_set():
            self._poll_workers()
            self.shutdown.wait(SUPERVISOR_POLL_SECONDS)

    def _poll_workers(self) -> None:
        for card in self.cards.values():
            proc = card.process
            if proc is None:
                continue
            if proc.is_alive():
                continue
            exitcode = proc.exitcode
            proc.join(timeout=0)  # reap zombie

            # If we asked the worker to stop, don't restart it.
            if self.shutdown.is_set():
                continue

            card.consecutive_failures += 1
            backoff = min(
                RESTART_BACKOFF_BASE_SECONDS * (2 ** (card.consecutive_failures - 1)),
                RESTART_BACKOFF_MAX_SECONDS,
            )
            card.snapshot.alive = False
            card.snapshot.stage = "error"
            card.snapshot.error = (
                f"worker exited (code={exitcode}); restarting in {backoff:.0f}s"
            )
            crash_event = {
                "card": card.snapshot.card_id,
                "event": worker_module.EVT_ERROR,
                "message": card.snapshot.error,
            }
            self._broadcast(crash_event)
            log.warning("card %d worker died (exitcode=%s, failures=%d), respawn in %.0fs",
                        card.snapshot.card_id, exitcode, card.consecutive_failures, backoff)

            time.sleep(backoff)
            if self.shutdown.is_set():
                return
            self._spawn(card)

    def _spawn(self, card: Card, *, start_offset: int | None = None) -> None:
        proc = self._ctx.Process(
            target=worker_module.run,
            name=f"quad-worker-{card.snapshot.device_id}",
            kwargs=dict(
                card_id=card.snapshot.card_id,
                device_id=card.snapshot.device_id,
                complexes=self.complexes,
                dispatch_counter=self._dispatch,
                events=self.events,
                play=self.play,
                shutdown=self.shutdown,
                start_offset=start_offset,
                fast=self.fast,
                sampling_steps=self.sampling_steps,
                linger_seconds=self.linger_seconds,
                mesh_graph_descriptor=card.mesh_graph_descriptor,
                log_dir=self.log_dir,
            ),
            daemon=True,
        )
        proc.start()
        card.process = proc
        card.snapshot.stage = "starting"
        card.snapshot.protein = ""
        card.snapshot.step = 0
        card.snapshot.total = 0
        card.snapshot.error = None
        log.info("spawned worker for card %d (pid=%d, start_offset=%s)",
                 card.snapshot.card_id, proc.pid, start_offset)
