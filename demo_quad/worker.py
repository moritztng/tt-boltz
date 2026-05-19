"""Per-card prediction worker for the quad-card demo.

Lives in its own subprocess pinned to one Tenstorrent chip via
``TT_VISIBLE_DEVICES``. Loads the Boltz-2 model once, pre-features the protein
rotation once, then loops forever predicting and streaming events through a
shared ``multiprocessing.Queue``.

Design notes
------------
* Heavy imports (``torch``, ``ttnn``, ``tt_boltz.boltz2``) happen *inside*
  :func:`run` so the parent process never opens a TT device and each worker
  can set its environment before the first ttnn import.
* ``PR_SET_PDEATHSIG`` makes the kernel kill the worker if the parent dies,
  so a crashed Flask process never leaves orphan subprocesses holding TT
  device file descriptors.
* stdout/stderr go to a rotating log file (not ``/dev/null``) so crashes are
  forensically recoverable.
* The worker process exits with a non-zero status on irrecoverable startup
  errors so the supervisor can respawn it. Per-prediction failures are
  reported as ``error`` events and the loop continues.
"""

from __future__ import annotations

import logging
import logging.handlers
import os
import signal
import sys
import tempfile
import time
import traceback
import warnings
from pathlib import Path
from typing import Sequence

from demo_quad.complexes import Complex


# Event names shared between worker and main process. Keeping them as module
# constants makes typos a hard error rather than a silent UI bug.
EVT_LOADING = "loading"
EVT_READY = "ready"
EVT_START = "start"
EVT_STAGE = "stage"
EVT_INTERMEDIATE = "intermediate"
EVT_COMPLETE = "complete"
EVT_ERROR = "error"
EVT_STOPPED = "stopped"


def _install_pdeathsig() -> None:
    """Ask the kernel to send SIGKILL to this process when the parent dies.

    Linux-only and best-effort: if the syscall fails for any reason we keep
    going, since the supervisor's join+terminate is the secondary safety net.
    """
    try:
        import ctypes
        PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(PR_SET_PDEATHSIG, signal.SIGKILL, 0, 0, 0)
    except Exception:
        pass


def _redirect_stdio_to_log(log_path: Path) -> logging.Logger:
    """Send stdout, stderr, and Python logging to a rotating per-worker log.

    Returns the configured logger so callers can write structured messages
    without having to know the file path.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    handler = logging.handlers.RotatingFileHandler(
        log_path, maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
    root.addHandler(handler)
    root.setLevel(logging.INFO)

    # Redirect the raw file descriptors so C-level prints from ttnn/torch end
    # up in the same place. open()ing log_path twice is fine; both handles
    # write through the rotating handler's underlying inode.
    fd = os.open(log_path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
    os.dup2(fd, 1)
    os.dup2(fd, 2)
    os.close(fd)
    sys.stdout = os.fdopen(1, "w", buffering=1)
    sys.stderr = os.fdopen(2, "w", buffering=1)

    return logging.getLogger("quad.worker")


def run(
    card_id: int,
    device_id: int,
    complexes: Sequence[Complex],
    dispatch_counter,
    events,
    play,
    shutdown,
    *,
    start_offset: int | None = None,
    fast: bool = True,
    sampling_steps: int = 200,
    intermediate_interval: int = 10,
    linger_seconds: float = 2.0,
    mesh_graph_descriptor: str | None = None,
    log_dir: str | None = None,
) -> None:
    """Worker entry point — spawn target for ``multiprocessing.Process``.

    Parameters
    ----------
    card_id:
        Logical card index (0..N-1) used as the routing key in event payloads.
    device_id:
        The physical TT device id (``/dev/tenstorrent/<id>``) this worker
        pins to via ``TT_VISIBLE_DEVICES``.
    complexes:
        The shared rotation. Workers pull the next item via
        ``dispatch_counter`` so the pool is naturally work-stealing — no card
        sits idle waiting on a slower one.
    dispatch_counter:
        ``multiprocessing.Value('i', ...)`` used as an atomic counter modulo
        ``len(complexes)``. The pool pre-seeds it so the first round assigns
        complex N to card N deterministically (see ``start_offset``).
    start_offset:
        If given, the first iteration of this worker uses index
        ``start_offset`` directly instead of pulling from the counter. The
        pool only sets this for the very first spawn; respawns leave it
        ``None`` so the new worker rejoins the global rotation seamlessly.
    events, play, shutdown:
        Shared ``multiprocessing`` primitives. ``events`` is the broadcast
        queue; ``play`` gates the inference loop; ``shutdown`` requests a
        clean exit.
    """
    # ── Process-level setup ──────────────────────────────────────────────
    _install_pdeathsig()

    # SIGTERM is sent by the supervisor for a graceful stop. We translate it
    # into the shared shutdown event so the loop can drain cleanly.
    def _on_sigterm(_signum, _frame):
        try:
            shutdown.set()
        except Exception:
            pass
    signal.signal(signal.SIGTERM, _on_sigterm)
    signal.signal(signal.SIGINT, signal.SIG_IGN)

    # ── Pin to one TT card before any ttnn import ────────────────────────
    os.environ["TT_VISIBLE_DEVICES"] = str(device_id)
    os.environ["TT_BOLTZ_LOGICAL_DEVICE_ID"] = "0"
    if mesh_graph_descriptor:
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = mesh_graph_descriptor
    os.environ.setdefault("LOGURU_LEVEL", "WARNING")
    os.environ.setdefault("TT_METAL_LOGGER_LEVEL", "FATAL")

    log_root = Path(log_dir or os.environ.get("BOLTZ_QUAD_LOG_DIR")
                    or str(Path("~/.boltz/logs").expanduser()))
    log = _redirect_stdio_to_log(log_root / f"quad-worker-{device_id}.log")
    log.info("worker boot: card=%d device=%d pid=%d", card_id, device_id, os.getpid())

    def emit(event: str, **payload) -> None:
        """Best-effort non-blocking event publish.

        We drop the event if the parent's drain thread is hopelessly behind.
        That should never happen in practice — the drain is a tight Python
        loop on a single ``multiprocessing.Queue`` — and if it does, dropping
        is the correct backpressure response for a UI stream.
        """
        try:
            events.put_nowait({"card": card_id, "event": event, **payload})
        except Exception:
            pass

    # ── Heavy imports + model load ───────────────────────────────────────
    try:
        import torch
        from rdkit import Chem
        from tt_boltz.boltz2 import Boltz2
        from tt_boltz.data import const
        from tt_boltz.data.featurizer import Boltz2Featurizer
        from tt_boltz.data.mol import load_canonicals
        from tt_boltz.data.tokenize import Boltz2Tokenizer
        from tt_boltz.main import download_all, prepare_features, to_batch
        from tt_boltz.predictor import (
            _DIFFUSION_ARGS,
            _MSA_ARGS,
            _PAIRFORMER_ARGS,
            _STEERING_ARGS,
            _build_intermediate_frame_maker,
            finalize_prediction,
        )
        from tt_boltz.tenstorrent import set_fast_mode

        warnings.filterwarnings("ignore")
        torch.set_grad_enabled(False)
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

        cache = Path(os.environ.get("BOLTZ_CACHE") or str(Path("~/.boltz").expanduser()))
        cache.mkdir(parents=True, exist_ok=True)
        download_all(cache)
        mol_dir = cache / "mols"
        # Share MSA cache with the single-card demo so identical sequences
        # are never re-fetched across demos.
        msa_dir = cache / "demo_msa"
        msa_dir.mkdir(parents=True, exist_ok=True)

        set_fast_mode(fast)
        torch_device = torch.device("cpu")

        emit(EVT_LOADING, message="featurizer")
        ccd = load_canonicals(mol_dir)
        tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()

        emit(EVT_LOADING, message="model")
        model = Boltz2.load_from_checkpoint(
            cache / "boltz2_conf.ckpt",
            predict_args={
                "recycling_steps": 3,
                "sampling_steps": sampling_steps,
                "diffusion_samples": 1,
                "max_parallel_samples": 1,
            },
            diffusion_process_args=_DIFFUSION_ARGS,
            pairformer_args=_PAIRFORMER_ARGS,
            msa_args=_MSA_ARGS,
            steering_args=_STEERING_ARGS,
            use_kernels=False,
            use_tenstorrent=True,
        ).eval().to(torch_device)
    except Exception as exc:
        log.exception("startup failed")
        emit(EVT_ERROR, message=f"startup: {exc}"[:200])
        # Non-zero exit so the supervisor knows this isn't a clean stop and
        # backs off appropriately before respawning.
        sys.exit(1)

    def prepare_complex(spec: Complex):
        """Featurize one complex from its YAML.

        Done lazily right before each prediction — first time a sequence is
        seen, this generates an MSA via ColabFold (~30 s); subsequent calls
        for the same sequence hit the shared on-disk cache and return in
        ~1-2 s. We trade a small per-iteration cost for zero startup latency
        and a flat memory profile (no pre-prepped features hanging around
        for all 16 complexes × 4 workers = 64 dict trees).
        """
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False,
        ) as tf:
            tf.write(spec.yaml)
            yaml_path = Path(tf.name)
        try:
            return prepare_features(
                yaml_path, ccd, mol_dir, msa_dir, tokenizer, featurizer,
                use_msa=True, msa_url="https://api.colabfold.com",
                msa_strategy="greedy",
                msa_user=None, msa_pass=None, api_key=None,
                max_msa=const.max_msa_seqs,
            )
        finally:
            try:
                yaml_path.unlink()
            except Exception:
                pass

    def next_index() -> int:
        """Atomic fetch-and-increment over the shared rotation counter.

        Wrapped in ``get_lock()`` so concurrent workers never grab the same
        slot. Modulo here, not on the counter itself, so the raw counter
        value stays monotonically increasing and useful for debugging.
        """
        with dispatch_counter.get_lock():
            n = dispatch_counter.value
            dispatch_counter.value = n + 1
        return n % len(complexes)

    # ── Progress hook ────────────────────────────────────────────────────
    # progress_fn runs inside the model thread. It must be cheap and must
    # never block.
    state = {"name": "", "template_sent": False, "make_frame": None}

    def progress_fn(stage: str, step: int = 0, total: int = 0, coords=None, **_):
        emit(EVT_STAGE, stage=stage, step=step, total=total, name=state["name"])
        if (
            coords is not None
            and stage == "diffusion"
            and (step % intermediate_interval == 0 or step >= total)
        ):
            make_frame = state["make_frame"]
            if make_frame is None:
                return
            frame = make_frame(coords, include_cif=not state["template_sent"])
            if frame is None:
                return
            state["template_sent"] = state["template_sent"] or "cif" in frame
            emit(EVT_INTERMEDIATE, name=state["name"], **frame)

    model.progress_fn = progress_fn

    emit(EVT_READY)
    log.info("ready: %d complexes in rotation (start_offset=%s)",
             len(complexes), start_offset)

    # ── Prediction loop ──────────────────────────────────────────────────
    # Pull index from `start_offset` for the very first iteration so the
    # initial round is the deterministic processor↔complex mapping the demo
    # promises; after that, fall back to the shared atomic counter.
    pending_first = start_offset

    while not shutdown.is_set():
        if not play.wait(timeout=0.5):
            continue

        if pending_first is not None:
            idx = pending_first % len(complexes)
            pending_first = None
        else:
            idx = next_index()
        spec = complexes[idx]

        # Featurize lazily. On the cold path this can hit the network for
        # an MSA; surface a stage event so the UI cover stays meaningful.
        emit(EVT_STAGE, stage="prep", step=0, total=1, name=spec.name)
        try:
            feats, input_struct = prepare_complex(spec)
        except Exception as exc:
            log.exception("featurize failed for %s", spec.name)
            emit(EVT_ERROR, name=spec.name, message=f"featurize: {exc}"[:200])
            if shutdown.wait(1.0):
                break
            continue

        batch = to_batch(feats, torch_device)
        atom_mask = batch["atom_pad_mask"].squeeze(0).bool()
        state["name"] = spec.name
        state["template_sent"] = False
        state["make_frame"] = _build_intermediate_frame_maker(
            input_struct.remove_invalid_chains(), atom_mask,
        )

        t0 = time.time()
        emit(EVT_START, name=spec.name, seq_len=spec.seq_len)
        try:
            with torch.no_grad(), torch.autocast(
                device_type=torch_device.type, dtype=torch.bfloat16,
            ):
                pred = model.predict_step(batch)
            if pred is None or pred.get("exception"):
                emit(EVT_ERROR, name=spec.name, message="empty prediction")
                log.warning("empty prediction for %s", spec.name)
                continue
            cif, confidence = finalize_prediction(pred, input_struct)
            emit(
                EVT_COMPLETE,
                name=spec.name,
                cif=cif,
                confidence=confidence,
                elapsed=round(time.time() - t0, 1),
            )
            # Let viewers see the final pLDDT-colored result before the
            # cover snaps back up for the next prediction. shutdown.wait
            # returns immediately if shutdown is set, so we never hold up
            # a clean stop.
            shutdown.wait(linger_seconds)
        except Exception as exc:
            log.exception("prediction failed for %s", spec.name)
            emit(EVT_ERROR, name=spec.name, message=str(exc)[:200])
            # Short backoff so a misbehaving input doesn't tight-loop and
            # saturate the event queue.
            if shutdown.wait(1.0):
                break

    emit(EVT_STOPPED)
    log.info("worker exit clean")
