"""Long-lived prediction worker.

A worker process owns one accelerator slot for its entire lifetime: it loads the
Boltz-2 model once, then pulls jobs from a scheduler over HTTP and runs them
until cancelled. The same loop runs for local single-machine runs and for
multi-host runs; only the scheduler URL differs.
"""

from __future__ import annotations

import base64
import gc
import os
import shutil
import signal
import sys
import tempfile
import time
import traceback
from functools import partial
from pathlib import Path
from typing import Any

import torch

from tt_boltz.distributed import ControllerClient, HttpProgressQueue


def _silence_subprocess_output() -> None:
    """Send stdout/stderr to /dev/null so kernel/library noise stays hidden."""
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    sys.stderr = devnull
    dn_fd = os.open(os.devnull, os.O_WRONLY)
    os.dup2(dn_fd, 1)
    os.dup2(dn_fd, 2)
    os.close(dn_fd)


def _apply_tt_environment(worker_info: dict[str, Any]) -> None:
    """Configure TT visibility for this worker before importing ttnn."""
    if worker_info["accelerator"] != "tenstorrent":
        return
    os.environ["TT_VISIBLE_DEVICES"] = str(worker_info.get("visible_devices") or worker_info["device_id"])
    os.environ["TT_BOLTZ_LOGICAL_DEVICE_ID"] = str(worker_info.get("logical_device_id", 0))
    mgd = worker_info.get("mesh_graph_descriptor")
    if mgd and not os.environ.get("TT_MESH_GRAPH_DESC_PATH"):
        os.environ["TT_MESH_GRAPH_DESC_PATH"] = str(mgd)


def _ensure_local_artifacts(cfg: dict[str, Any]) -> None:
    """Make sure model files and caches exist locally for this worker.

    Model checkpoints and the molecule library are always resolved to the
    worker's own ~/.boltz/ cache. For the MSA directory we prefer the path
    the controller asked for (so single-machine and shared-filesystem runs
    keep populating <out_dir>/msa/ exactly like the legacy pipeline) and
    only fall back to the local cache when that path is not writable on
    this host (the no-shared-FS multi-machine case).
    """
    from tt_boltz.main import download_all

    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
    cache.mkdir(parents=True, exist_ok=True)
    download_all(cache)
    cfg["conf_ckpt"] = str(cache / "boltz2_conf.ckpt")
    cfg["aff_ckpt"] = str(cache / "boltz2_aff.ckpt")
    cfg["mol_dir"] = str(cache / "mols")
    cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)


def _resolve_msa_dir(requested: str | None, cache: Path) -> str:
    """Honor controller's msa_dir if it already exists and is writable on this
    host (covers single-machine runs and shared-filesystem multi-machine
    setups); otherwise fall back to ~/.boltz/msa/ on the worker."""
    if requested:
        path = Path(requested)
        if path.is_dir() and os.access(path, os.W_OK):
            return str(path)
    fallback = cache / "msa"
    fallback.mkdir(parents=True, exist_ok=True)
    return str(fallback)


class _WorkerState:
    """Holds the loaded model and per-run helpers."""

    def __init__(self, accelerator: str) -> None:
        self.accelerator = accelerator
        self.run_id: str | None = None
        self.config_hash: str | None = None
        self.model = None
        self.aff_model = None
        self.prepare = None
        if accelerator == "gpu" and torch.cuda.is_available():
            self.torch_device = torch.device("cuda:0")
        else:
            self.torch_device = torch.device("cpu")

    def configured_for(self, run_id: str, cfg: dict[str, Any]) -> bool:
        return self.run_id == run_id and self.config_hash == _hash_run_config(cfg)

    def reset(self) -> None:
        self.model = None
        self.aff_model = None
        self.prepare = None
        self.run_id = None
        self.config_hash = None
        gc.collect()
        if self.accelerator == "tenstorrent":
            try:
                from tt_boltz.tenstorrent import cleanup as _tt_cleanup

                _tt_cleanup()
            except Exception:
                pass

    def load(self, run_id: str, cfg: dict[str, Any]) -> None:
        from tt_boltz.boltz2 import Boltz2
        from tt_boltz.data.featurizer import Boltz2Featurizer
        from tt_boltz.data.mol import load_canonicals
        from tt_boltz.data.tokenize import Boltz2Tokenizer
        from tt_boltz.main import prepare_features

        if self.accelerator == "tenstorrent":
            from tt_boltz.tenstorrent import set_fast_mode

            set_fast_mode(cfg.get("fast", False))
        tokenizer, featurizer = Boltz2Tokenizer(), Boltz2Featurizer()
        ccd = load_canonicals(Path(cfg["mol_dir"]))
        self.prepare = partial(
            prepare_features,
            ccd=ccd,
            mol_dir=Path(cfg["mol_dir"]),
            msa_dir=Path(cfg["msa_dir"]),
            tokenizer=tokenizer,
            featurizer=featurizer,
            use_msa=cfg["use_msa_server"],
            msa_url=cfg["msa_server_url"],
            msa_strategy=cfg["msa_pairing_strategy"],
            msa_user=cfg["msa_server_username"],
            msa_pass=cfg["msa_server_password"],
            api_key=cfg["api_key_value"],
            max_msa=cfg["max_msa_seqs"],
            msa_db_path=cfg.get("msa_db_path"),
            use_envdb=cfg.get("use_envdb", False),
        )
        self.model = (
            Boltz2.load_from_checkpoint(cfg["conf_ckpt"], **cfg["conf_kwargs"])
            .eval()
            .to(self.torch_device)
        )
        self.run_id = run_id
        self.config_hash = _hash_run_config(cfg)

    def predict_one(self, path: Path, cfg: dict[str, Any]):
        from tt_boltz.main import to_batch, write_result

        feats, input_struct = self.prepare(path, method=cfg.get("method"))
        batch = to_batch(feats, self.torch_device)
        with torch.no_grad():
            pred = self.model.predict_step(batch)
        metrics, best = write_result(
            pred,
            batch,
            input_struct,
            Path(cfg["struct_dir"]),
            cfg["output_format"],
            cfg["write_pae"],
            cfg["write_pde"],
            cfg["write_embeddings"],
        )
        return metrics, best, feats

    def predict_affinity(self, path: Path, pred_structure, cfg: dict[str, Any]) -> dict[str, float]:
        from tt_boltz.boltz2 import Boltz2
        from tt_boltz.main import to_batch

        if self.aff_model is None:
            self.aff_model = (
                Boltz2.load_from_checkpoint(cfg["aff_ckpt"], **cfg["aff_kwargs"])
                .eval()
                .to(self.torch_device)
            )

        feats, _ = self.prepare(path, method="other", affinity=True, pred_structure=pred_structure)
        batch = to_batch(feats, self.torch_device)
        with torch.no_grad():
            pred = self.aff_model.predict_step(batch)
        if pred.get("exception"):
            return {}
        keys = [
            "affinity_pred_value",
            "affinity_probability_binary",
            "affinity_pred_value1",
            "affinity_probability_binary1",
            "affinity_pred_value2",
            "affinity_probability_binary2",
        ]
        return {k: round(pred[k].item(), 6) for k in keys if k in pred}


def _hash_run_config(cfg: dict[str, Any]) -> str:
    """Stable hash of the parts of the config that affect model setup."""
    import hashlib
    import json

    keep = {k: cfg.get(k) for k in ("conf_kwargs", "aff_kwargs", "fast", "method")}
    blob = json.dumps(keep, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def _install_signal_handlers() -> None:
    def _raise(signum, _frame):
        raise KeyboardInterrupt(f"worker received signal {signum}")

    try:
        signal.signal(signal.SIGTERM, _raise)
        signal.signal(signal.SIGINT, _raise)
    except Exception:
        pass


def run_worker_loop(
    controller_url: str,
    worker_info: dict[str, Any],
    debug: bool = False,
    idle_poll: float = 1.0,
) -> None:
    """Connect to a scheduler and process jobs until cancelled.

    Loads model artifacts once per run and reuses them for every job in that
    run. If the run's config changes, the model is reloaded.
    """
    if not debug:
        _silence_subprocess_output()
    _install_signal_handlers()
    _apply_tt_environment(worker_info)

    client = ControllerClient(controller_url)
    worker_id = worker_info["worker_id"]
    meta = {
        "dev": worker_info["device_id"],
        "worker": worker_id,
        "host": worker_info["host"],
        "accelerator": worker_info["accelerator"],
        "label": worker_info["label"],
    }

    def emit(run_id: str, event: str, **kw):
        try:
            client.event(run_id, worker_id, {"event": event, **meta, **kw})
        except Exception:
            pass

    state = _WorkerState(worker_info["accelerator"])
    try:
        while True:
            lease = client.lease(worker_info, batch_size=1)
            jobs = lease.get("jobs") or []
            if not jobs:
                time.sleep(idle_poll)
                continue

            run_id = lease["run_id"]
            cfg = dict(lease["config"])
            _ensure_local_artifacts(cfg)

            if not state.configured_for(run_id, cfg):
                state.reset()
                try:
                    emit(run_id, "loading")
                    state.load(run_id, cfg)
                    from tt_boltz.progress import make_progress_fn

                    state.model.progress_fn = make_progress_fn(
                        HttpProgressQueue(client, run_id, worker_id),
                        worker_info["device_id"], worker_id, meta,
                    )
                except Exception as exc:
                    traceback.print_exc()
                    _complete_failure(client, run_id, worker_id, meta, jobs, str(exc)[:200])
                    state.reset()
                    continue

            for job in jobs:
                _execute_job(state, job, cfg, run_id, client, worker_id, meta)
    except KeyboardInterrupt:
        pass
    finally:
        state.reset()


def _execute_job(
    state: _WorkerState,
    job: dict[str, Any],
    cfg: dict[str, Any],
    run_id: str,
    client: ControllerClient,
    worker_id: str,
    meta: dict[str, Any],
) -> None:
    job_id = job["id"]
    filename = job.get("name") or f"{job_id}.yaml"
    row: dict[str, Any] = {"id": job_id, "status": "failed"}
    t0 = time.time()

    def emit(event: str, **kw):
        try:
            client.event(run_id, worker_id, {"event": event, **meta, **kw})
        except Exception:
            pass

    workdir = Path(tempfile.mkdtemp(prefix=f"tt-boltz-{job_id}-"))
    input_path = workdir / filename
    output_dir = workdir / "out"
    output_dir.mkdir(parents=True, exist_ok=True)

    job_cfg = dict(cfg)
    job_cfg["struct_dir"] = str(output_dir)

    outputs: dict[str, str] = {}
    emit("start", name=job_id)
    try:
        try:
            input_path.write_bytes(base64.b64decode(job.get("input_b64", "")))
        except Exception as exc:
            raise RuntimeError(f"failed to decode input bytes: {exc}") from exc

        emit("stage", stage="msa")
        metrics, best, feats = state.predict_one(input_path, job_cfg)
        emit("stage", stage="saving")
        if metrics:
            row.update(metrics)
            row["status"] = "ok"
            row["runtime_s"] = round(time.time() - t0, 1)
            if feats["record"].affinity and best is not None:
                try:
                    aff = state.predict_affinity(input_path, best, job_cfg)
                    row.update(aff)
                except Exception:
                    traceback.print_exc()
        outputs = _read_outputs(output_dir)
    except Exception as exc:
        traceback.print_exc()
        row["error"] = str(exc)[:200]
    finally:
        shutil.rmtree(workdir, ignore_errors=True)

    try:
        client.complete(
            run_id,
            worker_id,
            row,
            {
                **meta,
                "event": "done",
                "name": job_id,
                "time": round(time.time() - t0, 1),
                "status": row["status"],
                "error": row.get("error", ""),
                "row": row,
            },
            outputs=outputs or None,
        )
    except Exception:
        traceback.print_exc()


def _read_outputs(output_dir: Path) -> dict[str, str]:
    """Read every file in output_dir and return name -> base64 bytes."""
    outputs: dict[str, str] = {}
    if not output_dir.exists():
        return outputs
    for path in sorted(output_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(output_dir).as_posix()
        outputs[rel] = base64.b64encode(path.read_bytes()).decode("ascii")
    return outputs


def _complete_failure(
    client: ControllerClient,
    run_id: str,
    worker_id: str,
    meta: dict[str, Any],
    jobs: list[dict[str, Any]],
    error: str,
) -> None:
    """Mark each leased job as failed when worker setup itself fails."""
    for job in jobs:
        row = {"id": job["id"], "status": "failed", "error": error}
        try:
            client.complete(
                run_id,
                worker_id,
                row,
                {**meta, "event": "done", "name": job["id"], "status": "failed",
                 "time": 0, "error": error, "row": row},
            )
        except Exception:
            pass
