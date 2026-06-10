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

from tt_bio.distributed import ControllerClient, HttpProgressQueue


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
    os.environ["TT_BIO_LOGICAL_DEVICE_ID"] = str(worker_info.get("logical_device_id", 0))
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
    cache = Path(os.environ.get("BOLTZ_CACHE", str(Path("~/.boltz").expanduser())))
    cache.mkdir(parents=True, exist_ok=True)
    # Protenix-v2: resolve the v2 checkpoint. Prefer $PROTENIX_CKPT, then the worker
    # cache, then download from the Hugging Face weights mirror on first use.
    if cfg.get("model") == "protenix-v2":
        cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)
        ckpt = os.environ.get("PROTENIX_CKPT") or str(cache / "protenix-v2.pt")
        if not Path(ckpt).exists():
            from huggingface_hub import hf_hub_download

            ckpt = hf_hub_download(repo_id="TMF001/protenix-v2-weights",
                                   filename="protenix-v2.pt", local_dir=str(cache))
        cfg["protenix_ckpt"] = ckpt
        return
    # ESMFold2 loads its weights from HF on the first fold and needs no Boltz-2
    # checkpoints / molecule library — only a writable MSA dir.
    if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
        cfg["msa_dir"] = _resolve_msa_dir(cfg.get("msa_dir"), cache)
        return
    from tt_bio.main import download_all

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
        self.pfn = None  # progress callback (set per run), shared by both models
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
        self.pfn = None
        self.run_id = None
        self.config_hash = None
        gc.collect()
        if self.accelerator == "tenstorrent":
            try:
                from tt_bio.tenstorrent import cleanup as _tt_cleanup

                _tt_cleanup()
            except Exception:
                pass

    def load(self, run_id: str, cfg: dict[str, Any]) -> None:
        if self.accelerator == "tenstorrent":
            from tt_bio.tenstorrent import set_fast_mode

            set_fast_mode(cfg.get("fast", False))

        # ESMFold2 family: load + patch the ttnn model (no tokenizer/featurizer;
        # chains are read straight from the input in predict_one). Symmetric to
        # the Boltz-2 load below — same _WorkerState, same scheduler/worker loop.
        if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
            from tt_bio.esmfold2_runtime import load_ttnn_esmfold2

            repo = "biohub/ESMFold2-Fast" if cfg["model"] == "esmfold2-fast" else "biohub/ESMFold2"
            self.model = load_ttnn_esmfold2(esmfold2_repo=repo, fast=cfg.get("fast", False))
            self.model._esmc.preload()
            self.prepare = None
            self.run_id = run_id
            self.config_hash = _hash_run_config(cfg)
            return

        # Protenix-v2 family: load the ttnn model (no Boltz-2 tokenizer/featurizer;
        # sequences are featurized in predict_one via tt_bio.protenix_data). Symmetric
        # to the ESMFold2 branch above.
        if cfg.get("model") == "protenix-v2":
            from tt_bio.protenix import Protenix

            self.model = Protenix.load_from_checkpoint(cfg["protenix_ckpt"])
            self.prepare = None
            self.run_id = run_id
            self.config_hash = _hash_run_config(cfg)
            return

        from tt_bio.boltz2 import Boltz2
        from tt_bio.data.featurizer import Boltz2Featurizer
        from tt_bio.data.mol import load_canonicals
        from tt_bio.data.tokenize import Boltz2Tokenizer
        from tt_bio.main import prepare_features

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
        if cfg.get("model") == "protenix-v2":
            return self._predict_protenix_one(path, cfg)
        if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
            return self._predict_esmfold2_one(path, cfg)

        from tt_bio.main import to_batch, write_result

        feats, input_struct = self.prepare(path, method=cfg.get("method"), progress=self.pfn)
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

    def _predict_esmfold2_one(self, path: Path, cfg: dict[str, Any]):
        import hashlib
        import types

        from tt_bio.esmfold2 import report_progress
        from tt_bio.esmfold2_runtime import fold_complex, resolve_msa
        from tt_bio.main import _generate_esmfold2_a3m, _read_protein_chains, _write_structure

        chains = _read_protein_chains(path)
        if not chains:
            raise RuntimeError("no protein sequences")
        msa_dir = Path(cfg["msa_dir"])
        max_msa = cfg.get("max_msa_seqs") or 16384
        # Only the checkpoints that ship an MSA encoder can use an MSA. ESMFold2
        # has one; ESMFold2-Fast does not (model.msa_encoder is None), so there's
        # nothing to consume an alignment — skip the search and fold single-seq
        # rather than do wasted work and falsely report msa=true.
        uses_msa = getattr(self.model, "msa_encoder", None) is not None

        # MSA phase — rendered as the "MSA" stage, exactly like Boltz-2 (which
        # generates worker-side in prepare_features). When a source is given we
        # search any chain whose {seq_hash}.a3m/.csv is not already cached, into
        # the shared msa_dir. MSA is optional: with no source, fold single-seq.
        report_progress("msa")
        if uses_msa and (cfg.get("use_msa_server") or cfg.get("msa_db_path")):
            to_gen = {}
            for _cid, seq, spec in chains:
                if spec and Path(spec).expanduser().exists():
                    continue
                h = hashlib.sha256(seq.encode()).hexdigest()[:16]
                if not (msa_dir / f"{h}.a3m").exists() and not (msa_dir / f"{h}.csv").exists():
                    to_gen[h] = seq
            if to_gen:
                _generate_esmfold2_a3m(
                    to_gen, path.stem, msa_dir, cfg.get("msa_db_path"), cfg.get("use_envdb", False),
                    cfg.get("msa_server_url"), cfg.get("msa_pairing_strategy"),
                    cfg.get("msa_server_username"), cfg.get("msa_server_password"),
                    cfg.get("api_key_value"))

        report_progress("prep")
        chains = [(cid, seq, resolve_msa(spec, seq, msa_dir, max_sequences=max_msa) if uses_msa else None)
                  for cid, seq, spec in chains]
        res = fold_complex(
            self.model, chains,
            num_loops=cfg["recycling_steps"], num_sampling_steps=cfg["sampling_steps"],
            num_diffusion_samples=cfg["diffusion_samples"], seed=cfg.get("seed") or 0,
        )
        out = Path(cfg["struct_dir"]) / f"{path.stem}.{cfg['output_format']}"
        _write_structure(res.complex, out, cfg["output_format"])
        metrics = {
            "plddt": round(float(res.plddt.mean()), 4),
            "n_residues": sum(len(c[1]) for c in chains), "n_chains": len(chains),
            "msa": any(c[2] is not None for c in chains),
            "samples": cfg["diffusion_samples"],  # best-of-N: report N (plddt is the winner's)
        }
        if getattr(res, "ptm", None) is not None:
            metrics["ptm"] = round(float(res.ptm), 4)
        # _execute_job inspects feats["record"].affinity; ESMFold2 has no affinity.
        feats = {"record": types.SimpleNamespace(affinity=False)}
        return metrics, None, feats

    def _predict_protenix_one(self, path: Path, cfg: dict[str, Any]):
        """Protenix-v2 protein fold: sequence -> (optional MSA) -> on-device fold -> structure.

        Rides the same MSA stage as ESMFold2/Boltz-2: any chain whose {seq_hash}.a3m is not
        cached is searched into the shared msa_dir, then resolved and featurized. Protenix-v2's
        MSA module consumes one chain's alignment; multi-chain inputs are concatenated and
        folded single-sequence (no inter-chain MSA pairing)."""
        import hashlib
        import types

        from tt_bio.esmfold2 import report_progress
        from tt_bio.main import (_generate_esmfold2_a3m, _read_protein_chains,
                                 _resolve_a3m_text, _write_protenix_structure)
        from tt_bio.protenix_data import aatype_from_sequence, build_protein_features

        chains = _read_protein_chains(path)
        if not chains:
            raise RuntimeError("no protein sequences")
        msa_dir = Path(cfg["msa_dir"])
        seq = "".join(c[1] for c in chains)

        report_progress("msa")
        a3m = None
        if len(chains) == 1:
            _cid, cseq, spec = chains[0]
            have_spec = bool(spec and Path(spec).expanduser().exists())
            if (cfg.get("use_msa_server") or cfg.get("msa_db_path")) and not have_spec:
                h = hashlib.sha256(cseq.encode()).hexdigest()[:16]
                if not (msa_dir / f"{h}.a3m").exists():
                    _generate_esmfold2_a3m(
                        {h: cseq}, path.stem, msa_dir, cfg.get("msa_db_path"),
                        cfg.get("use_envdb", False), cfg.get("msa_server_url"),
                        cfg.get("msa_pairing_strategy"), cfg.get("msa_server_username"),
                        cfg.get("msa_server_password"), cfg.get("api_key_value"))
            a3m = _resolve_a3m_text(spec, cseq, msa_dir)

        report_progress("prep")
        feats = build_protein_features(seq, a3m=a3m)

        def _pfn(stage, step, total):
            report_progress("diffusion" if stage == "trunk" else stage)

        coords, plddt = self.model.fold(
            feats, n_step=cfg["sampling_steps"], n_sample=cfg["diffusion_samples"],
            seed=cfg.get("seed") or 0, progress_fn=_pfn, return_confidence=True,
        )
        out = Path(cfg["struct_dir"]) / f"{path.stem}.{cfg['output_format']}"
        _write_protenix_structure(coords[0], feats, aatype_from_sequence(seq), out, cfg["output_format"])
        metrics = {
            "plddt": round(float(plddt), 4), "n_residues": len(seq), "n_chains": len(chains),
            "msa": a3m is not None, "n_atoms": int(coords.shape[1]), "samples": cfg["diffusion_samples"],
        }
        return metrics, None, {"record": types.SimpleNamespace(affinity=False)}

    def predict_affinity(self, path: Path, pred_structure, cfg: dict[str, Any]) -> dict[str, float]:
        from tt_bio.boltz2 import Boltz2
        from tt_bio.main import to_batch

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

    keep = {k: cfg.get(k) for k in ("model", "conf_kwargs", "aff_kwargs", "fast", "method")}
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
                    from tt_bio.progress import make_progress_fn

                    pfn = make_progress_fn(
                        HttpProgressQueue(client, run_id, worker_id),
                        worker_info["device_id"], worker_id, meta,
                    )
                    state.pfn = pfn
                    if cfg.get("model", "boltz2") in ("esmfold2", "esmfold2-fast"):
                        from tt_bio import esmfold2 as _E

                        _E.set_progress(pfn)  # trunk/diffusion per-step bars
                    else:
                        state.model.progress_fn = pfn
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

    workdir = Path(tempfile.mkdtemp(prefix=f"tt-bio-{job_id}-"))
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

        # Both model families start in the MSA stage and resolve/search MSAs
        # worker-side; the esmfold2 path then reports "prep" before folding.
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
