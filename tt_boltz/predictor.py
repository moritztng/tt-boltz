"""Streaming single-sequence prediction for interactive front-ends.

This is a thin wrapper around :mod:`tt_boltz.main` that turns a single amino
acid sequence into an NDJSON-shaped event stream suitable for a web UI.
It reuses the same preprocessing (``prepare_features``), batching
(``to_batch``) and model kwargs the CLI uses, so behavior is identical to
``tt-boltz predict`` on a one-sequence YAML input.

Events yielded by :func:`predict_structure`::

    {"type": "progress", "stage": <str>, "step": <int>, "total": <int>}
    {"type": "intermediate", "stage": "diffusion", "step": ..., "coords": [...], "cif": <str optional>}
    {"type": "complete",   "cif": <str>, "confidence": {...}, "sequence_length": int}
    {"type": "error",      "message": <str>}

Stage names match :mod:`tt_boltz.progress` (``loading``, ``msa``, ``prep``,
``trunk``, ``diffusion``, ``confidence``, ``saving``, ``done``).
"""

from __future__ import annotations

import os
import queue
import tempfile
import threading
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Generator, Optional

import numpy as np
import torch
from rdkit import Chem

from tt_boltz.boltz2 import Boltz2
from tt_boltz.data import const
from tt_boltz.data.featurizer import Boltz2Featurizer
from tt_boltz.data.mol import load_canonicals
from tt_boltz.data.tokenize import Boltz2Tokenizer
from tt_boltz.data.types import Coords, Interface, StructureV2
from tt_boltz.data.write import to_mmcif
from tt_boltz.main import download_all, prepare_features, to_batch


AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")
MIN_LEN = 10
MAX_LEN = 1024
INTERMEDIATE_FRAME_INTERVAL = 10


def validate_sequence(sequence: str) -> tuple[bool, str]:
    """Validate an amino-acid sequence. Returns ``(is_valid, error_message)``."""
    sequence = sequence.upper().strip()
    if not sequence:
        return False, "Sequence is empty"
    if len(sequence) < MIN_LEN:
        return False, f"Sequence too short (minimum {MIN_LEN} residues)"
    if len(sequence) > MAX_LEN:
        return False, f"Sequence too long (maximum {MAX_LEN} residues)"
    invalid = set(sequence) - AMINO_ACIDS
    if invalid:
        return False, f"Invalid characters: {', '.join(sorted(invalid))}"
    return True, ""


# Default model kwargs — mirrors tt_boltz.main's `predict` command for a
# single-sample CPU/TT run without affinity or potentials.
_DIFFUSION_ARGS = {
    "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0, "noise_scale": 1.003,
    "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0, "sigma_data": 16.0,
    "P_mean": -1.2, "P_std": 1.5,
    "coordinate_augmentation": True,
    # Disable the per-step random SO(3) rotation during sampling so the
    # intermediate frames we stream to the browser stay in a fixed frame
    # and the structure visibly settles instead of tumbling.
    "coordinate_augmentation_inference": False,
    "alignment_reverse_diff": True,
    "synchronize_sigmas": True,
}
_PAIRFORMER_ARGS = {"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True}
_MSA_ARGS = {"subsample_msa": False, "num_subsampled_msa": 1024, "use_paired_feature": True}
_STEERING_ARGS = {
    "fk_steering": False, "physical_guidance_update": False,
    "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
    "fk_resampling_interval": 3, "num_gd_steps": 20,
}


def _build_intermediate_frame_maker(base_struct: StructureV2, atom_mask: torch.Tensor):
    """Return a fn that converts coords into compact streamable frame data."""
    atoms_template = base_struct.atoms.copy()
    residues_template = base_struct.residues.copy()
    residues_template["is_present"] = True

    def make(coords_tensor: torch.Tensor, include_cif: bool = False) -> Optional[dict]:
        try:
            coords = coords_tensor.float()[atom_mask].cpu().numpy()
            frame = {
                "coords": np.round(coords.reshape(-1), 3).tolist(),
                "atom_count": int(coords.shape[0]),
            }
            if include_cif:
                atoms = atoms_template.copy()
                atoms["coords"] = coords
                atoms["is_present"] = True
                snapshot = replace(
                    base_struct,
                    atoms=atoms,
                    residues=residues_template,
                    interfaces=np.array([], dtype=Interface),
                    coords=np.array([(x,) for x in coords], dtype=Coords),
                )
                frame["cif"] = to_mmcif(snapshot, plddts=None, boltz2=True)
            return frame
        except Exception:
            return None

    return make


def prepare_sequence_features(
    sequence: str,
    ccd,
    mol_dir: Path,
    msa_dir: Path,
    tokenizer,
    featurizer,
    use_msa_server: bool = True,
    msa_server_url: str = "https://api.colabfold.com",
) -> tuple[dict, StructureV2]:
    """Build features for a single protein chain. MSA results land in
    ``msa_dir`` keyed by sequence hash, so repeat calls re-use the cache."""
    msa_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        yaml_path = Path(tmp) / "input.yaml"
        yaml_path.write_text(
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {sequence}\n"
        )
        return prepare_features(
            yaml_path, ccd, mol_dir, msa_dir, tokenizer, featurizer,
            use_msa=use_msa_server, msa_url=msa_server_url, msa_strategy="greedy",
            msa_user=None, msa_pass=None, api_key=None,
            max_msa=const.max_msa_seqs,
        )


def finalize_prediction(pred: dict, input_struct: StructureV2) -> tuple[str, dict]:
    """Turn the model's raw prediction dict into a final CIF + confidence."""
    struct = input_struct.remove_invalid_chains()
    mask_1d = pred["masks"].squeeze(0) if pred["masks"].dim() > 1 else pred["masks"]
    coords = pred["coords"][0][mask_1d.bool()].cpu().numpy()

    atoms = struct.atoms.copy()
    residues = struct.residues.copy()
    atoms["coords"] = coords
    atoms["is_present"] = True
    residues["is_present"] = True
    new_struct = replace(
        struct, atoms=atoms, residues=residues,
        interfaces=np.array([], dtype=Interface),
        coords=np.array([(x,) for x in coords], dtype=Coords),
    )

    plddt = pred.get("plddt", [None])[0]
    cif = to_mmcif(new_struct, plddt, boltz2=True)
    confidence = {
        k: float(pred[k][0].item())
        for k in ("confidence_score", "ptm", "iptm", "complex_plddt")
        if k in pred
    }
    return cif, confidence


def predict_structure(
    sequence: str,
    cache_dir: Optional[Path] = None,
    use_msa_server: bool = True,
    accelerator: str = "tenstorrent",
    fast: bool = False,
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    msa_server_url: str = "https://api.colabfold.com",
) -> Generator[dict, None, None]:
    """Predict a structure for a single protein sequence and stream events."""

    sequence = sequence.upper().strip()
    ok, err = validate_sequence(sequence)
    if not ok:
        yield {"type": "error", "message": err}
        return

    yield {"type": "progress", "stage": "loading", "step": 0, "total": 1}

    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)

    cache = Path(
        cache_dir or os.environ.get("BOLTZ_CACHE", Path("~/.boltz").expanduser())
    ).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    download_all(cache)

    mol_dir = cache / "mols"
    msa_dir = cache / "demo_msa"
    use_tt = accelerator == "tenstorrent"
    if use_tt:
        from tt_boltz.tenstorrent import set_fast_mode
        set_fast_mode(fast)
    device = torch.device("cpu")

    try:
        yield {"type": "progress", "stage": "msa", "step": 0, "total": 1}
        ccd = load_canonicals(mol_dir)
        feats, input_struct = prepare_sequence_features(
            sequence, ccd, mol_dir, msa_dir,
            Boltz2Tokenizer(), Boltz2Featurizer(),
            use_msa_server=use_msa_server, msa_server_url=msa_server_url,
        )
    except Exception as e:
        yield {"type": "error", "message": f"Preprocessing failed: {e}"}
        return

    yield {"type": "progress", "stage": "prep", "step": 1, "total": 1}
    batch = to_batch(feats, device)

    yield {"type": "progress", "stage": "loading", "step": 1, "total": 1}
    model = Boltz2.load_from_checkpoint(
        cache / "boltz2_conf.ckpt",
        predict_args={
            "recycling_steps": recycling_steps,
            "sampling_steps": sampling_steps,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
        },
        diffusion_process_args=_DIFFUSION_ARGS,
        pairformer_args=_PAIRFORMER_ARGS,
        msa_args=_MSA_ARGS,
        steering_args=_STEERING_ARGS,
        use_kernels=False,
        use_tenstorrent=use_tt,
    ).eval().to(device)

    atom_mask = batch["atom_pad_mask"].squeeze(0).bool()
    make_intermediate_frame = _build_intermediate_frame_maker(
        input_struct.remove_invalid_chains(), atom_mask,
    )

    # Bridge the model's progress_fn (called from the model thread) to
    # this generator (running on the request thread).
    events: queue.Queue = queue.Queue()
    sent_intermediate_template = False

    def progress_fn(stage: str, step: int = 0, total: int = 0,
                    coords: Optional[torch.Tensor] = None, **_):
        nonlocal sent_intermediate_template
        event = {"type": "progress", "stage": stage, "step": step, "total": total}
        should_stream_coords = (
            coords is not None
            and stage == "diffusion"
            and (step % INTERMEDIATE_FRAME_INTERVAL == 0 or step >= total)
        )
        if should_stream_coords:
            frame = make_intermediate_frame(coords, include_cif=not sent_intermediate_template)
            if frame is not None:
                sent_intermediate_template = sent_intermediate_template or "cif" in frame
                event = {
                    "type": "intermediate", "stage": stage,
                    "step": step, "total": total, **frame,
                }
        events.put(event)

    model.progress_fn = progress_fn

    result: dict = {"pred": None, "error": None}

    def run_inference():
        try:
            with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                result["pred"] = model.predict_step(batch)
        except Exception as e:
            result["error"] = str(e)
        finally:
            events.put(None)

    thread = threading.Thread(target=run_inference, daemon=True)
    thread.start()

    try:
        while True:
            try:
                ev = events.get(timeout=0.1)
            except queue.Empty:
                continue
            if ev is None:
                break
            yield ev
    finally:
        # If the browser disconnects mid-stream, keep this generator alive
        # until the TT inference thread finishes so the demo never starts a
        # second prediction while the device is still busy.
        thread.join()

    if result["error"]:
        yield {"type": "error", "message": f"Prediction failed: {result['error']}"}
        return

    pred = result["pred"]
    if pred is None or pred.get("exception"):
        yield {"type": "error", "message": "Prediction failed"}
        return

    yield {"type": "progress", "stage": "saving", "step": 0, "total": 1}

    cif, confidence = finalize_prediction(pred, input_struct)
    yield {
        "type": "complete",
        "cif": cif,
        "confidence": confidence,
        "sequence_length": len(sequence),
    }
