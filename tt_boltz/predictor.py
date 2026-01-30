"""Streaming protein structure prediction."""

import json
import os
import pickle
import queue
import tempfile
import threading
import warnings
from dataclasses import replace
from pathlib import Path
from typing import Generator

import numpy as np
import torch
from rdkit import Chem

from tt_boltz.data import const
from tt_boltz.data.featurizer import Boltz2Featurizer
from tt_boltz.data.mol import load_canonicals, load_molecules
from tt_boltz.data.msa import run_mmseqs2
from tt_boltz.data.parse import parse_csv, parse_yaml
from tt_boltz.data.tokenize import Boltz2Tokenizer
from tt_boltz.data.types import Coords, Input, Interface, MSA, Record, StructureV2
from tt_boltz.data.write import to_mmcif
from tt_boltz.boltz2 import Boltz2
from tt_boltz.main import download_all


# Valid amino acid characters
AMINO_ACIDS = set("ACDEFGHIKLMNPQRSTVWY")


def validate_sequence(sequence: str) -> tuple[bool, str]:
    """Validate an amino acid sequence.
    
    Returns (is_valid, error_message).
    """
    sequence = sequence.upper().strip()
    
    if not sequence:
        return False, "Sequence is empty"
    
    if len(sequence) < 10:
        return False, "Sequence too short (minimum 10 residues)"
    
    if len(sequence) > 1024:
        return False, "Sequence too long (maximum 1024 residues)"
    
    invalid_chars = set(sequence) - AMINO_ACIDS
    if invalid_chars:
        return False, f"Invalid characters: {', '.join(sorted(invalid_chars))}"
    
    return True, ""


def predict_structure(
    sequence: str,
    cache_dir: Path = None,
    use_msa_server: bool = True,
    accelerator: str = "tenstorrent",
    recycling_steps: int = 3,
    sampling_steps: int = 200,
    msa_server_url: str = "https://api.colabfold.com",
) -> Generator[dict, None, None]:
    """
    Predict protein structure from amino acid sequence.
    
    Yields progress events and final result as dictionaries.
    
    Yields:
        {"type": "progress", "stage": str, "message": str, "step": int, "total": int}
        {"type": "complete", "cif": str, "confidence": dict}
        {"type": "error", "message": str}
    """
    # Normalize sequence
    sequence = sequence.upper().strip()
    
    # Validate
    is_valid, error_msg = validate_sequence(sequence)
    if not is_valid:
        yield {"type": "error", "message": error_msg}
        return
    
    # Setup
    yield {"type": "progress", "stage": "setup", "message": "Initializing...", "step": 0, "total": 1}
    
    warnings.filterwarnings("ignore")
    torch.set_grad_enabled(False)
    Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    
    # Setup paths
    cache = Path(cache_dir or os.environ.get("BOLTZ_CACHE", Path("~/.boltz").expanduser())).expanduser()
    cache.mkdir(parents=True, exist_ok=True)
    
    # Download models if needed
    yield {"type": "progress", "stage": "setup", "message": "Checking models...", "step": 0, "total": 1}
    download_all(cache)
    
    mol_dir = cache / "mols"
    
    # Create temporary directory for this prediction
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        
        # Create YAML input file
        yaml_content = f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {sequence}
"""
        yaml_path = tmpdir / "input.yaml"
        yaml_path.write_text(yaml_content)
        
        # Setup directories
        msa_dir = tmpdir / "msa"
        msa_dir.mkdir()
        records_dir = tmpdir / "processed" / "records"
        records_dir.mkdir(parents=True)
        structures_dir = tmpdir / "processed" / "structures"
        structures_dir.mkdir(parents=True)
        msa_out_dir = tmpdir / "processed" / "msa"
        msa_out_dir.mkdir(parents=True)
        constraints_dir = tmpdir / "processed" / "constraints"
        constraints_dir.mkdir(parents=True)
        templates_dir = tmpdir / "processed" / "templates"
        templates_dir.mkdir(parents=True)
        mols_dir = tmpdir / "processed" / "mols"
        mols_dir.mkdir(parents=True)
        
        # Load CCD molecules
        ccd = load_canonicals(mol_dir)
        
        # Parse input
        yield {"type": "progress", "stage": "preprocess", "message": "Parsing input...", "step": 0, "total": 1}
        try:
            target = parse_yaml(yaml_path, ccd, mol_dir, boltz2=True)
        except Exception as e:
            yield {"type": "error", "message": f"Failed to parse input: {str(e)}"}
            return
        
        tid = target.record.id
        
        # Generate MSA
        if use_msa_server:
            yield {"type": "progress", "stage": "msa", "message": "Generating MSA...", "step": 0, "total": 1}
            
            try:
                # Identify sequences needing MSA
                seqs_to_gen = {}
                for chain in target.record.chains:
                    if chain.mol_type == const.chain_type_ids["PROTEIN"] and chain.msa_id == 0:
                        msa_name = f"{tid}_{chain.entity_id}"
                        seqs_to_gen[msa_name] = target.sequences[chain.entity_id]
                        chain.msa_id = msa_dir / f"{msa_name}.csv"
                    elif chain.msa_id == 0:
                        chain.msa_id = -1
                
                if seqs_to_gen:
                    # Generate unpaired MSAs
                    seqs_list = list(seqs_to_gen.values())
                    unpaired = run_mmseqs2(
                        seqs_list, 
                        msa_dir / f"{tid}_unpaired_tmp", 
                        use_env=True,
                        use_pairing=False, 
                        host_url=msa_server_url,
                        pairing_strategy="greedy",
                    )
                    
                    # Write MSA CSV files
                    for i, name in enumerate(seqs_to_gen):
                        unpaired_seqs = unpaired[i].strip().splitlines()[1::2][:const.max_msa_seqs]
                        keys = [-1] * len(unpaired_seqs)
                        lines = ["key,sequence"] + [f"{k},{s}" for k, s in zip(keys, unpaired_seqs)]
                        (msa_dir / f"{name}.csv").write_text("\n".join(lines))
                    
            except Exception as e:
                yield {"type": "error", "message": f"MSA generation failed: {str(e)}"}
                return
        else:
            # No MSA - create dummy
            for chain in target.record.chains:
                if chain.mol_type == const.chain_type_ids["PROTEIN"] and chain.msa_id == 0:
                    msa_name = f"{tid}_{chain.entity_id}"
                    seq = target.sequences[chain.entity_id]
                    msa_path = msa_dir / f"{msa_name}.csv"
                    msa_path.write_text(f"key,sequence\n-1,{seq}")
                    chain.msa_id = msa_path
                elif chain.msa_id == 0:
                    chain.msa_id = -1
        
        yield {"type": "progress", "stage": "msa", "message": "MSA complete", "step": 1, "total": 1}
        
        # Process MSAs
        yield {"type": "progress", "stage": "preprocess", "message": "Processing MSA...", "step": 0, "total": 1}
        msa_map = {}
        for i, msa_id in enumerate(sorted({c.msa_id for c in target.record.chains if c.msa_id != -1})):
            msa_path = Path(msa_id)
            out_path = msa_out_dir / f"{tid}_{i}.npz"
            msa_map[msa_id] = f"{tid}_{i}"
            
            if not out_path.exists():
                msa = parse_csv(msa_path, const.max_msa_seqs)
                msa.dump(out_path)
        
        # Update chain MSA IDs
        for chain in target.record.chains:
            if chain.msa_id in msa_map:
                chain.msa_id = msa_map[chain.msa_id]
        
        # Save processed data
        target.residue_constraints.dump(constraints_dir / f"{tid}.npz")
        
        with (mols_dir / f"{tid}.pkl").open("wb") as f:
            pickle.dump(target.extra_mols, f)
        
        target.structure.dump(structures_dir / f"{tid}.npz")
        target.record.dump(records_dir / f"{tid}.json")
        
        # Setup directories dict for load_features
        dirs = {
            "structures": structures_dir,
            "msa": msa_out_dir,
            "constraints": constraints_dir,
            "templates": templates_dir,
            "mols": mols_dir,
            "pred": tmpdir / "predictions",
        }
        
        # Load features
        yield {"type": "progress", "stage": "preprocess", "message": "Featurizing...", "step": 0, "total": 1}
        
        tokenizer = Boltz2Tokenizer()
        featurizer = Boltz2Featurizer()
        canonicals = load_canonicals(mol_dir)
        
        record = Record.load(records_dir / f"{tid}.json")
        struct = StructureV2.load(structures_dir / f"{tid}.npz")
        
        # Load MSAs
        msas = {c.chain_id: MSA.load(msa_out_dir / f"{c.msa_id}.npz")
               for c in record.chains if c.msa_id != -1}
        
        # Load constraints
        constraints = None
        if constraints_dir.exists():
            constraint_path = constraints_dir / f"{record.id}.npz"
            if constraint_path.exists():
                from tt_boltz.data.types import ResidueConstraints
                constraints = ResidueConstraints.load(constraint_path)
        
        # Load extra molecules
        mol_path = mols_dir / f"{record.id}.pkl"
        extra_mols = pickle.load(mol_path.open("rb")) if mol_path.exists() else {}
        
        # Tokenize input
        inp = Input(struct, msa=msas, record=record, residue_constraints=constraints,
                   templates=None, extra_mols=extra_mols)
        tok = tokenizer.tokenize(inp)
        
        # Load molecules needed for featurization
        mols = {**canonicals, **extra_mols}
        needed = set(tok.tokens["res_name"].tolist()) - set(mols)
        if needed:
            mols.update(load_molecules(mol_dir, needed))
        
        # Featurize
        feats = featurizer.process(
            tok, 
            np.random.default_rng(42), 
            mols, 
            False, 
            const.max_msa_seqs,
            pad_to_max_seqs=False, 
            single_sequence_prop=0.0, 
            compute_frames=True,
            compute_constraint_features=True,
        )
        feats["record"] = record
        
        # Convert to batch
        device = torch.device("cpu")
        use_tt = accelerator == "tenstorrent"
        
        skip = {"all_coords", "all_resolved_mask", "crop_to_all_atom_map", "chain_symmetries",
                "amino_acids_symmetries", "ligand_symmetries", "record", "affinity_mw"}
        
        batch = {}
        for k, v in feats.items():
            if k in skip:
                batch[k] = [v] if k == "record" else v
            elif hasattr(v, 'unsqueeze'):
                batch[k] = v.unsqueeze(0).to(device)
            else:
                batch[k] = v
        
        # Load model
        yield {"type": "progress", "stage": "model", "message": "Loading model...", "step": 0, "total": 1}
        
        model = Boltz2.load_from_checkpoint(
            cache / "boltz2_conf.ckpt",
            predict_args={
                "recycling_steps": recycling_steps, 
                "sampling_steps": sampling_steps,
                "diffusion_samples": 1, 
                "max_parallel_samples": 1
            },
            diffusion_process_args={
                "step_scale": 1.5, "gamma_0": 0.8, "gamma_min": 1.0,
                "noise_scale": 1.003, "rho": 7, "sigma_min": 0.0001, "sigma_max": 160.0,
                "sigma_data": 16.0, "P_mean": -1.2, "P_std": 1.5,
                "coordinate_augmentation": True, "coordinate_augmentation_inference": False,
                "alignment_reverse_diff": True,
                "synchronize_sigmas": True
            },
            pairformer_args={"num_blocks": 64, "num_heads": 16, "dropout": 0.0, "v2": True},
            msa_args={"subsample_msa": False, "num_subsampled_msa": 1024, "use_paired_feature": True},
            steering_args={
                "fk_steering": False, "physical_guidance_update": False,
                "contact_guidance_update": True, "num_particles": 3, "fk_lambda": 4.0,
                "fk_resampling_interval": 3, "num_gd_steps": 20
            },
            use_kernels=False, 
            use_tenstorrent=use_tt,
        ).eval().to(device)
        
        yield {"type": "progress", "stage": "model", "message": "Model loaded", "step": 1, "total": 1}
        
        # Pre-load structure for intermediate visualization
        base_struct = StructureV2.load(structures_dir / f"{record.id}.npz").remove_invalid_chains()
        atom_mask = batch["atom_pad_mask"].squeeze(0).bool()
        
        def make_intermediate_cif(coords_tensor):
            """Generate CIF from intermediate coordinates."""
            try:
                coords = coords_tensor.float()[atom_mask].cpu().numpy()
                atoms = base_struct.atoms.copy()
                residues = base_struct.residues.copy()
                atoms["coords"] = coords
                atoms["is_present"] = True
                residues["is_present"] = True
                intermediate_struct = replace(
                    base_struct,
                    atoms=atoms,
                    residues=residues,
                    interfaces=np.array([], dtype=Interface),
                    coords=np.array([(x,) for x in coords], dtype=Coords)
                )
                return to_mmcif(intermediate_struct, plddts=None, boltz2=True)
            except Exception:
                return None
        
        # Use a queue for real-time progress updates from the model
        progress_queue = queue.Queue()
        result_holder = {"pred": None, "error": None}
        
        def progress_callback(stage, step, total, intermediate_coords=None):
            event = {
                "type": "progress",
                "stage": stage,
                "message": f"{stage.capitalize()} step {step}/{total}",
                "step": step,
                "total": total
            }
            
            # If we have intermediate coordinates, generate CIF for visualization
            if intermediate_coords is not None:
                cif = make_intermediate_cif(intermediate_coords)
                if cif:
                    event["type"] = "intermediate"
                    event["cif"] = cif
            
            progress_queue.put(event)
        
        def run_inference():
            try:
                with torch.no_grad(), torch.autocast(device_type=device.type, dtype=torch.bfloat16):
                    result_holder["pred"] = model.predict_step(batch, progress_callback=progress_callback)
            except Exception as e:
                result_holder["error"] = str(e)
            finally:
                progress_queue.put(None)  # Signal completion
        
        # Run prediction in background thread
        yield {"type": "progress", "stage": "inference", "message": "Starting prediction...", "step": 0, "total": 1}
        
        inference_thread = threading.Thread(target=run_inference)
        inference_thread.start()
        
        # Yield progress updates in real-time
        while True:
            try:
                update = progress_queue.get(timeout=0.1)
                if update is None:  # Inference complete
                    break
                yield update
            except queue.Empty:
                continue
        
        inference_thread.join()
        
        if result_holder["error"]:
            yield {"type": "error", "message": f"Prediction failed: {result_holder['error']}"}
            return
        
        pred = result_holder["pred"]
        if pred.get("exception"):
            yield {"type": "error", "message": "Prediction failed"}
            return
        
        yield {"type": "progress", "stage": "postprocess", "message": "Processing results...", "step": 0, "total": 1}
        
        # Load structure again and update with predicted coordinates
        struct = StructureV2.load(structures_dir / f"{record.id}.npz").remove_invalid_chains()
        
        # Extract coordinates
        mask_1d = pred["masks"].squeeze(0) if pred["masks"].dim() > 1 else pred["masks"]
        coords = pred["coords"][0][mask_1d.bool()].cpu().numpy()
        
        # Update structure with predicted coordinates
        atoms = struct.atoms.copy()
        residues = struct.residues.copy()
        atoms["coords"] = coords
        atoms["is_present"] = True
        residues["is_present"] = True
        
        new_struct = replace(
            struct, 
            atoms=atoms, 
            residues=residues,
            interfaces=np.array([], dtype=Interface),
            coords=np.array([(x,) for x in coords], dtype=Coords)
        )
        
        # Generate CIF
        plddt = pred.get("plddt", [None])[0]
        cif_content = to_mmcif(new_struct, plddt, boltz2=True)
        
        # Extract confidence metrics
        confidence = {}
        if "confidence_score" in pred:
            confidence["confidence_score"] = float(pred["confidence_score"][0].item())
        if "ptm" in pred:
            confidence["ptm"] = float(pred["ptm"][0].item())
        if "iptm" in pred:
            confidence["iptm"] = float(pred["iptm"][0].item())
        if "complex_plddt" in pred:
            confidence["complex_plddt"] = float(pred["complex_plddt"][0].item())
        
        yield {
            "type": "complete",
            "cif": cif_content,
            "confidence": confidence,
            "sequence_length": len(sequence),
        }
