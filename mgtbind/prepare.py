#!/usr/bin/env python3
"""Prepare Boltz-2 YAML inputs for MGTBind molecular glue ternary complexes.

Reads examples.csv, filters AF3-modeled structures, and generates one YAML
per complex plus a reference CSV with AF3 baseline scores.

Usage:
    python prepare.py                          # all structures, no size limit
    python prepare.py --num 100 --max_aa 1500  # random subset
    python prepare.py --seed 123               # reproducible sampling

Outputs:
    mgtbind_inputs/          YAML files ready for tt-boltz predict
    mgtbind/sample.csv       Reference CSV with AF3 scores for evaluate.py
"""

import argparse
import csv
import random
import string
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def make_chain_ids(n: int) -> list[str]:
    """Return n unique chain IDs: A … Z, AA … ZZ."""
    ids = []
    for c in string.ascii_uppercase:
        ids.append(c)
        if len(ids) == n:
            return ids
    for c1 in string.ascii_uppercase:
        for c2 in string.ascii_uppercase:
            ids.append(c1 + c2)
            if len(ids) == n:
                return ids
    raise ValueError(f"Cannot generate {n} chain IDs")


def build_yaml(protein_a_seq: str, protein_b_seq: str, smiles: str) -> tuple[str, int, int]:
    """Build a Boltz-2 YAML input from sequences and SMILES.

    Underscores in protein sequences denote multi-chain assemblies;
    each segment becomes a separate protein chain.

    Returns (yaml_str, mg_asym_id, total_chains).
    """
    a_chains = protein_a_seq.split("_")
    b_chains = protein_b_seq.split("_")
    total_chains = len(a_chains) + len(b_chains) + 1  # +1 for ligand
    cids = make_chain_ids(total_chains)

    lines = ["version: 1", "sequences:"]
    idx = 0

    for sub in a_chains:
        lines += [f"  - protein:", f"      id: {cids[idx]}", f"      sequence: {sub}"]
        idx += 1

    for sub in b_chains:
        lines += [f"  - protein:", f"      id: {cids[idx]}", f"      sequence: {sub}"]
        idx += 1

    mg_asym_id = idx
    lines += [f"  - ligand:", f"      id: {cids[idx]}", f"      smiles: '{smiles}'"]

    return "\n".join(lines) + "\n", mg_asym_id, total_chains


def main():
    parser = argparse.ArgumentParser(description="Prepare MGTBind inputs for Boltz-2")
    parser.add_argument("--num", type=int, default=0,
                        help="Number of structures to sample (0 = all)")
    parser.add_argument("--max_aa", type=int, default=0,
                        help="Max total amino acids across all chains (0 = no limit)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    parser.add_argument("--csv", type=str, default=str(ROOT / "examples.csv"),
                        help="Path to examples.csv")
    parser.add_argument("--out_dir", type=str, default=str(ROOT / "mgtbind_inputs"),
                        help="Output directory for YAML files")
    args = parser.parse_args()

    with open(args.csv) as f:
        rows = list(csv.DictReader(f))

    # Filter: AF3-modeled, valid SMILES, optional size limit
    eligible = []
    for r in rows:
        if r["structure_determination_method"] != "AlphaFold3 Modeled":
            continue
        if not r["canonical_smiles"].strip():
            continue
        total_aa = len(r["protein_a_seq"].replace("_", "")) + len(r["protein_b_seq"].replace("_", ""))
        if args.max_aa > 0 and total_aa > args.max_aa:
            continue
        eligible.append(r)

    limit_str = f" (≤ {args.max_aa} AA)" if args.max_aa > 0 else ""
    print(f"Eligible structures{limit_str}: {len(eligible)}")

    # Sample or use all
    if args.num > 0:
        if args.num > len(eligible):
            print(f"WARNING: requested {args.num} but only {len(eligible)} available")
            args.num = len(eligible)
        random.seed(args.seed)
        selected = random.sample(eligible, args.num)
    else:
        selected = eligible

    # Sort by total AA for predictable ordering
    selected.sort(key=lambda r: len(r["protein_a_seq"].replace("_", "")) + len(r["protein_b_seq"].replace("_", "")))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ref_rows = []
    for r in selected:
        name = f"mgtbind_{r['id']}"
        yaml_str, mg_asym_id, total_chains = build_yaml(
            r["protein_a_seq"], r["protein_b_seq"], r["canonical_smiles"]
        )
        (out_dir / f"{name}.yaml").write_text(yaml_str)

        total_aa = len(r["protein_a_seq"].replace("_", "")) + len(r["protein_b_seq"].replace("_", ""))
        ref_rows.append({
            "name": name,
            "csv_id": r["id"],
            "compound_id": r["compound_id"],
            "protein_a_name": r["protein_a_name"],
            "protein_b_name": r["protein_b_name"],
            "total_aa": total_aa,
            "n_protein_a_chains": len(r["protein_a_seq"].split("_")),
            "n_protein_b_chains": len(r["protein_b_seq"].split("_")),
            "total_chains": total_chains,
            "mg_asym_id": mg_asym_id,
            "moa_type": r["moa_type"],
            "af3_ranking_score": r["highest_ranking_score"],
            "af3_mg_chain_iptm": r["highest_mg_chain_iptm"],
        })

    ref_path = Path(__file__).parent / "sample.csv"
    with open(ref_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ref_rows[0].keys())
        writer.writeheader()
        writer.writerows(ref_rows)

    print(f"Generated {len(selected)} YAML files in {out_dir}/")
    print(f"Reference CSV: {ref_path}")
    print(f"\nPredict:")
    print(f"  tt-boltz predict {out_dir} --out_dir {out_dir.parent} --use_msa_server --diffusion_samples 5")
    print(f"\nEvaluate:")
    print(f"  python {Path(__file__).parent / 'evaluate.py'}")


if __name__ == "__main__":
    main()
