#!/usr/bin/env python3
"""Evaluate Boltz-2 predictions against AF3 baselines for MGTBind complexes.

Extracts two key metrics from Boltz-2 outputs:
  1. MG Chain ipTM  — average cross-chain ipTM between molecular glue and proteins
  2. Avg pLDDT of MG — average per-atom pLDDT of molecular glue (from CIF B-factors)

Compares against AF3 values from sample.csv.

Usage:
    python evaluate.py
    python evaluate.py --results_dir /path/to/boltz_results_mgtbind_inputs

Acceptance thresholds (from MGTBind):
    MG Chain ipTM > 0.68
    Avg pLDDT of MG > 70
"""

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


# ── Metric extraction ──────────────────────────────────────────────────────────


def extract_mg_plddt_from_cif(cif_path: Path) -> float | None:
    """Average pLDDT for molecular glue heavy atoms (HETATM records in CIF).

    pLDDT is stored in B_iso_or_equiv, already on a 0–100 scale.
    """
    if not cif_path.exists():
        return None

    text = cif_path.read_text()
    b_factors: list[float] = []

    in_atom_site = False
    columns: list[str] = []
    col_group = col_biso = col_type = -1

    for line in text.splitlines():
        s = line.strip()

        if s == "loop_":
            in_atom_site = False
            columns = []
            continue

        if s.startswith("_atom_site."):
            in_atom_site = True
            name = s.split(".")[1].strip()
            columns.append(name)
            if name == "group_PDB":
                col_group = len(columns) - 1
            elif name == "B_iso_or_equiv":
                col_biso = len(columns) - 1
            elif name == "type_symbol":
                col_type = len(columns) - 1
            continue

        if in_atom_site and columns and not s.startswith("_"):
            if not s or s.startswith("#") or s == "loop_":
                in_atom_site = False
                continue

            parts = line.split()
            if len(parts) < len(columns):
                continue

            if col_group >= 0 and col_biso >= 0 and parts[col_group] == "HETATM":
                if col_type >= 0 and parts[col_type] == "H":
                    continue  # skip hydrogens
                try:
                    b_factors.append(float(parts[col_biso]))
                except ValueError:
                    pass

    return sum(b_factors) / len(b_factors) if b_factors else None


def extract_mg_chain_iptm(result: dict, mg_asym_id: int) -> float | None:
    """Average cross-chain ipTM between the molecular glue and all protein chains.

    pair_chains_iptm[i][j] is the TM-score of chain j in chain i's frame.
    We average pair_chains_iptm[mg][p] for all protein chains p.
    """
    pci = result.get("pair_chains_iptm")
    if not pci:
        return None

    mg_key = str(mg_asym_id)
    if mg_key not in pci:
        return None

    values = [v for k, v in pci[mg_key].items() if k != mg_key and v > 0]
    return sum(values) / len(values) if values else None


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Evaluate MGTBind Boltz-2 predictions")
    parser.add_argument("--sample_csv", type=str,
                        default=str(Path(__file__).parent / "sample.csv"))
    parser.add_argument("--results_dir", type=str,
                        default=str(ROOT / "boltz_results_mgtbind_inputs"))
    args = parser.parse_args()

    sample_csv = Path(args.sample_csv)
    results_dir = Path(args.results_dir)

    if not sample_csv.exists():
        sys.exit(f"ERROR: {sample_csv} not found — run prepare.py first")
    if not results_dir.exists():
        sys.exit(f"ERROR: {results_dir} not found — run tt-boltz predict first")

    results_path = results_dir / "results.json"
    if not results_path.exists():
        sys.exit(f"ERROR: {results_path} not found")

    with open(sample_csv) as f:
        samples = list(csv.DictReader(f))
    with open(results_path) as f:
        results_by_id = {r["id"]: r for r in json.load(f)}

    # ── Collect per-structure metrics ──────────────────────────────────────

    rows: list[dict] = []
    for s in samples:
        name = s["name"]
        mg_asym_id = int(s["mg_asym_id"])
        af3_iptm = float(s["af3_mg_chain_iptm"])

        boltz = results_by_id.get(name)
        if not boltz or boltz.get("status") != "ok":
            rows.append({**s, "boltz_status": "failed" if boltz else "missing",
                         "boltz_confidence_score": "", "boltz_mg_chain_iptm": "",
                         "boltz_mg_plddt": "", "boltz_iptm": "", "boltz_ligand_iptm": ""})
            continue

        mg_iptm = extract_mg_chain_iptm(boltz, mg_asym_id)
        mg_plddt = extract_mg_plddt_from_cif(results_dir / "structures" / f"{name}.cif")

        rows.append({
            **s,
            "boltz_status": "ok",
            "boltz_confidence_score": f"{boltz.get('confidence_score', 0):.4f}",
            "boltz_iptm": f"{boltz.get('iptm', 0):.4f}",
            "boltz_ligand_iptm": f"{boltz.get('ligand_iptm', 0):.4f}",
            "boltz_mg_chain_iptm": f"{mg_iptm:.4f}" if mg_iptm is not None else "N/A",
            "boltz_mg_plddt": f"{mg_plddt:.1f}" if mg_plddt is not None else "N/A",
        })

    # ── Print table ───────────────────────────────────────────────────────

    W = 120
    print()
    print("=" * W)
    print(f"{'Name':<24} {'AA':>5} {'AF3 ipTM':>9} {'Boltz ipTM':>11} "
          f"{'Δ':>7} {'Boltz pLDDT':>12} {'Status':>8}")
    print("-" * W)

    ok = 0
    boltz_iptms: list[float] = []
    af3_iptms: list[float] = []
    boltz_plddts: list[float] = []
    deltas: list[float] = []

    for r in rows:
        af3 = float(r["af3_mg_chain_iptm"])

        if r["boltz_status"] != "ok":
            print(f"{r['name']:<24} {r['total_aa']:>5} {af3:>9.4f} "
                  f"{'---':>11} {'---':>7} {'---':>12} {r['boltz_status']:>8}")
            continue

        ok += 1
        bi = r["boltz_mg_chain_iptm"]
        bp = r["boltz_mg_plddt"]

        if bi != "N/A":
            biv = float(bi)
            d = biv - af3
            boltz_iptms.append(biv)
            af3_iptms.append(af3)
            deltas.append(d)
            d_str = f"{d:>+.4f}"
        else:
            d_str = "  N/A"

        if bp != "N/A":
            boltz_plddts.append(float(bp))

        print(f"{r['name']:<24} {r['total_aa']:>5} {af3:>9.4f} "
              f"{bi:>11} {d_str:>7} {bp:>12} {r['boltz_status']:>8}")

    print("=" * W)

    # ── Summary statistics ────────────────────────────────────────────────

    n = len(rows)
    print(f"\n{'SUMMARY':=^60}")
    print(f"  Structures: {n}  ({ok} ok, {n - ok} failed/missing)")

    if boltz_iptms:
        avg_b = sum(boltz_iptms) / len(boltz_iptms)
        avg_a = sum(af3_iptms) / len(af3_iptms)
        avg_d = sum(deltas) / len(deltas)
        wins = sum(1 for d in deltas if d > 0)
        iptm_pass_b = sum(1 for v in boltz_iptms if v > 0.68)
        iptm_pass_a = sum(1 for v in af3_iptms if v > 0.68)

        print(f"\n  MG Chain ipTM (threshold > 0.68):")
        print(f"    AF3   avg: {avg_a:.4f}   pass: {iptm_pass_a}/{len(af3_iptms)}"
              f" ({100*iptm_pass_a/len(af3_iptms):.0f}%)")
        print(f"    Boltz avg: {avg_b:.4f}   pass: {iptm_pass_b}/{len(boltz_iptms)}"
              f" ({100*iptm_pass_b/len(boltz_iptms):.0f}%)")
        print(f"    Δ avg: {avg_d:+.4f}   Boltz > AF3: {wins}/{len(deltas)}")

    if boltz_plddts:
        avg_p = sum(boltz_plddts) / len(boltz_plddts)
        plddt_pass = sum(1 for v in boltz_plddts if v > 70)
        print(f"\n  Avg pLDDT of MG (threshold > 70):")
        print(f"    Boltz avg: {avg_p:.1f}   pass: {plddt_pass}/{len(boltz_plddts)}"
              f" ({100*plddt_pass/len(boltz_plddts):.0f}%)")

    if boltz_iptms and boltz_plddts:
        both = sum(1 for i in range(min(len(boltz_iptms), len(boltz_plddts)))
                   if boltz_iptms[i] > 0.68 and boltz_plddts[i] > 70)
        total = min(len(boltz_iptms), len(boltz_plddts))
        print(f"\n  Both criteria met: {both}/{total} ({100*both/total:.0f}%)")

    print()

    # ── Write comparison CSV ──────────────────────────────────────────────

    out_csv = Path(__file__).parent / "comparison.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Comparison CSV: {out_csv}")


if __name__ == "__main__":
    main()
