import argparse
import warnings
from itertools import permutations
from pathlib import Path

from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def get_ca_atoms(cif_path: str):
    """Extract CA atoms organized by chain ID, keyed by label_seq_id (entity position).

    label_seq_id is the 1-based position in the entity sequence, which matches
    the sequential residue numbering used in Boltz predictions.
    """
    d = MMCIF2Dict(cif_path)

    # Build per-chain dicts: {chain_id: {label_seq_id(int): CA_atom}}
    # We need the full structure for actual atom objects.
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("s", cif_path)

    # Build a lookup: (chain_id, auth_seq_id) -> label_seq_id
    auth_to_label = {}
    for asym, auth, label in zip(
        d.get("_atom_site.label_asym_id", []),
        d.get("_atom_site.auth_seq_id", []),
        d.get("_atom_site.label_seq_id", []),
    ):
        if label != ".":
            key = (asym, int(auth))
            auth_to_label[key] = int(label)

    chains = {}
    for chain in structure[0]:
        ca_by_label = {}
        for res in chain.get_residues():
            if "CA" not in res:
                continue
            key = (chain.id, res.id[1])
            label_id = auth_to_label.get(key)
            if label_id is not None:
                ca_by_label[label_id] = res["CA"]
        chains[chain.id] = ca_by_label
    return chains


def compute_rmsd(protein_name: str, model_idx: int = 0):
    """Compute RMSD with optimal chain matching (handles partial coverage)."""
    if model_idx == 0:
        pred_file = Path(f"boltz_results_{protein_name}/structures/{protein_name}.cif")
    else:
        pred_file = Path(f"boltz_results_{protein_name}/structures/{protein_name}_model_{model_idx}.cif")
    truth_file = Path(f"examples/ground_truth_structures/{protein_name}.cif")
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")
    
    pred_chains = get_ca_atoms(str(pred_file))
    truth_chains = get_ca_atoms(str(truth_file))
    
    # Filter out empty chains
    pred_chains = {k: v for k, v in pred_chains.items() if v}
    truth_chains = {k: v for k, v in truth_chains.items() if v}
    
    print(f"Predicted chains: {[(c, len(pred_chains[c])) for c in sorted(pred_chains)]}")
    print(f"Ground truth chains: {[(c, len(truth_chains[c])) for c in sorted(truth_chains)]}")
    
    pred_ids = sorted(pred_chains)
    truth_ids = sorted(truth_chains)
    
    if len(pred_ids) != len(truth_ids):
        raise ValueError(
            f"Chain count mismatch: {len(pred_ids)} predicted vs {len(truth_ids)} ground truth"
        )
    
    # Try all permutations of truth chains, pick lowest RMSD
    best_rmsd = float('inf')
    best_matching = None
    best_n_common = 0
    
    def get_atoms_for_pair(pred_by_id, truth_by_id):
        """Match residues by rank when lengths are equal, by label_seq_id otherwise."""
        pred_list = list(pred_by_id.values())
        truth_list = list(truth_by_id.values())
        if len(pred_list) == len(truth_list):
            # Same length: match residue-by-residue in sequential order
            return pred_list, truth_list
        # Different lengths (partial coverage): match by label_seq_id
        common = sorted(set(pred_by_id) & set(truth_by_id))
        return [pred_by_id[k] for k in common], [truth_by_id[k] for k in common]

    for truth_perm in permutations(truth_ids):
        matching = list(zip(pred_ids, truth_perm))
        
        # Collect CA atoms across all chain pairs
        atoms_pred, atoms_truth = [], []
        for pred_id, truth_id in matching:
            ap, at = get_atoms_for_pair(pred_chains[pred_id], truth_chains[truth_id])
            atoms_pred.extend(ap)
            atoms_truth.extend(at)
        
        if len(atoms_pred) < 3:
            continue
        
        sup = Superimposer()
        sup.set_atoms(atoms_pred, atoms_truth)
        
        if sup.rms < best_rmsd:
            best_rmsd = sup.rms
            best_matching = matching
            best_n_common = len(atoms_pred)
    
    if best_matching is None:
        raise ValueError("No valid chain matching found")
    
    total_pred = sum(len(v) for v in pred_chains.values())
    total_truth = sum(len(v) for v in truth_chains.values())
    
    print(f"\nBest chain matching ({best_n_common} common residues "
          f"of {total_pred} predicted / {total_truth} ground truth):")
    for pred_id, truth_id in best_matching:
        ap, _ = get_atoms_for_pair(pred_chains[pred_id], truth_chains[truth_id])
        print(f"  {pred_id} -> {truth_id} ({len(ap)} common residues)")
    
    print(f"\n{'='*50}")
    print(f"Protein: {protein_name}, Model: {model_idx}")
    print(f"RMSD: {best_rmsd:.4f} Å")
    print(f"{'='*50}\n")
    
    return best_rmsd


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute RMSD between predicted and ground truth structures")
    parser.add_argument("protein", help="Protein name (e.g., hemoglobin, prot)")
    parser.add_argument("--model", type=int, default=0, help="Model index (default: 0)")
    args = parser.parse_args()
    compute_rmsd(args.protein, args.model)
