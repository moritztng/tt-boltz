import argparse
import warnings
from itertools import permutations, product
from pathlib import Path

from Bio.PDB import MMCIFParser, Superimposer
from Bio.PDB.PDBExceptions import PDBConstructionWarning

warnings.filterwarnings('ignore', category=PDBConstructionWarning)


def get_ca_atoms(structure):
    """Extract CA atoms organized by chain ID."""
    return {
        chain.id: [res['CA'] for res in chain.get_residues() if 'CA' in res]
        for chain in structure[0]
    }


def group_chains_by_length(chains):
    """Group chain IDs by their length."""
    by_length = {}
    for chain_id, atoms in chains.items():
        length = len(atoms)
        by_length.setdefault(length, []).append(chain_id)
    return by_length


def generate_chain_matchings(pred_chains, truth_chains):
    """Generate all valid chain matchings (permute within same-length groups)."""
    pred_by_len = group_chains_by_length(pred_chains)
    truth_by_len = group_chains_by_length(truth_chains)
    
    # For each length group, generate all permutations
    for length in pred_by_len:
        if length not in truth_by_len:
            continue
        pred_group = pred_by_len[length]
        truth_group = truth_by_len[length]
        
        # Generate all permutations of truth chains for this length
        for truth_perm in permutations(truth_group):
            yield list(zip(pred_group, truth_perm))


def compute_rmsd(protein_name: str, model_idx: int = 0):
    """Compute RMSD with optimal chain matching."""
    pred_file = Path(f"boltz_results_{protein_name}/predictions/{protein_name}/{protein_name}_model_{model_idx}.cif")
    truth_file = Path(f"examples/ground_truth_structures/{protein_name}.cif")
    
    if not pred_file.exists():
        raise FileNotFoundError(f"Prediction file not found: {pred_file}")
    if not truth_file.exists():
        raise FileNotFoundError(f"Ground truth file not found: {truth_file}")
    
    parser = MMCIFParser(QUIET=True)
    pred_chains = get_ca_atoms(parser.get_structure("pred", str(pred_file)))
    truth_chains = get_ca_atoms(parser.get_structure("truth", str(truth_file)))
    
    print(f"Predicted chains: {[(c, len(pred_chains[c])) for c in sorted(pred_chains)]}")
    print(f"Ground truth chains: {[(c, len(truth_chains[c])) for c in sorted(truth_chains)]}")
    
    # Find best chain matching
    best_rmsd = float('inf')
    best_matching = None
    
    for matching in generate_chain_matchings(pred_chains, truth_chains):
        atoms_pred = [atom for pred_id, _ in matching for atom in pred_chains[pred_id]]
        atoms_truth = [atom for _, truth_id in matching for atom in truth_chains[truth_id]]
        
        if len(atoms_pred) != len(atoms_truth):
            continue
        
        sup = Superimposer()
        sup.set_atoms(atoms_pred, atoms_truth)
        
        if sup.rms < best_rmsd:
            best_rmsd = sup.rms
            best_matching = matching
    
    if best_matching is None:
        raise ValueError("No valid chain matching found")
    
    print("\nBest chain matching:")
    for pred_id, truth_id in best_matching:
        print(f"  {pred_id} -> {truth_id} ({len(pred_chains[pred_id])} residues)")
    
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
