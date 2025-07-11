
from Bio.PDB import MMCIFParser, Superimposer

# Load structures
parser = MMCIFParser()
structure0 = parser.get_structure("structure0", "boltz_results_prot/predictions/prot/prot_model_0.cif")
structure1 = parser.get_structure("structure1", "examples/ground_truth/prot.cif")

# Extract CA atoms from first model of both structures
atoms0 = [res['CA'] for res in structure0[0].get_residues() if 'CA' in res]
atoms1 = [res['CA'] for res in structure1[0].get_residues() if 'CA' in res]

# Match lengths (assumes corresponding residues)
min_len = min(len(atoms0), len(atoms1))
atoms0 = atoms0[:min_len]
atoms1 = atoms1[:min_len]

# Compute RMSD
sup = Superimposer()
sup.set_atoms(atoms0, atoms1)
print("RMSD:", sup.rms)
