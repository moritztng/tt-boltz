"""Parity of the tt-bio ligand featurization vs the Protenix-v2 reference.

Golden dicts from scripts/protenix_extract_lig_gold.py -> ~/protenix_lig_gold.pkl.
CCD ligands (deterministic CCD atom order) must match the reference exactly, including
the per-atom tokenization, intra-ligand token_bonds, and the global ref_space_uid counter.
SMILES ligands use an independent rdkit embedding, so only modality invariants are checked
(atom count, all-UNK restype, all-distogram, element multiset, bond count)."""
import os
import pickle

import pytest
import torch

from tt_bio.protenix_data import build_complex_features

_GOLD = os.path.expanduser("~/protenix_lig_gold.pkl")
_MOLS = os.path.expanduser("~/.boltz/mols")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(_GOLD) and os.path.exists(_MOLS)),
    reason="ligand golden pkl or CCD mols dir missing",
)

_EXACT = ["restype", "residue_index", "asym_id", "entity_id", "sym_id", "token_index",
          "ref_element", "ref_charge", "ref_atom_name_chars", "ref_mask",
          "atom_to_token_idx", "ref_space_uid", "atom_to_tokatom_idx",
          "distogram_rep_atom_mask", "token_bonds"]

_CCD_CASES = {
    "ccd_sah": [("CCD_SAH", None, "ligand")],
    "prot_lig": [("GSSGSSG", None, "protein"), ("CCD_SAH", None, "ligand")],
}


@pytest.mark.parametrize("name", list(_CCD_CASES))
def test_ccd_ligand_exact(name):
    gold = pickle.load(open(_GOLD, "rb"))[name]
    feats = build_complex_features(_CCD_CASES[name], mol_dir=_MOLS)
    for k in _EXACT:
        assert feats[k].shape == gold[k].shape, f"{name}/{k} shape mismatch"
        assert torch.equal(feats[k].float(), gold[k].float()), f"{name}/{k} values differ"
    names = ["".join(chr(c + 32) for c in row).strip()
             for row in feats["ref_atom_name_chars"].argmax(-1).tolist()]
    assert names == gold["_atom_names"]


def test_smiles_ligand_structural():
    gold = pickle.load(open(_GOLD, "rb"))["smiles_tyr"]
    feats = build_complex_features([("N[C@@H](Cc1ccc(O)cc1)C(=O)O", None, "ligand")], mol_dir=_MOLS)
    assert feats["restype"].shape[0] == gold["restype"].shape[0]
    assert (feats["restype"].argmax(-1) == 20).all()                 # all UNK
    assert (feats["distogram_rep_atom_mask"] == 1).all()             # every atom is a token rep
    assert (feats["residue_index"] == 1).all() and (feats["ref_space_uid"] == 0).all()
    assert sorted((feats["ref_element"].argmax(-1) + 1).tolist()) == \
        sorted((gold["ref_element"].argmax(-1) + 1).tolist())        # same element composition
    assert float(feats["token_bonds"].sum()) == float(gold["token_bonds"].sum())
