"""Exact parity of the tt-bio nucleic-acid featurization vs the Protenix-v2 reference.

Golden feature dicts are captured from the reference venv by
scripts/protenix_extract_na_gold.py into ~/protenix_na_gold.pkl. Every token- and
atom-level feature must match the reference exactly; ref_pos is exempt (the reference
uses a stochastic RDKit conformer, so any valid conformer geometry is acceptable)."""
import os
import pickle

import pytest
import torch

from tt_bio.protenix_data import build_complex_features

_GOLD = os.path.expanduser("~/protenix_na_gold.pkl")
_MOLS = os.path.expanduser("~/.boltz/mols")
pytestmark = pytest.mark.skipif(
    not (os.path.exists(_GOLD) and os.path.exists(_MOLS)),
    reason="NA golden pkl or CCD mols dir missing",
)

_EXACT = ["restype", "residue_index", "asym_id", "entity_id", "sym_id", "token_index",
          "ref_element", "ref_charge", "ref_atom_name_chars", "ref_mask",
          "atom_to_token_idx", "ref_space_uid", "atom_to_tokatom_idx",
          "distogram_rep_atom_mask", "token_bonds"]

_CASES = {
    "rna": [("GACUUA", None, "rna")],
    "dna": [("GATTCA", None, "dna")],
    "prot_rna": [("GSSGSSG", None, "protein"), ("GACUA", None, "rna")],
}


@pytest.mark.parametrize("name", list(_CASES))
def test_na_featurization_exact(name):
    gold = pickle.load(open(_GOLD, "rb"))[name]
    feats = build_complex_features(_CASES[name], mol_dir=_MOLS)
    for k in _EXACT:
        assert feats[k].shape == gold[k].shape, f"{name}/{k} shape {tuple(feats[k].shape)} != {tuple(gold[k].shape)}"
        assert torch.equal(feats[k].float(), gold[k].float()), f"{name}/{k} values differ"
    # ref_pos: only require same atom count + finite (stochastic reference conformer)
    assert feats["ref_pos"].shape == gold["ref_pos"].shape
    assert torch.isfinite(feats["ref_pos"]).all()
    # decoded atom names must match the reference order exactly
    names = ["".join(chr(c + 32) for c in row).strip()
             for row in feats["ref_atom_name_chars"].argmax(-1).tolist()]
    assert names == gold["_atom_names"]
