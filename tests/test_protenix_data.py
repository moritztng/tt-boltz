# Validate the tt_bio.protenix_data token-level featurizer (offline protein case)
# exactly reproduces the v2 reference token features. Gated on the golden feat dict.
import os, pickle
import pytest
import torch

_GOLD = os.path.expanduser("~/protenix_ref_out.pkl")
pytestmark = pytest.mark.skipif(not os.path.exists(_GOLD), reason="needs ~/protenix_ref_out.pkl")


def test_protein_token_features_exact():
    from tt_bio.protenix_data import protein_token_features, aatype_from_sequence, RESTYPE_ORDER
    ie = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder"]["in"][0]
    aatype = ie["restype"].argmax(-1)
    f = protein_token_features(aatype)
    for k in ["restype", "profile", "deletion_mean", "msa", "has_deletion", "deletion_value",
              "token_bonds", "asym_id", "entity_id", "sym_id", "residue_index", "token_index"]:
        assert f[k].shape == ie[k].shape, f"{k} shape {tuple(f[k].shape)} != {tuple(ie[k].shape)}"
        assert torch.equal(f[k].float(), ie[k].float()), f"{k} mismatch"
    # aatype_from_sequence round-trips the standard restype order
    seq = "".join(RESTYPE_ORDER[i] for i in aatype.tolist() if i < len(RESTYPE_ORDER))
    rt = aatype_from_sequence(seq)
    assert torch.equal(rt, torch.tensor([i for i in aatype.tolist() if i < len(RESTYPE_ORDER)]))
