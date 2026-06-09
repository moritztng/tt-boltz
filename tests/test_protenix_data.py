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


def test_protein_atom_metadata_exact():
    """Atom-level metadata (element/charge/name/mask/indices) reproduces the v2 reference
    exactly (incl. C-terminal OXT and ARG/LYS formal charges). ref_pos is a stochastic
    conformer (not bit-matched) so it is excluded here."""
    from tt_bio.protenix_data import protein_atom_features, RESTYPE_ORDER
    from tt_bio.data import const
    ie = pickle.load(open(_GOLD, "rb"))["intermediates"]["input_embedder"]["in"][0]
    aatype = ie["restype"].argmax(-1); a2t = ie["atom_to_token_idx"].long(); ref_pos = ie["ref_pos"].float()
    l2r = {v: k for k, v in const.prot_token_to_letter.items()}
    conf = {}
    for t in range(len(aatype)):
        res = l2r[RESTYPE_ORDER[int(aatype[t])]]
        if res not in conf:
            idx = (a2t == t).nonzero().flatten()
            conf[res] = ref_pos[idx][:len(const.ref_atoms[res])].clone()
    f = protein_atom_features(aatype, conf)
    for k in ["ref_element", "ref_charge", "ref_atom_name_chars", "ref_mask",
              "atom_to_token_idx", "ref_space_uid"]:
        assert f[k].shape == ie[k].shape, f"{k} shape {tuple(f[k].shape)} != {tuple(ie[k].shape)}"
        assert torch.equal(f[k].float(), ie[k].float()), f"{k} mismatch"
