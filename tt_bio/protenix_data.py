"""Protenix-v2 inference featurization (offline protein case) for tt-bio.

Builds the model-ready input_feature_dict that tt_bio.protenix.Protenix.fold consumes,
WITHOUT the protenix data package. The model already regenerates the derived features
(relp, d_lm/v_lm/mask_trunked) internally (tt_bio.protenix), so the featurizer only emits
base features:

  token-level (this module, exactly reproduced vs the reference -- see below):
    restype, profile, deletion_mean, msa, has_deletion, deletion_value, token_bonds,
    asym_id, residue_index, entity_id, token_index, sym_id
  atom-level (CCD residue templates; see protein_atom_features):
    ref_pos, ref_space_uid, ref_charge, ref_element, ref_atom_name_chars, ref_mask,
    atom_to_token_idx

Offline = single protein chain, no external MSA/template (msa = the query row; profile =
restype; deletions = 0; templates empty), matching the reference's use_msa/use_template=False
inference path. restype uses the standard AF order (A R N D C Q E G H I L K M F P S T W Y V).
"""

from __future__ import annotations

import torch

# standard AlphaFold restype order (index -> one-letter); index 7=G, 15=S (matches v2 golden)
RESTYPE_ORDER = "ARNDCQEGHILKMFPSTWYV"
RESTYPE_DIM = 32  # protenix restype width (20 aa + X/gap/other slots)
_AA1_TO_IDX = {c: i for i, c in enumerate(RESTYPE_ORDER)}


def aatype_from_sequence(seq: str) -> torch.Tensor:
    """One-letter protein sequence -> aatype indices (unknown -> 20)."""
    return torch.tensor([_AA1_TO_IDX.get(c.upper(), 20) for c in seq], dtype=torch.long)


def protein_token_features(aatype: torch.Tensor) -> dict:
    """Token-level features for a single offline protein chain from aatype (N,).

    Exactly reproduces the v2 reference token features for the no-MSA / no-template
    inference path (validated vs golden, tests/test_protenix_data.py):
      restype = one_hot(aatype, 32); profile = restype; msa = aatype[None] (1 row);
      deletion_mean/has_deletion/deletion_value = 0; token_bonds = 0; single chain
      (asym/entity/sym = 0); residue_index = 1..N (1-based); token_index = 0..N-1.
    """
    N = aatype.shape[0]
    restype = torch.nn.functional.one_hot(aatype.clamp(max=RESTYPE_DIM - 1), RESTYPE_DIM).float()
    zeros_n = torch.zeros(N)
    return {
        "restype": restype,
        "profile": restype.clone(),
        "deletion_mean": zeros_n.clone(),
        "msa": aatype[None, :].long(),
        "has_deletion": torch.zeros(1, N),
        "deletion_value": torch.zeros(1, N),
        "token_bonds": torch.zeros(N, N),
        "asym_id": torch.zeros(N, dtype=torch.long),
        "entity_id": torch.zeros(N, dtype=torch.long),
        "sym_id": torch.zeros(N, dtype=torch.long),
        "residue_index": torch.arange(1, N + 1, dtype=torch.long),
        "token_index": torch.arange(0, N, dtype=torch.long),
    }
