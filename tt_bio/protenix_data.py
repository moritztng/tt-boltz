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

# CCD formal charges on amino-acid atoms (the v2 reference's neutral-CCD convention:
# only the protonated guanidinium / ammonium nitrogens carry +1; validated vs golden).
_FORMAL_CHARGE = {("ARG", "NH2"): 1.0, ("LYS", "NZ"): 1.0}


def aatype_from_sequence(seq: str) -> torch.Tensor:
    """One-letter protein sequence -> aatype indices (unknown -> 20)."""
    return torch.tensor([_AA1_TO_IDX.get(c.upper(), 20) for c in seq], dtype=torch.long)


def protein_atom_features(aatype: torch.Tensor, conformers: dict) -> dict:
    """Atom-level features for a single offline protein chain.

    Reuses tt_bio.data.const.ref_atoms (per-residue atom names, in the canonical CCD order
    that matches the v2 reference exactly -- validated 20/20 residue types). Per-atom:
      ref_element  = one_hot(atomic_number - 1, 128)   (C->5, N->6, O->7, ... validated)
      ref_charge   = 0  (neutral CCD reference)
      ref_atom_name_chars = one_hot over 64 of (ord(c)-32) for 4 chars (space-padded)
      ref_mask     = 1
      atom_to_token_idx = residue index per atom; ref_space_uid = same (per-residue frame)
      ref_pos      = conformers[res] (a valid per-residue reference conformer; the reference
                     uses a STOCHASTIC RDKit conformer, so any valid one folds correctly).
    conformers: {3-letter resname: (n_atom, 3) local coords in ref_atoms order}.
    """
    from .data import const
    letter_to_res = {v: k for k, v in const.prot_token_to_letter.items()}
    z_of = const.element_to_atomic_num  # symbol -> atomic number
    n_tok = aatype.shape[0]
    ref_pos, ref_charge, ref_mask, a2t, ruid = [], [], [], [], []
    elem_idx, name_chars = [], []
    for t, aa in enumerate(aatype.tolist()):
        res = letter_to_res[RESTYPE_ORDER[aa]] if aa < len(RESTYPE_ORDER) else "UNK"
        atoms = list(const.ref_atoms[res])
        conf = torch.as_tensor(conformers[res], dtype=torch.float32)
        if t == n_tok - 1:  # C-terminal residue carries the extra OXT (carboxylate) oxygen
            atoms = atoms + ["OXT"]
            # synthesize OXT as the carboxylate mirror of O through C (any valid ref conformer)
            c_i, o_i = const.ref_atoms[res].index("C"), const.ref_atoms[res].index("O")
            conf = torch.cat([conf, (2 * conf[c_i] - conf[o_i])[None]], 0)
        for k, nm in enumerate(atoms):
            elem_idx.append(z_of[nm[0]] - 1)
            ref_charge.append(_FORMAL_CHARGE.get((res, nm), 0.0))
            ref_mask.append(1.0)
            a2t.append(t); ruid.append(t)
            padded = (nm + "    ")[:4]
            name_chars.append([ord(c) - 32 for c in padded])
        ref_pos.append(conf)
    ref_pos = torch.cat(ref_pos, 0)
    N = ref_pos.shape[0]
    return {
        "ref_pos": ref_pos,
        "ref_element": torch.nn.functional.one_hot(torch.tensor(elem_idx), const.num_elements).float(),
        "ref_charge": torch.tensor(ref_charge),
        "ref_atom_name_chars": torch.nn.functional.one_hot(torch.tensor(name_chars), 64).float(),
        "ref_mask": torch.tensor(ref_mask),
        "atom_to_token_idx": torch.tensor(a2t, dtype=torch.long),
        "ref_space_uid": torch.tensor(ruid, dtype=torch.long),
    }


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
