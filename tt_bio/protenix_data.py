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

import math

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


def load_ref_conformers() -> dict:
    """Bundled standard-residue reference conformers (CCD ideal coordinates, in
    const.ref_atoms order). {3-letter resname: (n_atom, 3) tensor}."""
    import json
    import os
    path = os.path.join(os.path.dirname(__file__), "data", "protein_ref_conformers.json")
    return {k: torch.tensor(v, dtype=torch.float32) for k, v in json.load(open(path)).items()}


MSA_GAP_IDX = RESTYPE_DIM - 1  # protenix MSA vocab: gap '-' is the last class (31)


def _msa_token(c: str) -> int:
    """a3m residue char -> protenix MSA token: aa->0-19 (AF order), '-'->31, else->UNK(20)."""
    if c == "-":
        return MSA_GAP_IDX
    i = _AA1_TO_IDX.get(c.upper(), -1)
    return i if 0 <= i < 20 else 20


def _parse_a3m(a3m: str) -> list[str]:
    """a3m text -> list of aligned sequence strings (query first)."""
    seqs, buf = [], []
    for line in a3m.splitlines():
        if line.startswith(">"):
            if buf:
                seqs.append("".join(buf)); buf = []
        elif line:
            buf.append(line.strip())
    if buf:
        seqs.append("".join(buf))
    return seqs


def protein_msa_features(a3m: str, query: str) -> dict | None:
    """Build protenix MSA features from an a3m aligned to `query`.

    a3m convention: query is row 0; uppercase = match columns, lowercase = insertions
    (deletions relative to the query), '-' = gap. Per protenix
    (data/msa/msa_featurizer.py): profile = per-column token frequency over 32 classes
    (incl. gap), deletion_value = arctan(deletions/3)*2/pi, has_deletion = clip(deletions,0,1),
    deletion_mean = raw column-mean deletions. Returns None if the a3m is not aligned to this
    exact query (caller falls back to single-sequence)."""
    rows = _parse_a3m(a3m)
    if not rows:
        return None

    def featurize(seq: str):
        toks, dels, d = [], [], 0
        for ch in seq:
            if ch.islower():          # insertion relative to query
                d += 1
            else:                      # match column (uppercase residue or '-')
                toks.append(_msa_token(ch)); dels.append(d); d = 0
        return toks, dels

    n_tok = len(query)
    if len(featurize(rows[0])[0]) != n_tok:
        return None                    # a3m query not aligned to this exact sequence
    msa, delmat, seen = [], [], set()
    for seq in rows:
        if seq in seen:                # dedup identical alignment rows (incl. insertions)
            continue
        seen.add(seq)
        toks, dels = featurize(seq)
        if len(toks) != n_tok:
            continue
        msa.append(toks); delmat.append(dels)
    M = torch.tensor(msa, dtype=torch.long)
    DM = torch.tensor(delmat, dtype=torch.float32)
    profile = torch.stack([(M == k).float().mean(0) for k in range(RESTYPE_DIM)], dim=-1)
    return {
        "msa": M,
        "has_deletion": DM.clamp(0.0, 1.0),
        "deletion_value": torch.atan(DM / 3.0) * (2.0 / math.pi),
        "profile": profile,
        "deletion_mean": DM.mean(0),
    }


def build_protein_features(sequence: str, a3m: str | None = None) -> dict:
    """Full model-ready input_feature_dict for a single protein chain from a one-letter
    sequence. Combines token + atom features (bundled reference conformers). If `a3m` is
    given (an alignment whose query matches `sequence`), the single-sequence MSA features
    are replaced with the real MSA features; otherwise it folds single-sequence. The model
    (tt_bio.protenix.Protenix.fold) regenerates relp / d_lm / v_lm / mask_trunked internally,
    so this is everything fold needs."""
    aatype = aatype_from_sequence(sequence)
    feats = protein_token_features(aatype)
    feats.update(protein_atom_features(aatype, load_ref_conformers()))
    if a3m:
        msa_feats = protein_msa_features(a3m, sequence)
        if msa_feats is not None:
            feats.update(msa_feats)
    return feats


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
    elem_idx, name_chars, tokatom, disto_rep = [], [], [], []
    for t, aa in enumerate(aatype.tolist()):
        res = letter_to_res[RESTYPE_ORDER[aa]] if aa < len(RESTYPE_ORDER) else "UNK"
        atoms = list(const.ref_atoms[res])
        disto_atom = const.res_to_disto_atom.get(res, "CA")  # distogram rep atom (CB, or CA for GLY)
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
            tokatom.append(k)                       # index of this atom within its token
            disto_rep.append(1.0 if nm == disto_atom else 0.0)  # distogram rep atom (res_to_disto_atom)
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
        "atom_to_tokatom_idx": torch.tensor(tokatom, dtype=torch.long),
        "distogram_rep_atom_mask": torch.tensor(disto_rep),
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
