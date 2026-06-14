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


# protenix STD_RESIDUES_WITH_GAP token names beyond protein: RNA 21-25, DNA 26-30, gap 31
_RNA_NAMES = ["A", "G", "C", "U", "N"]
_DNA_NAMES = ["DA", "DG", "DC", "DT", "DN"]


def restype_to_resname(idx: torch.Tensor) -> list:
    """Per-token restype index -> CCD residue name (protein 3-letter / RNA / DNA / UNK).
    Modality-agnostic; used by the structure writer to label residues for any entity type."""
    from .data import const
    l2r = {v: k for k, v in const.prot_token_to_letter.items()}
    out = []
    for i in idx.tolist():
        if i < 20:
            out.append(l2r[RESTYPE_ORDER[i]])
        elif i < 26:
            out.append("UNK" if i == 20 else _RNA_NAMES[i - 21])
        elif i < 31:
            out.append(_DNA_NAMES[i - 26])
        else:
            out.append("UNK")
    return out


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


def _parse_a3m_to_msa(a3m: str, query: str):
    """a3m aligned to `query` -> (msa_tokens (M,n) long, raw_deletions (M,n) float), deduped
    (query is row 0). Returns None if the a3m query isn't aligned to this exact sequence.
    a3m convention: uppercase = match columns, lowercase = insertions (deletions relative to
    the query), '-' = gap."""
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
    return torch.tensor(msa, dtype=torch.long), torch.tensor(delmat, dtype=torch.float32)


def _msa_to_features(M: torch.Tensor, DM: torch.Tensor) -> dict:
    """MSA tokens (M,N) + RAW deletion counts (M,N) -> protenix MSA features. Per protenix
    (data/msa/msa_featurizer.py): profile = per-column token frequency over 32 classes (incl.
    gap), deletion_value = arctan(deletions/3)*2/pi, has_deletion = clip(deletions,0,1),
    deletion_mean = raw column-mean deletions. Single source for the single- and multi-chain paths."""
    profile = torch.stack([(M == k).float().mean(0) for k in range(RESTYPE_DIM)], dim=-1)
    return {
        "msa": M,
        "has_deletion": DM.clamp(0.0, 1.0),
        "deletion_value": torch.atan(DM / 3.0) * (2.0 / math.pi),
        "profile": profile,
        "deletion_mean": DM.mean(0),
    }


def protein_msa_features(a3m: str, query: str) -> dict | None:
    """Single-chain protenix MSA features from an a3m aligned to `query` (None if unaligned)."""
    raw = _parse_a3m_to_msa(a3m, query)
    return _msa_to_features(*raw) if raw is not None else None


def build_protein_features(sequence: str, a3m: str | None = None) -> dict:
    """Single protein chain -> model-ready input_feature_dict. Thin wrapper over
    build_complex_features for the common one-chain case (back-compatible)."""
    return build_complex_features([(sequence, a3m)])


def build_complex_features(chains: list, mol_dir: str | None = None) -> dict:
    """Multi-chain biomolecular complex -> model-ready input_feature_dict.

    chains: list of (sequence, a3m_or_None[, mol_type]); mol_type is "protein" (default),
    "rna" or "dna". Identical (mol_type, sequence) pairs share an entity_id and get distinct
    sym_ids (homomer copies); each chain gets its own asym_id. Token/atom features are the
    per-chain features concatenated with global token indexing (protein atoms from bundled
    CCD conformers, nucleic-acid atoms from the CCD `mols` rdkit templates); residue_index
    restarts per chain (so the relative-position encoding sees chain breaks), token_index is
    global. The MSA is assembled BLOCK-DIAGONALLY (each chain's alignment over its own columns,
    gap elsewhere) on top of a shared query row -- the standard unpaired multi-chain assembly;
    NA chains contribute only the query row (single-sequence). The model (tt_bio.protenix)
    regenerates relp / d_lm / v_lm / mask_trunked from asym/entity/sym/residue/token indices.
    token_bonds = 0 (no inter-chain covalent bonds for plain polymer complexes)."""
    norm = [(e[0], e[1], e[2] if len(e) > 2 else "protein") for e in chains]
    conformers = load_ref_conformers()
    na_codes = {c for seq, _, mt in norm if mt != "protein" for c in _na_res_codes(seq, mt)}
    mols = {}
    if na_codes:
        from .data.mol import load_molecules
        mols = load_molecules(str(mol_dir or _default_mol_dir()), sorted(na_codes))
    N_tot = sum(len(seq) for seq, _, _ in norm)
    entity_of = {}                                   # (mol_type, sequence) -> entity_id
    sym_counter = {}                                 # entity_id -> next copy index
    restype, asym, entity, sym, resid = [], [], [], [], []
    atom_feats = []                                  # per-chain atom-feature dicts
    tok_off = 0
    per_chain_msa = []                               # (start_col, n_tok, raw_msa|None, restype_idx)
    for ci, (seq, a3m, mt) in enumerate(norm):
        rt_idx = seq_to_restype(seq, mt)                      # 32-class indices (0..30)
        n = rt_idx.shape[0]
        eid = entity_of.setdefault((mt, seq), len(entity_of))
        sid = sym_counter.get(eid, 0); sym_counter[eid] = sid + 1
        restype.append(torch.nn.functional.one_hot(rt_idx.clamp(max=RESTYPE_DIM - 1), RESTYPE_DIM).float())
        asym.append(torch.full((n,), ci, dtype=torch.long))
        entity.append(torch.full((n,), eid, dtype=torch.long))
        sym.append(torch.full((n,), sid, dtype=torch.long))
        resid.append(torch.arange(1, n + 1, dtype=torch.long))
        if mt == "protein":
            af = protein_atom_features(rt_idx, conformers)    # local token indices 0..n-1
        else:
            af = na_atom_features(_na_res_codes(seq, mt), mols)
        af["atom_to_token_idx"] = af["atom_to_token_idx"] + tok_off
        af["ref_space_uid"] = af["ref_space_uid"] + tok_off
        atom_feats.append(af)
        raw = _parse_a3m_to_msa(a3m, seq) if (mt == "protein" and a3m) else None
        per_chain_msa.append((tok_off, n, raw, rt_idx))
        tok_off += n

    # block-diagonal MSA assembled from RAW counts (shared query row 0; each chain's extra rows
    # over its own columns, gap elsewhere), then transformed once via _msa_to_features.
    GAP = MSA_GAP_IDX
    query = torch.cat([a for _, _, _, a in per_chain_msa])  # query restypes (row 0), 0..30
    msa_rows, del_rows = [query], [torch.zeros(N_tot)]
    for start, n, raw, aatype in per_chain_msa:
        if raw is None:
            continue
        M, DM = raw
        for r in range(1, M.shape[0]):                        # skip the chain's own query (row 0)
            row = torch.full((N_tot,), GAP, dtype=torch.long); row[start:start + n] = M[r]
            drow = torch.zeros(N_tot); drow[start:start + n] = DM[r]
            msa_rows.append(row); del_rows.append(drow)
    msa_feats = _msa_to_features(torch.stack(msa_rows, 0), torch.stack(del_rows, 0))

    feats = {
        "restype": torch.cat(restype, 0),
        "token_bonds": torch.zeros(N_tot, N_tot),
        "asym_id": torch.cat(asym, 0),
        "entity_id": torch.cat(entity, 0),
        "sym_id": torch.cat(sym, 0),
        "residue_index": torch.cat(resid, 0),
        "token_index": torch.arange(0, N_tot, dtype=torch.long),
        **msa_feats,
    }
    # concat atom features across chains (atom_to_token_idx / ref_space_uid already offset)
    for k in atom_feats[0]:
        feats[k] = torch.cat([af[k] for af in atom_feats], 0)
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
    return _assemble_atom_features(torch.cat(ref_pos, 0), elem_idx, ref_charge, ref_mask,
                                   a2t, ruid, tokatom, disto_rep, name_chars)


def _assemble_atom_features(ref_pos, elem_idx, ref_charge, ref_mask, a2t, ruid,
                            tokatom, disto_rep, name_chars) -> dict:
    """Pack per-atom lists into the model-ready atom feature tensors (shared by every
    modality's atom-feature builder). elem_idx = atomic_number - 1 (one_hot over 128);
    name_chars = per-atom [4] of ord(c)-32; the rest are 1:1 per-atom annotations."""
    from .data import const
    F = torch.nn.functional
    return {
        "ref_pos": ref_pos,
        "ref_element": F.one_hot(torch.tensor(elem_idx), const.num_elements).float(),
        "ref_charge": torch.tensor(ref_charge),
        "ref_atom_name_chars": F.one_hot(torch.tensor(name_chars), 64).float(),
        "ref_mask": torch.tensor(ref_mask),
        "atom_to_token_idx": torch.tensor(a2t, dtype=torch.long),
        "ref_space_uid": torch.tensor(ruid, dtype=torch.long),
        "atom_to_tokatom_idx": torch.tensor(tokatom, dtype=torch.long),
        "distogram_rep_atom_mask": torch.tensor(disto_rep),
    }


# Nucleic-acid residue tokenization (v2 STD_RESIDUES: RNA 21-25, DNA 26-30; N/DN = unknown).
_RNA_LETTER_IDX = {"A": 21, "G": 22, "C": 23, "U": 24}
_DNA_LETTER_IDX = {"A": 26, "G": 27, "C": 28, "T": 29}
_NA_PURINES = {"A", "G", "DA", "DG"}
_NA_PYRIMIDINES = {"C", "U", "DC", "DT"}


def seq_to_restype(seq: str, mol_type: str = "protein") -> torch.Tensor:
    """One-letter sequence -> 32-class restype indices for the given modality
    (protein 0-20, RNA 21-25, DNA 26-30); unknown letters map to the modality's UNK."""
    if mol_type == "protein":
        return aatype_from_sequence(seq)
    table, unk = (_RNA_LETTER_IDX, 25) if mol_type == "rna" else (_DNA_LETTER_IDX, 30)
    return torch.tensor([table.get(c.upper(), unk) for c in seq], dtype=torch.long)


def _na_res_codes(seq: str, mol_type: str) -> list:
    """Per-residue CCD codes for a nucleic-acid sequence (RNA: A/G/C/U/N; DNA: DA/.../DN)."""
    if mol_type == "rna":
        return [c.upper() if c.upper() in ("A", "G", "C", "U") else "N" for c in seq]
    return [("D" + c.upper()) if c.upper() in ("A", "G", "C", "T") else "DN" for c in seq]


def na_atom_features(res_codes: list, mols: dict) -> dict:
    """Atom-level features for one nucleic-acid chain (one CCD code per residue token).

    Mirrors protein_atom_features but sources heavy atoms (name / element / charge /
    reference-conformer coords) from the CCD rdkit Mol -- whose heavy-atom order matches
    the v2 reference RES_ATOMS_DICT exactly (validated). The 5'-terminal phosphate oxygen
    OP3 is kept only on the first residue and dropped from the rest (the reference's chain
    convention, analogous to protein OXT). Distogram rep atom: C4 for purines, C2 for
    pyrimidines, C1' for unknown (N/DN), per AF3 SI 4.4."""
    ref_pos, elem_idx, ref_charge, ref_mask = [], [], [], []
    a2t, ruid, tokatom, disto_rep, name_chars = [], [], [], [], []
    for t, code in enumerate(res_codes):
        mol = mols[code]
        conf = mol.GetConformer()
        disto = "C4" if code in _NA_PURINES else "C2" if code in _NA_PYRIMIDINES else "C1'"
        k = 0
        for a in mol.GetAtoms():
            if a.GetAtomicNum() <= 1:                     # skip hydrogens (heavy atoms only)
                continue
            nm = a.GetProp("name")
            if nm == "OP3" and t > 0:                     # free 5'-phosphate O only on first residue
                k += 1                                    # OP3 still occupies CCD slot 0 (atom_to_tokatom_idx)
                continue
            p = conf.GetAtomPosition(a.GetIdx())
            ref_pos.append([p.x, p.y, p.z])
            elem_idx.append(a.GetAtomicNum() - 1)
            ref_charge.append(float(a.GetFormalCharge()))
            ref_mask.append(1.0)
            a2t.append(t); ruid.append(t); tokatom.append(k)
            disto_rep.append(1.0 if nm == disto else 0.0)
            name_chars.append([ord(c) - 32 for c in (nm + "    ")[:4]])
            k += 1
    return _assemble_atom_features(torch.tensor(ref_pos, dtype=torch.float32), elem_idx,
                                   ref_charge, ref_mask, a2t, ruid, tokatom, disto_rep, name_chars)


def _default_mol_dir() -> str:
    """Best-effort location of the bundled CCD `mols` directory (CLI/worker pass mol_dir
    explicitly; this only backstops tests run from a checkout)."""
    import os
    for p in ("~/.boltz/mols", "~/.cache/tt_bio/mols"):
        ep = os.path.expanduser(p)
        if os.path.exists(ep):
            return ep
    return os.path.expanduser("~/.boltz/mols")


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
