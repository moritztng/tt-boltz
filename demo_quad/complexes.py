"""Sixteen example complexes for the quad-card live demo.

Each entry is a :class:`Complex` carrying

    * ``name``    – short display label shown in the UI banner
    * ``yaml``    – full Boltz YAML input (parsed by
                    :mod:`tt_boltz.main.prepare_features`)
    * ``seq_len`` – total protein residues across all chains (ligands not counted)

Mix is intentional and heavy on interactions:

    * 6 single proteins — monomers (ubiquitin, lysozyme, cytochrome c,
      crambin) plus two oligomeric proteins whose native biological
      assembly happens to have multiple chains (insulin A+B, hemoglobin
      α₂β₂). All are labelled simply ``protein``.
    * 6 protein–ligand complexes — drug discovery flavor (HIV PR +
      indinavir, DHFR + methotrexate, CA + acetazolamide, streptavidin +
      biotin, trypsin + benzamidine, myoglobin + heme)
    * 4 protein–protein complexes — genuine interactions between
      *distinct* proteins (barnase·barstar, MDM2·p53 peptide,
      calmodulin·MLCK peptide, trypsin·BPTI)

The classifier in :func:`_build` distinguishes *interacting* proteins from
*subunits of one* protein because that distinction cannot be inferred from
chain count alone — it requires biological context. For the oligomers we
pass an explicit ``kind="protein"`` override.

Every entry's protein sequence(s) totals ≤ 768 residues. Sequences come
straight from the canonical RCSB FASTA download for each PDB ID — no
hand-typed sequences, no species substitutions. Ligand CCD codes were
verified against the corresponding wwPDB nonpolymer entity records.
"""

from __future__ import annotations

from dataclasses import dataclass


MAX_SEQ_LEN = 768   # hard ceiling per complex (sum across protein chains)


@dataclass(frozen=True)
class Complex:
    name: str          # display label
    yaml: str          # full Boltz YAML input
    seq_len: int       # total protein residues across all chains
    kind: str          # 'protein' | 'protein–ligand' | 'protein–protein'
                       # Note: 'protein' covers both single chains AND
                       # oligomeric proteins (subunits of one biological
                       # molecule). 'protein–protein' is reserved for
                       # *distinct* proteins interacting with each other.
    pdb: str           # source PDB entry, e.g. '1HSG'
    chain_count: int   # total protein chains (counting copies)
    ligand_ccds: tuple[str, ...] = ()  # CCD codes of any small-molecule ligands


@dataclass(frozen=True)
class _Chain:
    """Protein chain to include in a complex."""

    seq: str
    copies: int = 1   # number of identical chains (e.g. homodimer → 2)


@dataclass(frozen=True)
class _Ligand:
    """Small-molecule ligand referenced by its wwPDB CCD code."""

    ccd: str
    copies: int = 1


def _build(
    name: str,
    *parts: _Chain | _Ligand,
    pdb: str,
    kind: str | None = None,
) -> Complex:
    """Assemble a :class:`Complex` from chain / ligand parts.

    Chain identifiers are assigned sequentially (``A``, ``B``, ``C``, …) in
    the order parts are listed. Multi-copy entries become an id list so
    Boltz knows the chains share a sequence (e.g. a homodimer):

        >>> _build("HIV PR · indinavir",
        ...        _Chain("PQITLWQR...", copies=2),
        ...        _Ligand("MK1"),
        ...        pdb="1HSG")

    produces

        version: 1
        sequences:
          - protein:
              id: [A, B]
              sequence: PQITLWQR...
          - ligand:
              id: C
              ccd: MK1

    Auto-classification (``kind``) keeps the taxonomy deliberately small:

    * any small-molecule ligand present → ``protein–ligand``
    * single chain, no ligand            → ``protein``
    * multiple distinct chains, no ligand → ``protein–protein``

    The last default assumes the chains belong to *different* proteins
    interacting. For oligomeric proteins whose native biological unit
    happens to have multiple chains (hemoglobin α₂β₂, insulin A+B), pass
    ``kind="protein"`` explicitly — there's no way to tell that case
    apart from a true protein–protein interaction without biological
    context.
    """
    lines = ["version: 1", "sequences:"]
    seq_len = 0
    chain_idx = 0  # 0 → 'A', 1 → 'B', …
    chain_count = 0
    ligand_ccds: list[str] = []

    def _ids(start: int, n: int) -> str:
        ids = [chr(ord("A") + start + i) for i in range(n)]
        return ids[0] if n == 1 else "[" + ", ".join(ids) + "]"

    for part in parts:
        ids = _ids(chain_idx, part.copies)
        if isinstance(part, _Chain):
            lines.append("  - protein:")
            lines.append(f"      id: {ids}")
            lines.append(f"      sequence: {part.seq}")
            seq_len += len(part.seq) * part.copies
            chain_count += part.copies
        elif isinstance(part, _Ligand):
            lines.append("  - ligand:")
            lines.append(f"      id: {ids}")
            lines.append(f"      ccd: {part.ccd}")
            ligand_ccds.extend([part.ccd] * part.copies)
        else:  # pragma: no cover
            raise TypeError(f"unknown complex part: {part!r}")
        chain_idx += part.copies

    if kind is None:
        if ligand_ccds:
            # Ligand presence dominates — drug-discovery flavor wins.
            kind = "protein–ligand"
        elif chain_count == 1:
            kind = "protein"
        else:
            # Multiple chains, no ligand. Default to a true
            # protein–protein interaction; oligomeric proteins (e.g.
            # hemoglobin α₂β₂, insulin A+B) MUST pass kind="protein".
            kind = "protein–protein"

    return Complex(
        name=name,
        yaml="\n".join(lines) + "\n",
        seq_len=seq_len,
        kind=kind,
        pdb=pdb,
        chain_count=chain_count,
        ligand_ccds=tuple(ligand_ccds),
    )


# ── Canonical sequences from RCSB (PDB ID → FASTA) ──────────────────────────
# Pulled via https://www.rcsb.org/fasta/entry/<PDB_ID>. Each block lists the
# PDB entry the sequence is from. N-terminal 'X' placeholders for modified
# residues have been trimmed where present (e.g. 1HRC's acetyl-glycine).

# Single-chain proteins (rendered standalone in the rotation)
UBIQUITIN   = "MQIFVKTLTGKTITLEVEPSDTIENVKAKIQDKEGIPPDQQRLIFAGKQLEDGRTLSDYNIQKESTLHLVLRLRGG"                                                                                                                                                              # 1UBQ — 76 aa
CRAMBIN     = "TTCCPSIVARSNFNVCRLPGTPEAICATYTGCIIIPGATCPGDYAN"                                                                                                                                                                                                # 1CRN — 46 aa
LYSOZYME    = "KVFGRCELAAAMKRHGLDNYRGYSLGNWVCAAKFESNFNTQATNRNTDGSTDYGILQINSRWWCNDGRTPGSRNLCNIPCSALLSSDITASVNCAKKIVSDGNGMNAWVAWRNRCKGTDVQAWIRGCRL"                                                                                                              # 1HEL — 129 aa
CYT_C       = "GDVEKGKKIFVQKCAQCHTVEKGGKHKTGPNLHGLFGRKTGQAPGFTYTDANKNKGITWKEETLMEYLENPKKYIPGTKMIFAGIKKKTEREDLIAYLKKATNE"                                                                                                                                       # 1HRC — 104 aa (leading X trimmed)

# Protein chains paired with small-molecule ligands in the rotation
MYOGLOBIN   = "VLSEGEWQLVLHVWAKVEADVAGHGQDILIRLFKSHPETLEKFDRFKHLKTEAEMKASEDLKKHGVTVLTALGAILKKKGHHEAELKPLAQSHATKHKIPIKYLEFISEAIIHVLHSRHPGDFGADAQGAMNKALELFRKDIAAKYKELGYQG"                                                                                              # 1MBN — 153 aa
HIV_PROTEASE = "PQITLWQRPLVTIKIGGQLKEALLDTGADDTVLEEMSLPGRWKPKMIGGIGGFIKVRQYDQILIEICGHKAIGTVLVGPTPVNIIGRNLLTQIGCTLNF"                                                                                                                                              # 1HSG — 99 aa (homodimer)
CA_I        = "ASPDWGYDDKNGPEQWSKLYPIANGNNQSPVDIKTSETKHDTSLKPISVSYNPATAKEIINVGHSFHVNFEDNDNRSVLKGGPFSDSYRLFQFHFHWGSTNEHGSEHTVDGVKYSAELHVAHWNSAKYSSLAEAASKADGLAVIGVLMKVGEANPKLQKVLDALQAIKTKGKRAPFTNFDPSTLLPSSLDFWTYPGSLTHPPLYESVTWIICKESISVSSEQLAQFRSLLSNVEGDNAVPMQHNNRPTQPLKGRTVRASF"  # 1AZM — 260 aa
DHFR        = "MISLIAALAVDRVIGMENAMPWNLPADLAWFKRNTLDKPVIMGRHTWESIGRPLPGRKNIILSSQPGTDDRVTWVKSVDEAIAACGDVPEIMVIGGGRVYEQFLPKAQKLYLTHIDAEVEGDTHFPDYEPDDWESVFSEFHDADAQNSHSYCFKILERR"                                                                                          # 4DFR — 159 aa
STREPTAVIDIN = "DPSKDSKAQVSAAEAGITGTWYNQLGSTFIVTAGADGALTGTYESAVGNAESRYVLTGRYDSAPATDGSGTALGWTVAWKNNYRNAHSATTWSGQYVGGAEARINTQWLLTSGTTEANAWKSTLVGHDTFTKVKPSAASIDAAKKAGVNNGNPLDAVQQ"                                                                                         # 1STP — 158 aa
TRYPSIN     = "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQFISASKSIVHPSYNSNTLNNDIMLIKLKSAASLNSRVASISLPTSCASAGTQCLISGWGNTKSSGTSYPDVLKCLKAPILSDSSCKSAYPGQITSNMFCAGYLEGGKDSCQGDSGGPVVCSGKLQGIVSWGSGCAQKNKPGVYTKVCNYVSWIKQTIASN"                              # 3PTB / 2PTC — 223 aa
BPTI        = "RPDFCLEPPYTGPCKARIIRYFYNAKAGLCQTFVYGGCRAKRNNFKSAEDCMRTCGGA"                                                                                                                                                                                  # 2PTC — 58 aa

# Multi-chain entries: distinct interaction partners (barnase·barstar,
# MDM2·p53, calmodulin·MLCK, trypsin·BPTI) plus the chains that make up
# the oligomeric proteins insulin and hemoglobin α₂β₂.
INSULIN_A   = "GIVEQCCTSICSLYQLENYCN"                                                                                                                                                                                                                          # 4INS chain A — 21 aa
INSULIN_B   = "FVNQHLCGSHLVEALYLVCGERGFFYTPKA"                                                                                                                                                                                                                # 4INS chain B — 30 aa
BARNASE     = "AQVINTFDGVADYLQTYHKLPDNYITKSEAQALGWVASKGNLADVAPGKSIGGDIFSNREGKLPGKSGRTWREADINYTSGFRNSDRILYSSDWLIYKTTDHYQTFTKIR"                                                                                                                                       # 1BRS — 110 aa
BARSTAR     = "KKAVINGEQIRSISDLHQTLKKELALPEYYGENLDALWDALTGWVEYPLVLEWRQFEQSKQLTENGAESVLQVFREAKAEGADITIILS"                                                                                                                                                          # 1BRS — 89 aa
MDM2_N      = "SQIPASEQETLVRPKPLLLKLLKSVGAQKDTYTMKEVLFYLGQYIMTKRLYDEKQQHIVYCSNDLLGDLFGVPSFSVKEHRKIYTMIYRNLVVVNQQESSDSGTSVSEN"                                                                                                                                          # 1YCR — 109 aa
P53_PEPTIDE = "SQETFSDLWKLLPEN"                                                                                                                                                                                                                                # 1YCR — 15 aa
CALMODULIN  = "ADQLTEEQIAEFKEAFSLFDKDGDGTITTKELGTVMRSLGQNPTEAELQDMINEVDADGNGTIDFPEFLTMMARKMKDTDSEEEIREAFRVFDKDGNGFISAAELRHVMTNLGEKLTDEEVDEMIREADIDGDGQVNYEEFVTMMTSK"                                                                                                  # 2BBM — 148 aa (Drosophila)
MLCK_PEPTIDE = "KRRWKKNFIAVSAANRFKKISSSGAL"                                                                                                                                                                                                                    # 2BBM — 26 aa
HBA         = "VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR"                                                                                                              # 1A3N α — 141 aa
HBB         = "VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH"                                                                                                          # 1A3N β — 146 aa


# ── The rotation ────────────────────────────────────────────────────────────
# Ordered so the first four panels (W0..W3) open with a visually diverse
# tableau: a drug-target complex, a cofactor binder, a small multi-chain,
# and the full hemoglobin tetramer. The rest interleave monomers with
# protein-protein and protein-ligand complexes for variety across cycles.

ROTATION: list[Complex] = [
    # ── Opening tableau (first round, deterministic 1:1 with cards) ─────
    _build("HIV protease · indinavir",  _Chain(HIV_PROTEASE, copies=2), _Ligand("MK1"),      pdb="1HSG"),
    _build("myoglobin · heme",          _Chain(MYOGLOBIN),               _Ligand("HEM"),     pdb="1MBN"),
    # Insulin and hemoglobin are single oligomeric proteins (not
    # protein–protein interactions between two distinct proteins), so we
    # override the default classification.
    _build("insulin",                   _Chain(INSULIN_A),               _Chain(INSULIN_B),     pdb="4INS", kind="protein"),
    _build("hemoglobin α₂β₂",          _Chain(HBA, copies=2),           _Chain(HBB, copies=2), pdb="1A3N", kind="protein"),

    # ── Remaining 12 — interaction-heavy mix ────────────────────────────
    _build("ubiquitin",                 _Chain(UBIQUITIN),                                   pdb="1UBQ"),
    _build("trypsin · benzamidine",     _Chain(TRYPSIN),                 _Ligand("BEN"),     pdb="3PTB"),
    _build("barnase · barstar",         _Chain(BARNASE),                 _Chain(BARSTAR),    pdb="1BRS"),
    _build("lysozyme",                  _Chain(LYSOZYME),                                    pdb="1HEL"),
    _build("CA-I · acetazolamide",      _Chain(CA_I),                    _Ligand("AZM"),     pdb="1AZM"),
    _build("MDM2 · p53 peptide",        _Chain(MDM2_N),                  _Chain(P53_PEPTIDE), pdb="1YCR"),
    _build("cytochrome c",              _Chain(CYT_C),                                       pdb="1HRC"),
    _build("DHFR · methotrexate",       _Chain(DHFR),                    _Ligand("MTX"),     pdb="4DFR"),
    _build("calmodulin · MLCK peptide", _Chain(CALMODULIN),              _Chain(MLCK_PEPTIDE), pdb="2BBM"),
    _build("crambin",                   _Chain(CRAMBIN),                                     pdb="1CRN"),
    _build("streptavidin · biotin",     _Chain(STREPTAVIDIN),            _Ligand("BTN"),     pdb="1STP"),
    _build("trypsin · BPTI",            _Chain(TRYPSIN),                 _Chain(BPTI),       pdb="2PTC"),
]


assert len(ROTATION) == 16, f"expected 16 complexes, got {len(ROTATION)}"
_violations = [c for c in ROTATION if c.seq_len > MAX_SEQ_LEN]
assert not _violations, (
    f"seq_len > {MAX_SEQ_LEN} in: "
    + ", ".join(f"{c.name}({c.seq_len})" for c in _violations)
)
