"""Input file parsers for Boltz-2: a3m, csv, fasta, mmcif, pdb, schema, yaml."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Optional, TextIO
import numpy as np
from tt_boltz.data import const
from tt_boltz.data.types import MSA, MSADeletion, MSAResidue, MSASequence
import pandas as pd
from collections.abc import Mapping
from Bio import SeqIO
from rdkit.Chem.rdchem import Mol
from tt_boltz.data.types import Target
import yaml
import contextlib
from collections import defaultdict
from dataclasses import dataclass, replace
import gemmi
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, HybridizationType
from sklearn.neighbors import KDTree
from tt_boltz.data.mol import load_molecules
from tt_boltz.data.types import (
    AtomV2,
    BondV2,
    Chain,
    Coords,
    Ensemble,
    Interface,
    Residue,
    StructureInfo,
    StructureV2,
)
from tempfile import NamedTemporaryFile
import click
from Bio import Align
from chembl_structure_pipeline.exclude_flag import exclude_flag
from chembl_structure_pipeline.standardizer import standardize_mol
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.rdchem import BondStereo, Conformer
from rdkit.Chem.rdDistGeom import GetMoleculeBoundsMatrix
from rdkit.Chem.rdMolDescriptors import CalcNumHeavyAtoms
from scipy.optimize import linear_sum_assignment
from tt_boltz.data.types import (
    AffinityInfo,
    Atom,
    AtomV2,
    Bond,
    BondV2,
    Chain,
    ChainInfo,
    ChiralAtomConstraint,
    Connection,
    Coords,
    Ensemble,
    InferenceOptions,
    Interface,
    PlanarBondConstraint,
    PlanarRing5Constraint,
    PlanarRing6Constraint,
    RDKitBoundsConstraint,
    Record,
    Residue,
    ResidueConstraints,
    StereoBondConstraint,
    Structure,
    StructureInfo,
    StructureV2,
    Target,
    TemplateInfo,
)


# ---- a3m.py ----





def _parse_a3m(  # noqa: C901
    lines: TextIO,
    taxonomy: Optional[dict[str, str]],
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an MSA file.

    Parameters
    ----------
    lines : TextIO
        The lines of the MSA file.
    taxonomy : dict[str, str]
        The taxonomy database, if available.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line in lines:
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line or line.startswith("#"):
            continue

        # Get taxonomy, if annotated
        if line.startswith(">"):
            header = line.split()[0]
            if taxonomy and header.startswith(">UniRef100"):
                uniref_id = header.split("_")[1]
                taxonomy_id = taxonomy.get(uniref_id)
                if taxonomy_id is None:
                    taxonomy_id = -1
            else:
                taxonomy_id = -1
            continue

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            token = const.token_ids[token]
            residue.append(token)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa


def parse_a3m(
    path: Path,
    taxonomy: Optional[dict[str, str]],
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    taxonomy : Redis
        The taxonomy database.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read the file
    if path.suffix == ".gz":
        with gzip.open(str(path), "rt") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)
    else:
        with path.open("r") as f:
            msa = _parse_a3m(f, taxonomy, max_seqs)

    return msa



# ---- csv.py ----





def parse_csv(
    path: Path,
    max_seqs: Optional[int] = None,
) -> MSA:
    """Process an A3M file.

    Parameters
    ----------
    path : Path
        The path to the a3m(.gz) file.
    max_seqs : int, optional
        The maximum number of sequences.

    Returns
    -------
    MSA
        The MSA object.

    """
    # Read file
    data = pd.read_csv(path)

    # Check columns
    if tuple(sorted(data.columns)) != ("key", "sequence"):
        msg = "Invalid CSV format, expected columns: ['sequence', 'key']"
        raise ValueError(msg)

    # Create taxonomy mapping
    visited = set()
    sequences = []
    deletions = []
    residues = []

    seq_idx = 0
    for line, key in zip(data["sequence"], data["key"]):
        line: str
        line = line.strip()  # noqa: PLW2901
        if not line:
            continue

        # Get taxonomy, if annotated
        taxonomy_id = -1
        if (str(key) != "nan") and (key is not None) and (key != ""):
            taxonomy_id = key

        # Skip if duplicate sequence
        str_seq = line.replace("-", "").upper()
        if str_seq not in visited:
            visited.add(str_seq)
        else:
            continue

        # Process sequence
        residue = []
        deletion = []
        count = 0
        res_idx = 0
        for c in line:
            if c != "-" and c.islower():
                count += 1
                continue
            token = const.prot_letter_to_token[c]
            token = const.token_ids[token]
            residue.append(token)
            if count > 0:
                deletion.append((res_idx, count))
                count = 0
            res_idx += 1

        res_start = len(residues)
        res_end = res_start + len(residue)

        del_start = len(deletions)
        del_end = del_start + len(deletion)

        sequences.append((seq_idx, taxonomy_id, res_start, res_end, del_start, del_end))
        residues.extend(residue)
        deletions.extend(deletion)

        seq_idx += 1
        if (max_seqs is not None) and (seq_idx >= max_seqs):
            break

    # Create MSA object
    msa = MSA(
        residues=np.array(residues, dtype=MSAResidue),
        deletions=np.array(deletions, dtype=MSADeletion),
        sequences=np.array(sequences, dtype=MSASequence),
    )
    return msa



# ---- fasta.py ----





def parse_fasta(  # noqa: C901, PLR0912
    path: Path,
    ccd: Mapping[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Target:
    """Parse a fasta file.

    The name of the fasta file is used as the name of this job.
    We rely on the fasta record id to determine the entity type.

    > CHAIN_ID|ENTITY_TYPE|MSA_ID
    SEQUENCE
    > CHAIN_ID|ENTITY_TYPE|MSA_ID
    ...

    Where ENTITY_TYPE is either protein, rna, dna, ccd or smiles,
    and CHAIN_ID is the chain identifier, which should be unique.
    The MSA_ID is optional and should only be used on proteins.

    Parameters
    ----------
    fasta_file : Path
        Path to the fasta file.
    ccd : Dict
        Dictionary of CCD components.
    mol_dir : Path
        Path to the directory containing the molecules.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    # Read fasta file
    with path.open("r") as f:
        records = list(SeqIO.parse(f, "fasta"))

    # Make sure all records have a chain id and entity
    for seq_record in records:
        if "|" not in seq_record.id:
            msg = f"Invalid record id: {seq_record.id}"
            raise ValueError(msg)

        header = seq_record.id.split("|")
        assert len(header) >= 2, f"Invalid record id: {seq_record.id}"

        chain_id, entity_type = header[:2]
        if entity_type.lower() not in {"protein", "dna", "rna", "ccd", "smiles"}:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)
        if chain_id == "":
            msg = "Empty chain id in input fasta!"
            raise ValueError(msg)
        if entity_type == "":
            msg = "Empty entity type in input fasta!"
            raise ValueError(msg)

    # Convert to yaml format
    sequences = []
    for seq_record in records:
        # Get chain id, entity type and sequence
        header = seq_record.id.split("|")
        chain_id, entity_type = header[:2]
        if len(header) == 3 and header[2] != "":
            assert entity_type.lower() == "protein", (
                "MSA_ID is only allowed for proteins"
            )
            msa_id = header[2]
        else:
            msa_id = None

        entity_type = entity_type.upper()
        seq = str(seq_record.seq)

        if entity_type == "PROTEIN":
            molecule = {
                "protein": {
                    "id": chain_id,
                    "sequence": seq,
                    "modifications": [],
                    "msa": msa_id,
                },
            }
        elif entity_type == "RNA":
            molecule = {
                "rna": {
                    "id": chain_id,
                    "sequence": seq,
                    "modifications": [],
                },
            }
        elif entity_type == "DNA":
            molecule = {
                "dna": {
                    "id": chain_id,
                    "sequence": seq,
                    "modifications": [],
                }
            }
        elif entity_type.upper() == "CCD":
            molecule = {
                "ligand": {
                    "id": chain_id,
                    "ccd": seq,
                }
            }
        elif entity_type.upper() == "SMILES":
            molecule = {
                "ligand": {
                    "id": chain_id,
                    "smiles": seq,
                }
            }

        sequences.append(molecule)

    data = {
        "sequences": sequences,
        "bonds": [],
        "version": 1,
    }

    name = path.stem
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)



# ---- yaml.py ----





def parse_yaml(
    path: Path,
    ccd: dict[str, Mol],
    mol_dir: Path,
    boltz2: bool = False,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a yaml file with the following format:

    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
        - ligand:
            id: [F, G]
            ccd: []
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
    templates:
        - path: /path/to/template.pdb
          ids: [A] # optional, specify which chains to template

    version: 1

    Parameters
    ----------
    path : Path
        Path to the YAML input format.
    components : Dict
        Dictionary of CCD components.
    boltz2 : bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    with path.open("r") as file:
        data = yaml.safe_load(file)

    name = path.stem
    return parse_boltz_schema(name, data, ccd, mol_dir, boltz2)



# ---- mmcif.py ----




####################################################################################################
# DATACLASSES
####################################################################################################


@dataclass(frozen=True, slots=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    coords: tuple[float, float, float]
    is_present: bool
    bfactor: float
    plddt: Optional[float] = None


@dataclass(frozen=True, slots=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True, slots=True)
class ParsedResidue:
    """A parsed residue object."""

    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool


@dataclass(frozen=True, slots=True)
class ParsedChain:
    """A parsed chain object."""

    name: str
    entity: str
    type: int
    residues: list[ParsedResidue]
    sequence: Optional[str] = None


@dataclass(frozen=True, slots=True)
class ParsedConnection:
    """A parsed connection object."""

    chain_1: str
    chain_2: str
    residue_index_1: int
    residue_index_2: int
    atom_index_1: str
    atom_index_2: str


@dataclass(frozen=True, slots=True)
class ParsedStructure:
    """A parsed structure object."""

    data: StructureV2
    info: StructureInfo
    sequences: dict[str, str]


####################################################################################################
# HELPERS
####################################################################################################


def get_mol(ccd: str, mols: dict, moldir: str) -> Mol:
    """Get mol from CCD code.

    Return mol with ccd from mols if it is in mols. Otherwise load it from moldir,
    add it to mols, and return the mol.
    """
    mol = mols.get(ccd)
    if mol is None:
        # Load molecule
        mol = load_molecules(moldir, [ccd])[ccd]

        # Add to resource
        if isinstance(mols, dict):
            mols[ccd] = mol
        else:
            mols.set(ccd, mol)

    return mol


def get_dates(block: gemmi.cif.Block) -> tuple[str, str, str]:
    """Get the deposited, released, and last revision dates.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    str
        The deposited date.
    str
        The released date.
    str
        The last revision date.

    """
    deposited = "_pdbx_database_status.recvd_initial_deposition_date"
    revision = "_pdbx_audit_revision_history.revision_date"
    deposit_date = revision_date = release_date = ""
    with contextlib.suppress(Exception):
        deposit_date = block.find([deposited])[0][0]
        release_date = block.find([revision])[0][0]
        revision_date = block.find([revision])[-1][0]

    return deposit_date, release_date, revision_date


def get_resolution(block: gemmi.cif.Block) -> float:
    """Get the resolution from a gemmi structure.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    float
        The resolution.

    """
    resolution = 0.0
    for res_key in (
        "_refine.ls_d_res_high",
        "_em_3d_reconstruction.resolution",
        "_reflns.d_resolution_high",
    ):
        with contextlib.suppress(Exception):
            resolution = float(block.find([res_key])[0].str(0))
            break
    return resolution


def get_method(block: gemmi.cif.Block) -> str:
    """Get the method from a gemmi structure.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    str
        The method.

    """
    method = ""
    method_key = "_exptl.method"
    with contextlib.suppress(Exception):
        methods = block.find([method_key])
        method = ",".join([m.str(0).lower() for m in methods])

    return method


def get_experiment_conditions(
    block: gemmi.cif.Block,
) -> tuple[Optional[float], Optional[float]]:
    """Get temperature and pH.

    Parameters
    ----------
    block : gemmi.cif.Block
        The block to process.

    Returns
    -------
    tuple[float, float]
        The temperature and pH.
    """
    temperature = None
    ph = None

    keys_t = [
        "_exptl_crystal_grow.temp",
        "_pdbx_nmr_exptl_sample_conditions.temperature",
    ]
    for key in keys_t:
        with contextlib.suppress(Exception):
            temperature = float(block.find([key])[0][0])
            break

    keys_ph = ["_exptl_crystal_grow.pH", "_pdbx_nmr_exptl_sample_conditions.pH"]
    with contextlib.suppress(Exception):
        for key in keys_ph:
            ph = float(block.find([key])[0][0])
            break

    return temperature, ph


def get_unk_token(dtype: gemmi.PolymerType) -> str:
    """Get the unknown token for a given entity type.

    Parameters
    ----------
    dtype : gemmi.EntityType
        The entity type.

    Returns
    -------
    str
        The unknown token.

    """
    if dtype == gemmi.PolymerType.PeptideL:
        unk = const.unk_token["PROTEIN"]
    elif dtype == gemmi.PolymerType.Dna:
        unk = const.unk_token["DNA"]
    elif dtype == gemmi.PolymerType.Rna:
        unk = const.unk_token["RNA"]
    else:
        msg = f"Unknown polymer type: {dtype}"
        raise ValueError(msg)

    return unk


def compute_covalent_ligands(
    connections: list[gemmi.Connection],
    subchain_map: dict[tuple[str, int], str],
    entities: dict[str, gemmi.Entity],
) -> set[str]:
    """Compute the covalent ligands from a list of connections.

    Parameters
    ----------
    connections: list[gemmi.Connection]
        The connections to process.
    subchain_map: dict[tuple[str, int], str]
        The mapping from chain, residue index to subchain name.
    entities: dict[str, gemmi.Entity]
        The entities in the structure.

    Returns
    -------
    set
        The covalent ligand subchains.

    """
    # Get covalent chain ids
    covalent_chain_ids = set()
    for connection in connections:
        if connection.type.name != "Covale":
            continue

        # Map to correct subchain
        chain_1_name = connection.partner1.chain_name
        chain_2_name = connection.partner2.chain_name

        res_1_id = connection.partner1.res_id.seqid
        res_1_id = str(res_1_id.num) + str(res_1_id.icode).strip()

        res_2_id = connection.partner2.res_id.seqid
        res_2_id = str(res_2_id.num) + str(res_2_id.icode).strip()

        subchain_1 = subchain_map[(chain_1_name, res_1_id)]
        subchain_2 = subchain_map[(chain_2_name, res_2_id)]

        # If non-polymer or branched, add to set
        entity_1 = entities[subchain_1].entity_type.name
        entity_2 = entities[subchain_2].entity_type.name

        if entity_1 in {"NonPolymer", "Branched"}:
            covalent_chain_ids.add(subchain_1)
        if entity_2 in {"NonPolymer", "Branched"}:
            covalent_chain_ids.add(subchain_2)

    return covalent_chain_ids


def compute_interfaces(atom_data: np.ndarray, chain_data: np.ndarray) -> np.ndarray:
    """Compute the chain-chain interfaces from a gemmi structure.

    Parameters
    ----------
    atom_data : list[tuple]
        The atom data.
    chain_data : list[tuple]
        The chain data.

    Returns
    -------
    list[tuple[int, int]]
        The interfaces.

    """
    # Compute chain_id per atom
    chain_ids = []
    for idx, chain in enumerate(chain_data):
        chain_ids.extend([idx] * chain["atom_num"])
    chain_ids = np.array(chain_ids)

    # Filter to present atoms
    coords = atom_data["coords"]
    mask = atom_data["is_present"]

    coords = coords[mask]
    chain_ids = chain_ids[mask]

    # Compute the distance matrix
    tree = KDTree(coords, metric="euclidean")
    query = tree.query_radius(coords, const.atom_interface_cutoff)

    # Get unique chain pairs
    interfaces = set()
    for c1, pairs in zip(chain_ids, query):
        chains = np.unique(chain_ids[pairs])
        chains = chains[chains != c1]
        interfaces.update((c1, c2) for c2 in chains)

    # Get unique chain pairs
    interfaces = [(min(i, j), max(i, j)) for i, j in interfaces]
    interfaces = list({(int(i), int(j)) for i, j in interfaces})
    interfaces = np.array(interfaces, dtype=Interface)
    return interfaces


####################################################################################################
# PARSING
####################################################################################################


def parse_ccd_residue(  # noqa: PLR0915, C901
    name: str,
    ref_mol: Mol,
    res_idx: int,
    gemmi_mol: Optional[gemmi.Residue] = None,
    is_covalent: bool = False,
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    components : dict
        The preprocessed PDB components dictionary.
    res_idx : int
        The residue index.
    gemmi_mol : Optional[gemmi.Residue]
        The PDB molecule, as a gemmi Residue object, if any.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    # Check if we have a PDB structure for this residue,
    # it could be a missing residue from the sequence
    is_present = gemmi_mol is not None

    # Save original index (required for parsing connections)
    if is_present:
        orig_idx = gemmi_mol.seqid
        orig_idx = str(orig_idx.num) + str(orig_idx.icode).strip()
    else:
        orig_idx = None

    # Remove hydrogens
    ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

    # Check if this is a single atom CCD residue
    if ref_mol.GetNumAtoms() == 1:
        pos = (0, 0, 0)
        bfactor = 0
        if is_present:
            pos = (
                gemmi_mol[0].pos.x,
                gemmi_mol[0].pos.y,
                gemmi_mol[0].pos.z,
            )
            bfactor = gemmi_mol[0].b_iso
        ref_atom = ref_mol.GetAtoms()[0]
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            coords=pos,
            is_present=is_present,
            bfactor=bfactor,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            orig_idx=orig_idx,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=is_present,
        )
        return residue

    # If multi-atom, start by getting the PDB coordinates
    pdb_pos = {}
    bfactor = {}
    if is_present:
        # Match atoms based on names
        for atom in gemmi_mol:
            atom: gemmi.Atom
            pos = (atom.pos.x, atom.pos.y, atom.pos.z)
            pdb_pos[atom.name] = pos
            bfactor[atom.name] = atom.b_iso
    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")

        # If the atom is a leaving atom, skip if not in the PDB and is_covalent
        if (
            atom.HasProp("leaving_atom")
            and int(atom.GetProp("leaving_atom")) == 1
            and is_covalent
            and (atom_name not in pdb_pos)
        ):
            continue

        # Get PDB coordinates, if any
        coords = pdb_pos.get(atom_name)
        if coords is None:
            atom_is_present = False
            coords = (0, 0, 0)
        else:
            atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                coords=coords,
                is_present=atom_is_present,
                bfactor=bfactor.get(atom_name, 0),
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1

    # Load bonds
    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        # Skip bonds with atoms ignored
        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        idx_1 = idx_map[idx_1]
        idx_2 = idx_map[idx_2]
        start = min(idx_1, idx_2)
        end = max(idx_1, idx_2)
        bond_type = bond.GetBondType().name
        bond_type = const.bond_type_ids.get(bond_type, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        orig_idx=orig_idx,
        is_standard=False,
        is_present=is_present,
    )


def parse_polymer(  # noqa: C901, PLR0915, PLR0912
    polymer: gemmi.ResidueSpan,
    polymer_type: gemmi.PolymerType,
    sequence: list[str],
    chain_id: str,
    entity: str,
    mols: dict[str, Mol],
    moldir: str,
) -> Optional[ParsedChain]:
    """Process a gemmi Polymer into a chain object.

    Performs alignment of the full sequence to the polymer
    residues. Loads coordinates and masks for the atoms in
    the polymer, following the ordering in const.atom_order.

    Parameters
    ----------
    polymer : gemmi.ResidueSpan
        The polymer to process.
    polymer_type : gemmi.PolymerType
        The polymer type.
    sequence : str
        The full sequence of the polymer.
    chain_id : str
        The chain identifier.
    entity : str
        The entity name.
    components : dict[str, Mol]
        The preprocessed PDB components dictionary.

    Returns
    -------
    ParsedChain, optional
        The output chain, if successful.

    Raises
    ------
    ValueError
        If the alignment fails.

    """
    # Ignore microheterogeneities (pick first)
    sequence = [gemmi.Entity.first_mon(item) for item in sequence]

    # Align full sequence to polymer residues
    # This is a simple way to handle all the different numbering schemes
    result = gemmi.align_sequence_to_polymer(
        sequence,
        polymer,
        polymer_type,
        gemmi.AlignmentScoring(),
    )

    # Get coordinates and masks
    i = 0
    ref_res = set(const.tokens)
    parsed = []
    for j, match in enumerate(result.match_string):
        # Get residue name from sequence
        res_name = sequence[j]

        # Check if we have a match in the structure
        res = None
        name_to_atom = {}

        if match == "|":
            # Get pdb residue
            res = polymer[i]
            name_to_atom = {a.name.upper(): a for a in res}

            # Double check the match
            if res.name != res_name:
                msg = "Alignment mismatch!"
                raise ValueError(msg)

            # Increment polymer index
            i += 1

        # Map MSE to MET, put the selenium atom in the sulphur column
        if res_name == "MSE":
            res_name = "MET"
            if "SE" in name_to_atom:
                name_to_atom["SD"] = name_to_atom["SE"]

        # Handle non-standard residues
        elif res_name not in ref_res:
            modified_mol = get_mol(res_name, mols, moldir)
            if modified_mol is not None:
                residue = parse_ccd_residue(
                    name=res_name,
                    ref_mol=modified_mol,
                    res_idx=j,
                    gemmi_mol=res,
                    is_covalent=True,
                )
                parsed.append(residue)
                continue
            else:  # noqa: RET507
                res_name = "UNK"

        # Load regular residues
        ref_mol = get_mol(res_name, mols, moldir)
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

        # Only use reference atoms set in constants
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_name]]

        # Iterate, always in the same order
        atoms: list[ParsedAtom] = []

        for ref_atom in ref_atoms:
            # Get atom name
            atom_name = ref_atom.GetProp("name")

            # Get coordinates from PDB
            if atom_name in name_to_atom:
                atom: gemmi.Atom = name_to_atom[atom_name]
                atom_is_present = True
                coords = (atom.pos.x, atom.pos.y, atom.pos.z)
                bfactor = atom.b_iso
            else:
                atom_is_present = False
                coords = (0, 0, 0)
                bfactor = 0

            # Add atom to list
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    coords=coords,
                    is_present=atom_is_present,
                    bfactor=bfactor,
                )
            )

        # Fix naming errors in arginine residues where NH2 is
        # incorrectly assigned to be closer to CD than NH1
        if (res is not None) and (res_name == "ARG"):
            ref_atoms: list[str] = const.ref_atoms["ARG"]
            cd = atoms[ref_atoms.index("CD")]
            nh1 = atoms[ref_atoms.index("NH1")]
            nh2 = atoms[ref_atoms.index("NH2")]

            cd_coords = np.array(cd.coords)
            nh1_coords = np.array(nh1.coords)
            nh2_coords = np.array(nh2.coords)

            if all(atom.is_present for atom in (cd, nh1, nh2)) and (
                np.linalg.norm(nh1_coords - cd_coords)
                > np.linalg.norm(nh2_coords - cd_coords)
            ):
                atoms[ref_atoms.index("NH1")] = replace(nh1, coords=nh2.coords)
                atoms[ref_atoms.index("NH2")] = replace(nh2, coords=nh1.coords)

        # Add residue to parsed list
        if res is not None:
            orig_idx = res.seqid
            orig_idx = str(orig_idx.num) + str(orig_idx.icode).strip()
        else:
            orig_idx = None

        atom_center = const.res_to_center_atom_id[res_name]
        atom_disto = const.res_to_disto_atom_id[res_name]
        parsed.append(
            ParsedResidue(
                name=res_name,
                type=const.token_ids[res_name],
                atoms=atoms,
                bonds=[],
                idx=j,
                atom_center=atom_center,
                atom_disto=atom_disto,
                is_standard=True,
                is_present=res is not None,
                orig_idx=orig_idx,
            )
        )

    # Get polymer class
    if polymer_type == gemmi.PolymerType.PeptideL:
        chain_type = const.chain_type_ids["PROTEIN"]
    elif polymer_type == gemmi.PolymerType.Dna:
        chain_type = const.chain_type_ids["DNA"]
    elif polymer_type == gemmi.PolymerType.Rna:
        chain_type = const.chain_type_ids["RNA"]

    # Return polymer object
    return ParsedChain(
        name=chain_id,
        entity=entity,
        residues=parsed,
        type=chain_type,
        sequence=gemmi.one_letter_code(sequence),
    )


def parse_connection(
    connection: gemmi.Connection,
    chains: list[ParsedChain],
    subchain_map: dict[tuple[str, int], str],
) -> ParsedConnection:
    """Parse (covalent) connection from a gemmi Connection.

    Parameters
    ----------
    connections : gemmi.Connectionlist
        The connection list to parse.
    chains : list[Chain]
        The parsed chains.
    subchain_map : dict[tuple[str, int], str]
        The mapping from chain, residue index to subchain name.

    Returns
    -------
    list[Connection]
        The parsed connections.

    """
    # Map to correct subchains
    chain_1_name = connection.partner1.chain_name
    chain_2_name = connection.partner2.chain_name

    res_1_id = connection.partner1.res_id.seqid
    res_1_id = str(res_1_id.num) + str(res_1_id.icode).strip()

    res_2_id = connection.partner2.res_id.seqid
    res_2_id = str(res_2_id.num) + str(res_2_id.icode).strip()

    subchain_1 = subchain_map[(chain_1_name, res_1_id)]
    subchain_2 = subchain_map[(chain_2_name, res_2_id)]

    # Get chain indices
    chain_1 = next(chain for chain in chains if (chain.name == subchain_1))
    chain_2 = next(chain for chain in chains if (chain.name == subchain_2))

    # Get residue indices
    res_1_idx, res_1 = next(
        (idx, res)
        for idx, res in enumerate(chain_1.residues)
        if (res.orig_idx == res_1_id)
    )
    res_2_idx, res_2 = next(
        (idx, res)
        for idx, res in enumerate(chain_2.residues)
        if (res.orig_idx == res_2_id)
    )

    # Get atom indices
    atom_index_1 = next(
        idx
        for idx, atom in enumerate(res_1.atoms)
        if atom.name == connection.partner1.atom_name
    )
    atom_index_2 = next(
        idx
        for idx, atom in enumerate(res_2.atoms)
        if atom.name == connection.partner2.atom_name
    )

    conn = ParsedConnection(
        chain_1=subchain_1,
        chain_2=subchain_2,
        residue_index_1=res_1_idx,
        residue_index_2=res_2_idx,
        atom_index_1=atom_index_1,
        atom_index_2=atom_index_2,
    )

    return conn


def parse_mmcif(  # noqa: C901, PLR0915, PLR0912
    path: str,
    mols: Optional[dict[str, Mol]] = None,
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    compute_interfaces: bool = True,
) -> ParsedStructure:
    """Parse a structure in MMCIF format.

    Parameters
    ----------
    mmcif_file : PathLike
        Path to the MMCIF file.
    components: Mapping[str, Mol]
        The preprocessed PDB components dictionary.

    Returns
    -------
    ParsedStructure
        The parsed structure.

    """
    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # set mols
    mols = {} if mols is None else mols

    # Parse MMCIF input file
    block = gemmi.cif.read(str(path))[0]

    # Extract medatadata
    deposit_date, release_date, revision_date = get_dates(block)
    resolution = get_resolution(block)
    method = get_method(block)
    temperature, ph = get_experiment_conditions(block)

    # Load structure object
    structure = gemmi.make_structure_from_block(block)

    # Clean up the structure
    structure.merge_chain_parts()
    structure.remove_waters()
    structure.remove_hydrogens()
    structure.remove_alternative_conformations()
    structure.remove_empty_chains()

    # Expand assembly 1
    if use_assembly and structure.assemblies:
        how = gemmi.HowToNameCopiedChain.AddNumber
        assembly_name = structure.assemblies[0].name
        structure.transform_to_assembly(assembly_name, how=how)

    # Parse entities
    # Create mapping from subchain id to entity
    entities: dict[str, gemmi.Entity] = {}
    entity_ids: dict[str, int] = {}
    for entity_id, entity in enumerate(structure.entities):
        entity: gemmi.Entity
        if entity.entity_type.name == "Water":
            continue
        for subchain_id in entity.subchains:
            entities[subchain_id] = entity
            entity_ids[subchain_id] = entity_id

    # Create mapping from chain, residue to subchains
    # since a Connection uses the chains and not subchins
    subchain_map = {}
    for chain in structure[0]:
        for residue in chain:
            seq_id = residue.seqid
            seq_id = str(seq_id.num) + str(seq_id.icode).strip()
            subchain_map[(chain.name, seq_id)] = residue.subchain

    # Find covalent ligands
    covalent_chain_ids = compute_covalent_ligands(
        connections=structure.connections,
        subchain_map=subchain_map,
        entities=entities,
    )

    # Parse chains
    chains: list[ParsedChain] = []
    for raw_chain in structure[0].subchains():
        # Check chain type
        subchain_id = raw_chain.subchain_id()
        entity: gemmi.Entity = entities[subchain_id]
        entity_type = entity.entity_type.name

        # Parse a polymer
        if entity_type == "Polymer":
            # Skip PeptideD, DnaRnaHybrid, Pna, Other
            if entity.polymer_type.name not in {
                "PeptideL",
                "Dna",
                "Rna",
            }:
                continue

            # Add polymer if successful
            parsed_polymer = parse_polymer(
                polymer=raw_chain,
                polymer_type=entity.polymer_type,
                sequence=entity.full_sequence,
                chain_id=subchain_id,
                entity=entity.name,
                mols=mols,
                moldir=moldir,
            )
            if parsed_polymer is not None:
                chains.append(parsed_polymer)

        # Parse a non-polymer
        elif entity_type in {"NonPolymer", "Branched"}:
            # Skip UNL
            if any(lig.name == "UNL" for lig in raw_chain):
                continue

            residues = []
            for lig_idx, ligand in enumerate(raw_chain):
                # Check if ligand is covalent
                if entity_type == "Branched":
                    is_covalent = True
                else:
                    is_covalent = subchain_id in covalent_chain_ids

                ligand: gemmi.Residue
                ligand_mol = get_mol(ligand.name, mols, moldir)

                residue = parse_ccd_residue(
                    name=ligand.name,
                    ref_mol=ligand_mol,
                    res_idx=lig_idx,
                    gemmi_mol=ligand,
                    is_covalent=is_covalent,
                )
                residues.append(residue)

            if residues:
                chains.append(
                    ParsedChain(
                        name=subchain_id,
                        entity=entity.name,
                        residues=residues,
                        type=const.chain_type_ids["NONPOLYMER"],
                    )
                )

    # If no chains parsed fail
    if not chains:
        msg = "No chains parsed!"
        raise ValueError(msg)

    # Want to traverse subchains in same order as reference structure
    ref_chain_map = {ref_chain.name: i for i, ref_chain in enumerate(chains)}
    all_ensembles = [chains]

    # Loop through different structures in model
    for struct in list(structure)[1:]:
        struct: gemmi.Model
        ensemble_chains = {}

        for raw_chain in struct.subchains():
            # Check chain type
            subchain_id = raw_chain.subchain_id()
            entity: gemmi.Entity = entities[subchain_id]
            entity_type = entity.entity_type.name

            # Parse a polymer
            if entity_type == "Polymer":
                # Skip PeptideD, DnaRnaHybrid, Pna, Other
                if entity.polymer_type.name not in {
                    "PeptideL",
                    "Dna",
                    "Rna",
                }:
                    continue

                # Add polymer if successful
                parsed_polymer = parse_polymer(
                    polymer=raw_chain,
                    polymer_type=entity.polymer_type,
                    sequence=entity.full_sequence,
                    chain_id=subchain_id,
                    entity=entity.name,
                    mols=mols,
                    moldir=moldir,
                )
                if parsed_polymer is not None:
                    ensemble_chains[ref_chain_map[subchain_id]] = parsed_polymer

            # Parse a non-polymer
            elif entity_type in {"NonPolymer", "Branched"}:
                # Skip UNL
                if any(lig.name == "UNL" for lig in raw_chain):
                    continue

                residues = []
                for lig_idx, ligand in enumerate(raw_chain):
                    # Check if ligand is covalent
                    if entity_type == "Branched":
                        is_covalent = True
                    else:
                        is_covalent = subchain_id in covalent_chain_ids

                    ligand: gemmi.Residue
                    ligand_mol = get_mol(ligand.name, mols, moldir)

                    residue = parse_ccd_residue(
                        name=ligand.name,
                        ref_mol=ligand_mol,
                        res_idx=lig_idx,
                        gemmi_mol=ligand,
                        is_covalent=is_covalent,
                    )
                    residues.append(residue)

                if residues:
                    parsed_non_polymer = ParsedChain(
                        name=subchain_id,
                        entity=entity.name,
                        residues=residues,
                        type=const.chain_type_ids["NONPOLYMER"],
                    )
                    ensemble_chains[ref_chain_map[subchain_id]] = parsed_non_polymer

        # Ensure ensemble chains are in the same order as reference structure
        ensemble_chains = [ensemble_chains[idx] for idx in range(len(ensemble_chains))]
        all_ensembles.append(ensemble_chains)

    # Parse covalent connections
    connections: list[ParsedConnection] = []
    for connection in structure.connections:
        # Skip non-covalent connections
        connection: gemmi.Connection
        if connection.type.name != "Covale":
            continue
        try:
            parsed_connection = parse_connection(
                connection=connection,
                chains=chains,
                subchain_map=subchain_map,
            )
        except Exception:  # noqa: S112, BLE001
            continue
        connections.append(parsed_connection)

    # Create tables
    atom_data = []
    bond_data = []
    res_data = []
    chain_data = []
    ensemble_data = []
    coords_data = defaultdict(list)

    # Convert parsed chains to tables
    atom_idx = 0
    res_idx = 0
    sym_count = {}
    chain_to_idx = {}
    res_to_idx = {}
    chain_to_seq = {}

    for asym_id, chain in enumerate(chains):
        # Compute number of atoms and residues
        res_num = len(chain.residues)
        atom_num = sum(len(res.atoms) for res in chain.residues)

        # Get same chain across models in ensemble
        ensemble_chains = [ensemble[asym_id] for ensemble in all_ensembles]
        assert len(ensemble_chains) == len(all_ensembles)
        for ensemble_chain in ensemble_chains:
            assert len(ensemble_chain.residues) == res_num
            assert sum(len(res.atoms) for res in ensemble_chain.residues) == atom_num

        # Find all copies of this chain in the assembly
        entity_id = entity_ids[chain.name]
        sym_id = sym_count.get(entity_id, 0)
        chain_data.append(
            (
                chain.name,
                chain.type,
                entity_id,
                sym_id,
                asym_id,
                atom_idx,
                atom_num,
                res_idx,
                res_num,
                0,
            )
        )
        chain_to_idx[chain.name] = asym_id
        sym_count[entity_id] = sym_id + 1
        if chain.sequence is not None:
            chain_to_seq[chain.name] = chain.sequence

        # Add residue, atom, bond, data
        for i, res in enumerate(chain.residues):
            # Get same residue across models in ensemble
            ensemble_residues = [
                ensemble_chain.residues[i] for ensemble_chain in ensemble_chains
            ]
            assert len(ensemble_residues) == len(all_ensembles)
            for ensemble_res in ensemble_residues:
                assert ensemble_res.name == res.name

            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append(
                (
                    res.name,
                    res.type,
                    res.idx,
                    atom_idx,
                    len(res.atoms),
                    atom_center,
                    atom_disto,
                    res.is_standard,
                    res.is_present,
                )
            )
            res_to_idx[(chain.name, i)] = (res_idx, atom_idx)

            for bond in res.bonds:
                chain_1 = asym_id
                chain_2 = asym_id
                res_1 = res_idx
                res_2 = res_idx
                atom_1 = atom_idx + bond.atom_1
                atom_2 = atom_idx + bond.atom_2
                bond_data.append(
                    (
                        chain_1,
                        chain_2,
                        res_1,
                        res_2,
                        atom_1,
                        atom_2,
                        bond.type,
                    )
                )

            for a_idx, atom in enumerate(res.atoms):
                # Get same atom across models in ensemble
                ensemble_atoms = [
                    ensemble_res.atoms[a_idx] for ensemble_res in ensemble_residues
                ]
                assert len(ensemble_atoms) == len(all_ensembles)
                for e_idx, ensemble_atom in enumerate(ensemble_atoms):
                    assert ensemble_atom.name == atom.name
                    assert atom.is_present == ensemble_atom.is_present

                    coords_data[e_idx].append(ensemble_atom.coords)

                atom_data.append(
                    (
                        atom.name,
                        atom.coords,
                        atom.is_present,
                        atom.bfactor,
                        1.0,  # plddt is 1 for real data
                    )
                )
                atom_idx += 1

            res_idx += 1

    # Create coordinates table
    coords_data_ = []
    for e_idx in range(len(coords_data)):
        ensemble_data.append((e_idx * atom_idx, atom_idx))
        coords_data_.append(coords_data[e_idx])
    coords_data = [(x,) for xs in coords_data_ for x in xs]

    # Convert connections to tables
    for conn in connections:
        chain_1_idx = chain_to_idx[conn.chain_1]
        chain_2_idx = chain_to_idx[conn.chain_2]
        res_1_idx, atom_1_offset = res_to_idx[(conn.chain_1, conn.residue_index_1)]
        res_2_idx, atom_2_offset = res_to_idx[(conn.chain_2, conn.residue_index_2)]
        atom_1_idx = atom_1_offset + conn.atom_index_1
        atom_2_idx = atom_2_offset + conn.atom_index_2
        bond_data.append(
            (
                chain_1_idx,
                chain_2_idx,
                res_1_idx,
                res_2_idx,
                atom_1_idx,
                atom_2_idx,
                const.bond_type_ids["COVALENT"],
            )
        )

    # Convert into datatypes
    atoms = np.array(atom_data, dtype=AtomV2)
    bonds = np.array(bond_data, dtype=BondV2)
    residues = np.array(res_data, dtype=Residue)
    chains = np.array(chain_data, dtype=Chain)
    mask = np.ones(len(chain_data), dtype=bool)
    ensemble = np.array(ensemble_data, dtype=Ensemble)
    coords = np.array(coords_data, dtype=Coords)

    # Compute interface chains (find chains with a heavy atom within 5A)
    if compute_interfaces:
        interfaces = compute_interfaces(atoms, chains)
    else:
        interfaces = np.array([], dtype=Interface)

    # Return parsed structure
    info = StructureInfo(
        deposited=deposit_date,
        revised=revision_date,
        released=release_date,
        resolution=resolution,
        method=method,
        num_chains=len(chains),
        num_interfaces=len(interfaces),
        temperature=temperature,
        pH=ph,
    )

    data = StructureV2(
        atoms=atoms,
        bonds=bonds,
        residues=residues,
        chains=chains,
        interfaces=interfaces,
        mask=mask,
        ensemble=ensemble,
        coords=coords,
    )

    return ParsedStructure(
        data=data,
        info=info,
        sequences=chain_to_seq,
    )



# ---- pdb.py ----


def parse_pdb(
    path: str,
    mols: Optional[dict[str, Mol]] = None,
    moldir: Optional[str] = None,
    use_assembly: bool = True,
    compute_interfaces: bool = True,
) -> ParsedStructure:
    with NamedTemporaryFile(suffix=".cif") as tmp_cif_file:
        tmp_cif_path = tmp_cif_file.name
        structure = gemmi.read_structure(str(path))
        structure.setup_entities()

        subchain_counts, subchain_renaming = {}, {}
        for chain in structure[0]:
            subchain_counts[chain.name] = 0
            for res in chain:
                if res.subchain not in subchain_renaming:
                    subchain_renaming[res.subchain] = chain.name + str(subchain_counts[chain.name] + 1)
                    subchain_counts[chain.name] += 1
                res.subchain = subchain_renaming[res.subchain]
        for entity in structure.entities:
            entity.subchains = [subchain_renaming[subchain] for subchain in entity.subchains]

        doc = structure.make_mmcif_document()
        doc.write_file(tmp_cif_path)

        return parse_mmcif(
            path=tmp_cif_path,
            mols=mols,
            moldir=moldir,
            use_assembly=use_assembly,
            compute_interfaces=compute_interfaces
        )


# ---- schema.py ----





####################################################################################################
# DATACLASSES
####################################################################################################


@dataclass(frozen=True)
class ParsedAtom:
    """A parsed atom object."""

    name: str
    element: int
    charge: int
    coords: tuple[float, float, float]
    conformer: tuple[float, float, float]
    is_present: bool
    chirality: int


@dataclass(frozen=True)
class ParsedBond:
    """A parsed bond object."""

    atom_1: int
    atom_2: int
    type: int


@dataclass(frozen=True)
class ParsedRDKitBoundsConstraint:
    """A parsed RDKit bounds constraint object."""

    atom_idxs: tuple[int, int]
    is_bond: bool
    is_angle: bool
    upper_bound: float
    lower_bound: float


@dataclass(frozen=True)
class ParsedChiralAtomConstraint:
    """A parsed chiral atom constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_reference: bool
    is_r: bool


@dataclass(frozen=True)
class ParsedStereoBondConstraint:
    """A parsed stereo bond constraint object."""

    atom_idxs: tuple[int, int, int, int]
    is_check: bool
    is_e: bool


@dataclass(frozen=True)
class ParsedPlanarBondConstraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing5Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int]


@dataclass(frozen=True)
class ParsedPlanarRing6Constraint:
    """A parsed planar bond constraint object."""

    atom_idxs: tuple[int, int, int, int, int, int]


@dataclass(frozen=True)
class ParsedResidue:
    """A parsed residue object."""

    name: str
    type: int
    idx: int
    atoms: list[ParsedAtom]
    bonds: list[ParsedBond]
    orig_idx: Optional[int]
    atom_center: int
    atom_disto: int
    is_standard: bool
    is_present: bool
    rdkit_bounds_constraints: Optional[list[ParsedRDKitBoundsConstraint]] = None
    chiral_atom_constraints: Optional[list[ParsedChiralAtomConstraint]] = None
    stereo_bond_constraints: Optional[list[ParsedStereoBondConstraint]] = None
    planar_bond_constraints: Optional[list[ParsedPlanarBondConstraint]] = None
    planar_ring_5_constraints: Optional[list[ParsedPlanarRing5Constraint]] = None
    planar_ring_6_constraints: Optional[list[ParsedPlanarRing6Constraint]] = None


@dataclass(frozen=True)
class ParsedChain:
    """A parsed chain object."""

    entity: str
    type: int
    residues: list[ParsedResidue]
    cyclic_period: int
    sequence: Optional[str] = None
    affinity: Optional[bool] = False
    affinity_mw: Optional[float] = None


@dataclass(frozen=True)
class Alignment:
    """A parsed alignment object."""

    query_st: int
    query_en: int
    template_st: int
    template_en: int


####################################################################################################
# HELPERS
####################################################################################################


def convert_atom_name(name: str) -> tuple[int, int, int, int]:
    """Convert an atom name to a standard format.

    Parameters
    ----------
    name : str
        The atom name.

    Returns
    -------
    Tuple[int, int, int, int]
        The converted atom name.

    """
    name = name.strip()
    name = [ord(c) - 32 for c in name]
    name = name + [0] * (4 - len(name))
    return tuple(name)


def compute_3d_conformer(mol: Mol, version: str = "v3") -> bool:
    """Generate 3D coordinates using EKTDG method.

    Taken from `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The RDKit molecule to process
    version: str, optional
        The ETKDG version, defaults ot v3

    Returns
    -------
    bool
        Whether computation was successful.

    """
    if version == "v3":
        options = AllChem.ETKDGv3()
    elif version == "v2":
        options = AllChem.ETKDGv2()
    else:
        options = AllChem.ETKDGv2()

    options.clearConfs = False
    conf_id = -1

    try:
        conf_id = AllChem.EmbedMolecule(mol, options)

        if conf_id == -1:
            print(
                f"WARNING: RDKit ETKDGv3 failed to generate a conformer for molecule "
                f"{Chem.MolToSmiles(AllChem.RemoveHs(mol))}, so the program will start with random coordinates. "
                f"Note that the performance of the model under this behaviour was not tested."
            )
            options.useRandomCoords = True
            conf_id = AllChem.EmbedMolecule(mol, options)

        AllChem.UFFOptimizeMolecule(mol, confId=conf_id, maxIters=1000)

    except RuntimeError:
        pass  # Force field issue here
    except ValueError:
        pass  # sanitization issue here

    if conf_id != -1:
        conformer = mol.GetConformer(conf_id)
        conformer.SetProp("name", "Computed")
        conformer.SetProp("coord_generation", f"ETKDG{version}")

        return True

    return False


def get_conformer(mol: Mol) -> Conformer:
    """Retrieve an rdkit object for a deemed conformer.

    Inspired by `pdbeccdutils.core.component.Component`.

    Parameters
    ----------
    mol: Mol
        The molecule to process.

    Returns
    -------
    Conformer
        The desired conformer, if any.

    Raises
    ------
    ValueError
        If there are no conformers of the given tyoe.

    """
    # Try using the computed conformer
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Computed":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to the ideal coordinates
    for c in mol.GetConformers():
        try:
            if c.GetProp("name") == "Ideal":
                return c
        except KeyError:  # noqa: PERF203
            pass

    # Fallback to boltz2 format
    conf_ids = [int(conf.GetId()) for conf in mol.GetConformers()]
    if len(conf_ids) > 0:
        conf_id = conf_ids[0]
        conformer = mol.GetConformer(conf_id)
        return conformer

    msg = "Conformer does not exist."
    raise ValueError(msg)


def compute_geometry_constraints(mol: Mol, idx_map):
    if mol.GetNumAtoms() <= 1:
        return []

    # Ensure RingInfo is initialized
    mol.UpdatePropertyCache(strict=False)
    Chem.GetSymmSSSR(mol)  # Compute ring information

    bounds = GetMoleculeBoundsMatrix(
        mol,
        set15bounds=True,
        scaleVDW=True,
        doTriangleSmoothing=True,
        useMacrocycle14config=False,
    )
    bonds = set(
        tuple(sorted(b)) for b in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*"))
    )
    angles = set(
        tuple(sorted([a[0], a[2]]))
        for a in mol.GetSubstructMatches(Chem.MolFromSmarts("*~*~*"))
    )

    constraints = []
    for i, j in zip(*np.triu_indices(mol.GetNumAtoms(), k=1)):
        if i in idx_map and j in idx_map:
            constraint = ParsedRDKitBoundsConstraint(
                atom_idxs=(idx_map[i], idx_map[j]),
                is_bond=tuple(sorted([i, j])) in bonds,
                is_angle=tuple(sorted([i, j])) in angles,
                upper_bound=bounds[i, j],
                lower_bound=bounds[j, i],
            )
            constraints.append(constraint)
    return constraints


def compute_chiral_atom_constraints(mol, idx_map):
    constraints = []
    if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
        for center_idx, orientation in Chem.FindMolChiralCenters(
            mol, includeUnassigned=False
        ):
            center = mol.GetAtomWithIdx(center_idx)
            neighbors = [
                (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                for neighbor in center.GetNeighbors()
            ]
            neighbors = sorted(
                neighbors, key=lambda neighbor: neighbor[1], reverse=True
            )
            neighbors = tuple(neighbor[0] for neighbor in neighbors)
            is_r = orientation == "R"

            if len(neighbors) > 4 or center.GetHybridization() != HybridizationType.SP3:
                continue

            atom_idxs = (*neighbors[:3], center_idx)
            if all(i in idx_map for i in atom_idxs):
                constraints.append(
                    ParsedChiralAtomConstraint(
                        atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                        is_reference=True,
                        is_r=is_r,
                    )
                )

            if len(neighbors) == 4:
                for skip_idx in range(3):
                    chiral_set = neighbors[:skip_idx] + neighbors[skip_idx + 1 :]
                    if skip_idx % 2 == 0:
                        atom_idxs = chiral_set[::-1] + (center_idx,)
                    else:
                        atom_idxs = chiral_set + (center_idx,)
                    if all(i in idx_map for i in atom_idxs):
                        constraints.append(
                            ParsedChiralAtomConstraint(
                                atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                                is_reference=False,
                                is_r=is_r,
                            )
                        )
    return constraints


def compute_stereo_bond_constraints(mol, idx_map):
    constraints = []
    if all([atom.HasProp("_CIPRank") for atom in mol.GetAtoms()]):
        for bond in mol.GetBonds():
            stereo = bond.GetStereo()
            if stereo in {BondStereo.STEREOE, BondStereo.STEREOZ}:
                start_atom_idx, end_atom_idx = (
                    bond.GetBeginAtomIdx(),
                    bond.GetEndAtomIdx(),
                )
                start_neighbors = [
                    (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                    for neighbor in mol.GetAtomWithIdx(start_atom_idx).GetNeighbors()
                    if neighbor.GetIdx() != end_atom_idx
                ]
                start_neighbors = sorted(
                    start_neighbors, key=lambda neighbor: neighbor[1], reverse=True
                )
                start_neighbors = [neighbor[0] for neighbor in start_neighbors]
                end_neighbors = [
                    (neighbor.GetIdx(), int(neighbor.GetProp("_CIPRank")))
                    for neighbor in mol.GetAtomWithIdx(end_atom_idx).GetNeighbors()
                    if neighbor.GetIdx() != start_atom_idx
                ]
                end_neighbors = sorted(
                    end_neighbors, key=lambda neighbor: neighbor[1], reverse=True
                )
                end_neighbors = [neighbor[0] for neighbor in end_neighbors]
                is_e = stereo == BondStereo.STEREOE

                atom_idxs = (
                    start_neighbors[0],
                    start_atom_idx,
                    end_atom_idx,
                    end_neighbors[0],
                )
                if all(i in idx_map for i in atom_idxs):
                    constraints.append(
                        ParsedStereoBondConstraint(
                            atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                            is_check=True,
                            is_e=is_e,
                        )
                    )

                if len(start_neighbors) == 2 and len(end_neighbors) == 2:
                    atom_idxs = (
                        start_neighbors[1],
                        start_atom_idx,
                        end_atom_idx,
                        end_neighbors[1],
                    )
                    if all(i in idx_map for i in atom_idxs):
                        constraints.append(
                            ParsedStereoBondConstraint(
                                atom_idxs=tuple(idx_map[i] for i in atom_idxs),
                                is_check=False,
                                is_e=is_e,
                            )
                        )
    return constraints


def compute_flatness_constraints(mol, idx_map):
    planar_double_bond_smarts = Chem.MolFromSmarts("[C;X3;^2](*)(*)=[C;X3;^2](*)(*)")
    aromatic_ring_5_smarts = Chem.MolFromSmarts("[ar5^2]1[ar5^2][ar5^2][ar5^2][ar5^2]1")
    aromatic_ring_6_smarts = Chem.MolFromSmarts(
        "[ar6^2]1[ar6^2][ar6^2][ar6^2][ar6^2][ar6^2]1"
    )

    planar_double_bond_constraints = []
    aromatic_ring_5_constraints = []
    aromatic_ring_6_constraints = []
    for match in mol.GetSubstructMatches(planar_double_bond_smarts):
        if all(i in idx_map for i in match):
            planar_double_bond_constraints.append(
                ParsedPlanarBondConstraint(atom_idxs=tuple(idx_map[i] for i in match))
            )
    for match in mol.GetSubstructMatches(aromatic_ring_5_smarts):
        if all(i in idx_map for i in match):
            aromatic_ring_5_constraints.append(
                ParsedPlanarRing5Constraint(atom_idxs=tuple(idx_map[i] for i in match))
            )
    for match in mol.GetSubstructMatches(aromatic_ring_6_smarts):
        if all(i in idx_map for i in match):
            aromatic_ring_6_constraints.append(
                ParsedPlanarRing6Constraint(atom_idxs=tuple(idx_map[i] for i in match))
            )

    return (
        planar_double_bond_constraints,
        aromatic_ring_5_constraints,
        aromatic_ring_6_constraints,
    )


def get_global_alignment_score(query: str, template: str) -> float:
    """Align a sequence to a template.

    Parameters
    ----------
    query : str
        The query sequence.
    template : str
        The template sequence.

    Returns
    -------
    float
        The global alignment score.

    """
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "global"
    score = aligner.align(query, template)[0].score
    return score


def get_local_alignments(query: str, template: str) -> list[Alignment]:
    """Align a sequence to a template.

    Parameters
    ----------
    query : str
        The query sequence.
    template : str
        The template sequence.

    Returns
    -------
    Alignment
        The alignment between the query and template.

    """
    aligner = Align.PairwiseAligner(scoring="blastp")
    aligner.mode = "local"
    aligner.open_gap_score = -1000
    aligner.extend_gap_score = -1000

    alignments = []
    for result in aligner.align(query, template):
        coordinates = result.coordinates
        alignment = Alignment(
            query_st=int(coordinates[0][0]),
            query_en=int(coordinates[0][1]),
            template_st=int(coordinates[1][0]),
            template_en=int(coordinates[1][1]),
        )
        alignments.append(alignment)

    return alignments


def get_template_records_from_search(
    template_id: str,
    chain_ids: list[str],
    sequences: dict[str, str],
    template_chain_ids: list[str],
    template_sequences: dict[str, str],
    force: bool = False,
    threshold: Optional[float] = None,
) -> list[TemplateInfo]:
    """Get template records from an alignment."""
    # Compute pairwise scores
    score_matrix = []
    for chain_id in chain_ids:
        row = []
        for template_chain_id in template_chain_ids:
            chain_seq = sequences[chain_id]
            template_seq = template_sequences[template_chain_id]
            score = get_global_alignment_score(chain_seq, template_seq)
            row.append(score)
        score_matrix.append(row)

    # Find optimal mapping
    row_ind, col_ind = linear_sum_assignment(score_matrix, maximize=True)

    # Get alignment records
    template_records = []

    for row_idx, col_idx in zip(row_ind, col_ind):
        chain_id = chain_ids[row_idx]
        template_chain_id = template_chain_ids[col_idx]
        chain_seq = sequences[chain_id]
        template_seq = template_sequences[template_chain_id]
        alignments = get_local_alignments(chain_seq, template_seq)

        for alignment in alignments:
            template_record = TemplateInfo(
                name=template_id,
                query_chain=chain_id,
                query_st=alignment.query_st,
                query_en=alignment.query_en,
                template_chain=template_chain_id,
                template_st=alignment.template_st,
                template_en=alignment.template_en,
                force=force,
                threshold=threshold,
            )
            template_records.append(template_record)

    return template_records


def get_template_records_from_matching(
    template_id: str,
    chain_ids: list[str],
    sequences: dict[str, str],
    template_chain_ids: list[str],
    template_sequences: dict[str, str],
    force: bool = False,
    threshold: Optional[float] = None,
) -> list[TemplateInfo]:
    """Get template records from a given matching."""
    template_records = []

    for chain_id, template_chain_id in zip(chain_ids, template_chain_ids):
        # Align the sequences
        chain_seq = sequences[chain_id]
        template_seq = template_sequences[template_chain_id]
        alignments = get_local_alignments(chain_seq, template_seq)
        for alignment in alignments:
            template_record = TemplateInfo(
                name=template_id,
                query_chain=chain_id,
                query_st=alignment.query_st,
                query_en=alignment.query_en,
                template_chain=template_chain_id,
                template_st=alignment.template_st,
                template_en=alignment.template_en,
                force=force,
                threshold=threshold,
            )
            template_records.append(template_record)

    return template_records


def get_mol(ccd: str, mols: dict, moldir: str) -> Mol:
    """Get mol from CCD code.

    Return mol with ccd from mols if it is in mols. Otherwise load it from moldir,
    add it to mols, and return the mol.
    """
    mol = mols.get(ccd)
    if mol is None:
        mol = load_molecules(moldir, [ccd])[ccd]
    return mol


####################################################################################################
# PARSING
####################################################################################################


def parse_ccd_residue(
    name: str, ref_mol: Mol, res_idx: int, drop_leaving_atoms: bool = False
) -> Optional[ParsedResidue]:
    """Parse an MMCIF ligand.

    First tries to get the SMILES string from the RCSB.
    Then, tries to infer atom ordering using RDKit.

    Parameters
    ----------
    name: str
        The name of the molecule to parse.
    ref_mol: Mol
        The reference molecule to parse.
    res_idx : int
        The residue index.

    Returns
    -------
    ParsedResidue, optional
       The output ParsedResidue, if successful.

    """
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Check if this is a single heavy atom CCD residue
    if CalcNumHeavyAtoms(ref_mol) == 1:
        # Remove hydrogens
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)

        pos = (0, 0, 0)
        ref_atom = ref_mol.GetAtoms()[0]
        chirality_type = const.chirality_type_ids.get(
            str(ref_atom.GetChiralTag()), unk_chirality
        )
        atom = ParsedAtom(
            name=ref_atom.GetProp("name"),
            element=ref_atom.GetAtomicNum(),
            charge=ref_atom.GetFormalCharge(),
            coords=pos,
            conformer=(0, 0, 0),
            is_present=True,
            chirality=chirality_type,
        )
        unk_prot_id = const.unk_token_ids["PROTEIN"]
        residue = ParsedResidue(
            name=name,
            type=unk_prot_id,
            atoms=[atom],
            bonds=[],
            idx=res_idx,
            orig_idx=None,
            atom_center=0,  # Placeholder, no center
            atom_disto=0,  # Placeholder, no center
            is_standard=False,
            is_present=True,
        )
        return residue

    # Get reference conformer coordinates
    conformer = get_conformer(ref_mol)

    # Parse each atom in order of the reference mol
    atoms = []
    atom_idx = 0
    idx_map = {}  # Used for bonds later

    for i, atom in enumerate(ref_mol.GetAtoms()):
        # Ignore Hydrogen atoms
        if atom.GetAtomicNum() == 1:
            continue

        # Get atom name, charge, element and reference coordinates
        atom_name = atom.GetProp("name")

        # Drop leaving atoms for non-canonical amino acids.
        if drop_leaving_atoms and int(atom.GetProp("leaving_atom")):
            continue

        charge = atom.GetFormalCharge()
        element = atom.GetAtomicNum()
        ref_coords = conformer.GetAtomPosition(atom.GetIdx())
        ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)
        chirality_type = const.chirality_type_ids.get(
            str(atom.GetChiralTag()), unk_chirality
        )

        # Get PDB coordinates, if any
        coords = (0, 0, 0)
        atom_is_present = True

        # Add atom to list
        atoms.append(
            ParsedAtom(
                name=atom_name,
                element=element,
                charge=charge,
                coords=coords,
                conformer=ref_coords,
                is_present=atom_is_present,
                chirality=chirality_type,
            )
        )
        idx_map[i] = atom_idx
        atom_idx += 1

    # Load bonds
    bonds = []
    unk_bond = const.bond_type_ids[const.unk_bond_type]
    for bond in ref_mol.GetBonds():
        idx_1 = bond.GetBeginAtomIdx()
        idx_2 = bond.GetEndAtomIdx()

        # Skip bonds with atoms ignored
        if (idx_1 not in idx_map) or (idx_2 not in idx_map):
            continue

        idx_1 = idx_map[idx_1]
        idx_2 = idx_map[idx_2]
        start = min(idx_1, idx_2)
        end = max(idx_1, idx_2)
        bond_type = bond.GetBondType().name
        bond_type = const.bond_type_ids.get(bond_type, unk_bond)
        bonds.append(ParsedBond(start, end, bond_type))

    rdkit_bounds_constraints = compute_geometry_constraints(ref_mol, idx_map)
    chiral_atom_constraints = compute_chiral_atom_constraints(ref_mol, idx_map)
    stereo_bond_constraints = compute_stereo_bond_constraints(ref_mol, idx_map)
    planar_bond_constraints, planar_ring_5_constraints, planar_ring_6_constraints = (
        compute_flatness_constraints(ref_mol, idx_map)
    )

    unk_prot_id = const.unk_token_ids["PROTEIN"]
    return ParsedResidue(
        name=name,
        type=unk_prot_id,
        atoms=atoms,
        bonds=bonds,
        idx=res_idx,
        atom_center=0,
        atom_disto=0,
        orig_idx=None,
        is_standard=False,
        is_present=True,
        rdkit_bounds_constraints=rdkit_bounds_constraints,
        chiral_atom_constraints=chiral_atom_constraints,
        stereo_bond_constraints=stereo_bond_constraints,
        planar_bond_constraints=planar_bond_constraints,
        planar_ring_5_constraints=planar_ring_5_constraints,
        planar_ring_6_constraints=planar_ring_6_constraints,
    )


def parse_polymer(
    sequence: list[str],
    raw_sequence: str,
    entity: str,
    chain_type: str,
    components: dict[str, Mol],
    cyclic: bool,
    mol_dir: Path,
) -> Optional[ParsedChain]:
    """Process a sequence into a chain object.

    Performs alignment of the full sequence to the polymer
    residues. Loads coordinates and masks for the atoms in
    the polymer, following the ordering in const.atom_order.

    Parameters
    ----------
    sequence : list[str]
        The full sequence of the polymer.
    entity : str
        The entity id.
    entity_type : str
        The entity type.
    components : dict[str, Mol]
        The preprocessed PDB components dictionary.

    Returns
    -------
    ParsedChain, optional
        The output chain, if successful.

    Raises
    ------
    ValueError
        If the alignment fails.

    """
    ref_res = set(const.tokens)
    unk_chirality = const.chirality_type_ids[const.unk_chirality_type]

    # Get coordinates and masks
    parsed = []
    for res_idx, res_name in enumerate(sequence):
        # Check if modified residue
        # Map MSE to MET
        res_corrected = res_name if res_name != "MSE" else "MET"

        # Handle non-standard residues
        if res_corrected not in ref_res:
            ref_mol = get_mol(res_corrected, components, mol_dir)
            residue = parse_ccd_residue(
                name=res_corrected,
                ref_mol=ref_mol,
                res_idx=res_idx,
                drop_leaving_atoms=True,
            )
            parsed.append(residue)
            continue

        # Load ref residue
        ref_mol = get_mol(res_corrected, components, mol_dir)
        ref_mol = AllChem.RemoveHs(ref_mol, sanitize=False)
        ref_conformer = get_conformer(ref_mol)

        # Only use reference atoms set in constants
        ref_name_to_atom = {a.GetProp("name"): a for a in ref_mol.GetAtoms()}
        ref_atoms = [ref_name_to_atom[a] for a in const.ref_atoms[res_corrected]]

        # Iterate, always in the same order
        atoms: list[ParsedAtom] = []

        for ref_atom in ref_atoms:
            # Get atom name
            atom_name = ref_atom.GetProp("name")
            idx = ref_atom.GetIdx()

            # Get conformer coordinates
            ref_coords = ref_conformer.GetAtomPosition(idx)
            ref_coords = (ref_coords.x, ref_coords.y, ref_coords.z)

            # Set 0 coordinate
            atom_is_present = True
            coords = (0, 0, 0)

            # Add atom to list
            atoms.append(
                ParsedAtom(
                    name=atom_name,
                    element=ref_atom.GetAtomicNum(),
                    charge=ref_atom.GetFormalCharge(),
                    coords=coords,
                    conformer=ref_coords,
                    is_present=atom_is_present,
                    chirality=const.chirality_type_ids.get(
                        str(ref_atom.GetChiralTag()), unk_chirality
                    ),
                )
            )

        atom_center = const.res_to_center_atom_id[res_corrected]
        atom_disto = const.res_to_disto_atom_id[res_corrected]
        parsed.append(
            ParsedResidue(
                name=res_corrected,
                type=const.token_ids[res_corrected],
                atoms=atoms,
                bonds=[],
                idx=res_idx,
                atom_center=atom_center,
                atom_disto=atom_disto,
                is_standard=True,
                is_present=True,
                orig_idx=None,
            )
        )

    if cyclic:
        cyclic_period = len(sequence)
    else:
        cyclic_period = 0

    # Return polymer object
    return ParsedChain(
        entity=entity,
        residues=parsed,
        type=chain_type,
        cyclic_period=cyclic_period,
        sequence=raw_sequence,
    )


def token_spec_to_ids(
    chain_name, residue_index_or_atom_name, chain_to_idx, atom_idx_map, chains
):
    if chains[chain_name].type == const.chain_type_ids["NONPOLYMER"]:
        # Non-polymer chains are indexed by atom name
        _, _, atom_idx = atom_idx_map[(chain_name, 0, residue_index_or_atom_name)]
        return (chain_to_idx[chain_name], atom_idx)
    else:
        # Polymer chains are indexed by residue index
        return chain_to_idx[chain_name], residue_index_or_atom_name - 1


def parse_boltz_schema(  # noqa: C901, PLR0915, PLR0912
    name: str,
    schema: dict,
    ccd: Mapping[str, Mol],
    mol_dir: Optional[Path] = None,
    boltz_2: bool = False,
) -> Target:
    """Parse a Boltz input yaml / json.

    The input file should be a dictionary with the following format:

    version: 1
    sequences:
        - protein:
            id: A
            sequence: "MADQLTEEQIAEFKEAFSLF"
            msa: path/to/msa1.a3m
        - protein:
            id: [B, C]
            sequence: "AKLSILPWGHC"
            msa: path/to/msa2.a3m
        - rna:
            id: D
            sequence: "GCAUAGC"
        - ligand:
            id: E
            smiles: "CC1=CC=CC=C1"
    constraints:
        - bond:
            atom1: [A, 1, CA]
            atom2: [A, 2, N]
        - pocket:
            binder: E
            contacts: [[B, 1], [B, 2]]
            max_distance: 6
        - contact:
            token1: [A, 1]
            token2: [B, 1]
            max_distance: 6
    templates:
        - cif: path/to/template.cif
    properties:
        - affinity:
            binder: E

    Parameters
    ----------
    name : str
        A name for the input.
    schema : dict
        The input schema.
    components : dict
        Dictionary of CCD components.
    mol_dir: Path
        Path to the directory containing the molecules.
    boltz2: bool
        Whether to parse the input for Boltz2.

    Returns
    -------
    Target
        The parsed target.

    """
    # Assert version 1
    version = schema.get("version", 1)
    if version != 1:
        msg = f"Invalid version {version} in input!"
        raise ValueError(msg)

    # Disable rdkit warnings
    blocker = rdBase.BlockLogs()  # noqa: F841

    # First group items that have the same type, sequence and modifications
    items_to_group = {}
    chain_name_to_entity_type = {}

    for item in schema["sequences"]:
        # Get entity type
        entity_type = next(iter(item.keys())).lower()
        if entity_type not in {"protein", "dna", "rna", "ligand"}:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        # Get sequence
        if entity_type in {"protein", "dna", "rna"}:
            seq = str(item[entity_type]["sequence"])
        elif entity_type == "ligand":
            assert "smiles" in item[entity_type] or "ccd" in item[entity_type]
            assert "smiles" not in item[entity_type] or "ccd" not in item[entity_type]
            if "smiles" in item[entity_type]:
                seq = str(item[entity_type]["smiles"])
            else:
                seq = str(item[entity_type]["ccd"])

        # Group items by entity
        items_to_group.setdefault((entity_type, seq), []).append(item)

        # Map chain names to entity types
        chain_names = item[entity_type]["id"]
        chain_names = [chain_names] if isinstance(chain_names, str) else chain_names
        for chain_name in chain_names:
            chain_name_to_entity_type[chain_name] = entity_type

    # Check if any affinity ligand is present
    affinity_ligands = set()
    properties = schema.get("properties", [])
    if properties and not boltz_2:
        msg = "Affinity prediction is only supported for Boltz2!"
        raise ValueError(msg)

    for prop in properties:
        prop_type = next(iter(prop.keys())).lower()
        if prop_type == "affinity":
            binder = prop["affinity"]["binder"]
            if not isinstance(binder, str):
                # TODO: support multi residue ligands and ccd's
                msg = "Binder must be a single chain."
                raise ValueError(msg)

            if binder not in chain_name_to_entity_type:
                msg = f"Could not find binder with name {binder} in the input!"
                raise ValueError(msg)

            if chain_name_to_entity_type[binder] != "ligand":
                msg = (
                    f"Chain {binder} is not a ligand! "
                    "Affinity is currently only supported for ligands."
                )
                raise ValueError(msg)

            affinity_ligands.add(binder)

    # Check only one affinity ligand is present
    if len(affinity_ligands) > 1:
        msg = "Only one affinity ligand is currently supported!"
        raise ValueError(msg)

    # Go through entities and parse them
    extra_mols: dict[str, Mol] = {}
    chains: dict[str, ParsedChain] = {}
    chain_to_msa: dict[str, str] = {}
    entity_to_seq: dict[str, str] = {}
    is_msa_custom = False
    is_msa_auto = False
    ligand_id = 1
    for entity_id, items in enumerate(items_to_group.values()):
        # Get entity type and sequence
        entity_type = next(iter(items[0].keys())).lower()

        # Get ids
        ids = []
        for item in items:
            if isinstance(item[entity_type]["id"], str):
                ids.append(item[entity_type]["id"])
            elif isinstance(item[entity_type]["id"], list):
                ids.extend(item[entity_type]["id"])

        # Check if any affinity ligand is present
        if len(ids) == 1:
            affinity = ids[0] in affinity_ligands
        elif (len(ids) > 1) and any(x in affinity_ligands for x in ids):
            msg = "Cannot compute affinity for a ligand that has multiple copies!"
            raise ValueError(msg)
        else:
            affinity = False

        # Ensure all the items share the same msa
        msa = -1
        if entity_type == "protein":
            # Get the msa, default to 0, meaning auto-generated
            msa = items[0][entity_type].get("msa", 0)
            if (msa is None) or (msa == ""):
                msa = 0

            # Check if all MSAs are the same within the same entity
            for item in items:
                item_msa = item[entity_type].get("msa", 0)
                if (item_msa is None) or (item_msa == ""):
                    item_msa = 0

                if item_msa != msa:
                    msg = "All proteins with the same sequence must share the same MSA!"
                    raise ValueError(msg)

            # Set the MSA, warn if passed in single-sequence mode
            if msa == "empty":
                msa = -1
                msg = (
                    "Found explicit empty MSA for some proteins, will run "
                    "these in single sequence mode. Keep in mind that the "
                    "model predictions will be suboptimal without an MSA."
                )
                click.echo(msg)

            if msa not in (0, -1):
                is_msa_custom = True
            elif msa == 0:
                is_msa_auto = True

        # Parse a polymer
        if entity_type in {"protein", "dna", "rna"}:
            # Get token map
            if entity_type == "rna":
                token_map = const.rna_letter_to_token
            elif entity_type == "dna":
                token_map = const.dna_letter_to_token
            elif entity_type == "protein":
                token_map = const.prot_letter_to_token
            else:
                msg = f"Unknown polymer type: {entity_type}"
                raise ValueError(msg)

            # Get polymer info
            chain_type = const.chain_type_ids[entity_type.upper()]
            unk_token = const.unk_token[entity_type.upper()]

            # Extract sequence
            raw_seq = items[0][entity_type]["sequence"]
            entity_to_seq[entity_id] = raw_seq

            # Convert sequence to tokens
            seq = [token_map.get(c, unk_token) for c in list(raw_seq)]

            # Apply modifications
            for mod in items[0][entity_type].get("modifications", []):
                code = mod["ccd"]
                idx = mod["position"] - 1  # 1-indexed
                seq[idx] = code

            cyclic = items[0][entity_type].get("cyclic", False)

            # Parse a polymer
            parsed_chain = parse_polymer(
                sequence=seq,
                raw_sequence=raw_seq,
                entity=entity_id,
                chain_type=chain_type,
                components=ccd,
                cyclic=cyclic,
                mol_dir=mol_dir,
            )

        # Parse a non-polymer
        elif (entity_type == "ligand") and "ccd" in (items[0][entity_type]):
            seq = items[0][entity_type]["ccd"]

            if isinstance(seq, str):
                seq = [seq]

            if affinity and len(seq) > 1:
                msg = "Cannot compute affinity for multi residue ligands!"
                raise ValueError(msg)

            residues = []
            affinity_mw = None
            for res_idx, code in enumerate(seq):
                # Get mol
                ref_mol = get_mol(code, ccd, mol_dir)

                if affinity:
                    affinity_mw = AllChem.Descriptors.MolWt(ref_mol)

                    # Add error and warning messaging when computing affinity with ligands too large
                    if ref_mol.GetNumAtoms() > 128:
                        msg = f"The ligand for affinity is too large, ligands with more than 128 atoms are not " \
                              f"supported in the affinity prediction module"
                        raise ValueError(msg)
                    elif ref_mol.GetNumAtoms() > 56:
                        print("WARNING: the ligand used for affinity calculation is larger than 56 heavy-atoms, which "
                              "was the maximum during training, therefore the affinity output might be inaccurate.")

                # Parse residue
                residue = parse_ccd_residue(
                    name=code,
                    ref_mol=ref_mol,
                    res_idx=res_idx,
                )
                residues.append(residue)

            # Create multi ligand chain
            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=residues,
                type=const.chain_type_ids["NONPOLYMER"],
                cyclic_period=0,
                sequence=None,
                affinity=affinity,
                affinity_mw=affinity_mw,
            )

            assert not items[0][entity_type].get("cyclic", False), (
                "Cyclic flag is not supported for ligands"
            )

        elif (entity_type == "ligand") and ("smiles" in items[0][entity_type]):
            seq = items[0][entity_type]["smiles"]

            if affinity:
                seq = standardize(seq)

            mol = AllChem.MolFromSmiles(seq)
            mol = AllChem.AddHs(mol)

            # Set atom names
            canonical_order = AllChem.CanonicalRankAtoms(mol)
            Chem.AssignStereochemistry(mol, force=True, cleanIt=True)
            for atom, can_idx in zip(mol.GetAtoms(), canonical_order):
                atom_name = atom.GetSymbol().upper() + str(can_idx + 1)
                if len(atom_name) > 4:
                    msg = (
                        f"{seq} has an atom with a name longer than "
                        f"4 characters: {atom_name}."
                    )
                    raise ValueError(msg)
                atom.SetProp("name", atom_name)

            success = compute_3d_conformer(mol)
            if not success:
                msg = f"Failed to compute 3D conformer for {seq}"
                raise ValueError(msg)

            mol_no_h = AllChem.RemoveHs(mol, sanitize=False)

            if affinity:
                # Add error and warning messaging when computing affinity with ligands too large
                if mol_no_h.GetNumAtoms() > 128:
                    msg = f"The ligand for affinity is too large, ligands with more than 128 atoms are not supported in the affinity prediction module"
                    raise ValueError(msg)
                elif mol_no_h.GetNumAtoms() > 56:
                    print("WARNING: the ligand used for affinity calculation is larger than 56 heavy-atoms, "
                          "which was the maximum during training, therefore the affinity output might be inaccurate.")

            affinity_mw = AllChem.Descriptors.MolWt(mol_no_h) if affinity else None
            extra_mols[f"LIG{ligand_id}"] = mol_no_h
            residue = parse_ccd_residue(
                name=f"LIG{ligand_id}",
                ref_mol=mol,
                res_idx=0,
            )

            ligand_id += 1
            parsed_chain = ParsedChain(
                entity=entity_id,
                residues=[residue],
                type=const.chain_type_ids["NONPOLYMER"],
                cyclic_period=0,
                sequence=None,
                affinity=affinity,
                affinity_mw=affinity_mw,
            )

            assert not items[0][entity_type].get("cyclic", False), (
                "Cyclic flag is not supported for ligands"
            )

        else:
            msg = f"Invalid entity type: {entity_type}"
            raise ValueError(msg)

        # Add as many chains as provided ids
        for item in items:
            ids = item[entity_type]["id"]
            if isinstance(ids, str):
                ids = [ids]
            for chain_name in ids:
                chains[chain_name] = parsed_chain
                chain_to_msa[chain_name] = msa

    # Check if msa is custom or auto
    if is_msa_custom and is_msa_auto:
        msg = "Cannot mix custom and auto-generated MSAs in the same input!"
        raise ValueError(msg)

    # If no chains parsed fail
    if not chains:
        msg = "No chains parsed!"
        raise ValueError(msg)

    # Create tables
    atom_data = []
    bond_data = []
    res_data = []
    chain_data = []
    protein_chains = set()
    affinity_info = None

    rdkit_bounds_constraint_data = []
    chiral_atom_constraint_data = []
    stereo_bond_constraint_data = []
    planar_bond_constraint_data = []
    planar_ring_5_constraint_data = []
    planar_ring_6_constraint_data = []

    # Convert parsed chains to tables
    atom_idx = 0
    res_idx = 0
    asym_id = 0
    sym_count = {}
    chain_to_idx = {}

    # Keep a mapping of (chain_name, residue_idx, atom_name) to atom_idx
    atom_idx_map = {}

    for asym_id, (chain_name, chain) in enumerate(chains.items()):
        # Compute number of atoms and residues
        res_num = len(chain.residues)
        atom_num = sum(len(res.atoms) for res in chain.residues)

        # Save protein chains for later
        if chain.type == const.chain_type_ids["PROTEIN"]:
            protein_chains.add(chain_name)

        # Add affinity info
        if chain.affinity and affinity_info is not None:
            msg = "Cannot compute affinity for multiple ligands!"
            raise ValueError(msg)

        if chain.affinity:
            affinity_info = AffinityInfo(
                chain_id=asym_id,
                mw=chain.affinity_mw,
            )

        # Find all copies of this chain in the assembly
        entity_id = int(chain.entity)
        sym_id = sym_count.get(entity_id, 0)
        chain_data.append(
            (
                chain_name,
                chain.type,
                entity_id,
                sym_id,
                asym_id,
                atom_idx,
                atom_num,
                res_idx,
                res_num,
                chain.cyclic_period,
            )
        )
        chain_to_idx[chain_name] = asym_id
        sym_count[entity_id] = sym_id + 1

        # Add residue, atom, bond, data
        for res in chain.residues:
            atom_center = atom_idx + res.atom_center
            atom_disto = atom_idx + res.atom_disto
            res_data.append(
                (
                    res.name,
                    res.type,
                    res.idx,
                    atom_idx,
                    len(res.atoms),
                    atom_center,
                    atom_disto,
                    res.is_standard,
                    res.is_present,
                )
            )

            if res.rdkit_bounds_constraints is not None:
                for constraint in res.rdkit_bounds_constraints:
                    rdkit_bounds_constraint_data.append(  # noqa: PERF401
                        (
                            tuple(
                                c_atom_idx + atom_idx
                                for c_atom_idx in constraint.atom_idxs
                            ),
                            constraint.is_bond,
                            constraint.is_angle,
                            constraint.upper_bound,
                            constraint.lower_bound,
                        )
                    )
            if res.chiral_atom_constraints is not None:
                for constraint in res.chiral_atom_constraints:
                    chiral_atom_constraint_data.append(  # noqa: PERF401
                        (
                            tuple(
                                c_atom_idx + atom_idx
                                for c_atom_idx in constraint.atom_idxs
                            ),
                            constraint.is_reference,
                            constraint.is_r,
                        )
                    )
            if res.stereo_bond_constraints is not None:
                for constraint in res.stereo_bond_constraints:
                    stereo_bond_constraint_data.append(  # noqa: PERF401
                        (
                            tuple(
                                c_atom_idx + atom_idx
                                for c_atom_idx in constraint.atom_idxs
                            ),
                            constraint.is_check,
                            constraint.is_e,
                        )
                    )
            if res.planar_bond_constraints is not None:
                for constraint in res.planar_bond_constraints:
                    planar_bond_constraint_data.append(  # noqa: PERF401
                        (
                            tuple(
                                c_atom_idx + atom_idx
                                for c_atom_idx in constraint.atom_idxs
                            ),
                        )
                    )
            if res.planar_ring_5_constraints is not None:
                for constraint in res.planar_ring_5_constraints:
                    planar_ring_5_constraint_data.append(  # noqa: PERF401
                        (
                            tuple(
                                c_atom_idx + atom_idx
                                for c_atom_idx in constraint.atom_idxs
                            ),
                        )
                    )
            if res.planar_ring_6_constraints is not None:
                for constraint in res.planar_ring_6_constraints:
                    planar_ring_6_constraint_data.append(  # noqa: PERF401
                        (
                            tuple(
                                c_atom_idx + atom_idx
                                for c_atom_idx in constraint.atom_idxs
                            ),
                        )
                    )

            for bond in res.bonds:
                atom_1 = atom_idx + bond.atom_1
                atom_2 = atom_idx + bond.atom_2
                bond_data.append(
                    (
                        asym_id,
                        asym_id,
                        res_idx,
                        res_idx,
                        atom_1,
                        atom_2,
                        bond.type,
                    )
                )

            for atom in res.atoms:
                # Add atom to map
                atom_idx_map[(chain_name, res.idx, atom.name)] = (
                    asym_id,
                    res_idx,
                    atom_idx,
                )

                # Add atom to data
                atom_data.append(
                    (
                        atom.name,
                        atom.element,
                        atom.charge,
                        atom.coords,
                        atom.conformer,
                        atom.is_present,
                        atom.chirality,
                    )
                )
                atom_idx += 1

            res_idx += 1

    # Parse constraints
    connections = []
    pocket_constraints = []
    contact_constraints = []
    constraints = schema.get("constraints", [])
    for constraint in constraints:
        if "bond" in constraint:
            if "atom1" not in constraint["bond"] or "atom2" not in constraint["bond"]:
                msg = f"Bond constraint was not properly specified"
                raise ValueError(msg)

            c1, r1, a1 = tuple(constraint["bond"]["atom1"])
            c2, r2, a2 = tuple(constraint["bond"]["atom2"])
            c1, r1, a1 = atom_idx_map[(c1, r1 - 1, a1)]  # 1-indexed
            c2, r2, a2 = atom_idx_map[(c2, r2 - 1, a2)]  # 1-indexed
            connections.append((c1, c2, r1, r2, a1, a2))
        elif "pocket" in constraint:
            if (
                "binder" not in constraint["pocket"]
                or "contacts" not in constraint["pocket"]
            ):
                msg = f"Pocket constraint was not properly specified"
                raise ValueError(msg)

            if len(pocket_constraints) > 0 and not boltz_2:
                msg = f"Only one pocket binders is supported in Boltz-1!"
                raise ValueError(msg)

            max_distance = constraint["pocket"].get("max_distance", 6.0)
            if max_distance != 6.0 and not boltz_2:
                msg = f"Max distance != 6.0 is not supported in Boltz-1!"
                raise ValueError(msg)

            binder = constraint["pocket"]["binder"]
            binder = chain_to_idx[binder]

            contacts = []
            for chain_name, residue_index_or_atom_name in constraint["pocket"][
                "contacts"
            ]:
                contact = token_spec_to_ids(
                    chain_name,
                    residue_index_or_atom_name,
                    chain_to_idx,
                    atom_idx_map,
                    chains,
                )
                contacts.append(contact)

            force = constraint["pocket"].get("force", False)
            pocket_constraints.append((binder, contacts, max_distance, force))
        elif "contact" in constraint:
            if (
                "token1" not in constraint["contact"]
                or "token2" not in constraint["contact"]
            ):
                msg = f"Contact constraint was not properly specified"
                raise ValueError(msg)

            if not boltz_2:
                msg = f"Contact constraint is not supported in Boltz-1!"
                raise ValueError(msg)

            max_distance = constraint["contact"].get("max_distance", 6.0)

            chain_name1, residue_index_or_atom_name1 = constraint["contact"]["token1"]
            token1 = token_spec_to_ids(
                chain_name1,
                residue_index_or_atom_name1,
                chain_to_idx,
                atom_idx_map,
                chains,
            )
            chain_name2, residue_index_or_atom_name2 = constraint["contact"]["token2"]
            token2 = token_spec_to_ids(
                chain_name2,
                residue_index_or_atom_name2,
                chain_to_idx,
                atom_idx_map,
                chains,
            )
            force = constraint["contact"].get("force", False)

            contact_constraints.append((token1, token2, max_distance, force))
        else:
            msg = f"Invalid constraint: {constraint}"
            raise ValueError(msg)

    # Get protein sequences in this YAML
    protein_seqs = {name: chains[name].sequence for name in protein_chains}

    # Parse templates
    template_schema = schema.get("templates", [])
    if template_schema and not boltz_2:
        msg = "Templates are not supported in Boltz 1.0!"
        raise ValueError(msg)

    templates = {}
    template_records = []
    for template in template_schema:
        if "cif" in template:
            path = template["cif"]
            pdb = False
        elif "pdb" in template:
            path = template["pdb"]
            pdb = True
        else:
            msg = "Template was not properly specified, missing CIF or PDB path!"
            raise ValueError(msg)

        template_id = Path(path).stem
        chain_ids = template.get("chain_id", None)
        template_chain_ids = template.get("template_id", None)

        # Check validity of input
        matched = False

        if chain_ids is not None and not isinstance(chain_ids, list):
            chain_ids = [chain_ids]
        if template_chain_ids is not None and not isinstance(template_chain_ids, list):
            template_chain_ids = [template_chain_ids]

        if (
            template_chain_ids is not None
            and chain_ids is not None
        ):
           
                if len(template_chain_ids) == len(chain_ids):
                     if len(template_chain_ids) > 0 and len(chain_ids) > 0:
                        matched = True
                else:
                    msg = (
                        "When providing both the chain_id and template_id, the number of"
                        "template_ids provided must match the number of chain_ids!"
                    )
                    raise ValueError(msg)

        # Get relevant chains ids
        if chain_ids is None:
            chain_ids = list(protein_chains)

        for chain_id in chain_ids:
            if chain_id not in protein_chains:
                msg = (
                    f"Chain {chain_id} assigned for template"
                    f"{template_id} is not one of the protein chains!"
                )
                raise ValueError(msg)

        # Get relevant template chain ids
        if pdb:
            parsed_template = parse_pdb(
                path,
                mols=ccd,
                moldir=mol_dir,
                use_assembly=False,
                compute_interfaces=False,
            )
        else:
            parsed_template = parse_mmcif(
                path,
                mols=ccd,
                moldir=mol_dir,
                use_assembly=False,
                compute_interfaces=False,
            )
        template_proteins = {
            str(c["name"])
            for c in parsed_template.data.chains
            if c["mol_type"] == const.chain_type_ids["PROTEIN"]
        }
        if template_chain_ids is None:
            template_chain_ids = list(template_proteins)

        for chain_id in template_chain_ids:
            if chain_id not in template_proteins:
                msg = (
                    f"Template chain {chain_id} assigned for template"
                    f"{template_id} is not one of the protein chains!"
                )
                raise ValueError(msg)

        force = template.get("force", False)
        if force:
            if "threshold" in template:
                threshold = template["threshold"]
            else:
                msg = f"Template {template_id} must have threshold specified if force is set to True"
                raise ValueError(msg)
        else:
            threshold = float("inf")
        # Compute template records

        if matched:
            template_records.extend(
                get_template_records_from_matching(
                    template_id=template_id,
                    chain_ids=chain_ids,
                    sequences=protein_seqs,
                    template_chain_ids=template_chain_ids,
                    template_sequences=parsed_template.sequences,
                    force=force,
                    threshold=threshold,
                )
            )
        else:
            template_records.extend(
                get_template_records_from_search(
                    template_id=template_id,
                    chain_ids=chain_ids,
                    sequences=protein_seqs,
                    template_chain_ids=template_chain_ids,
                    template_sequences=parsed_template.sequences,
                    force=force,
                    threshold=threshold,
                )
            )
        # Save template
        templates[template_id] = parsed_template.data

    # Convert into datatypes
    residues = np.array(res_data, dtype=Residue)
    chains = np.array(chain_data, dtype=Chain)
    interfaces = np.array([], dtype=Interface)
    mask = np.ones(len(chain_data), dtype=bool)
    rdkit_bounds_constraints = np.array(
        rdkit_bounds_constraint_data, dtype=RDKitBoundsConstraint
    )
    chiral_atom_constraints = np.array(
        chiral_atom_constraint_data, dtype=ChiralAtomConstraint
    )
    stereo_bond_constraints = np.array(
        stereo_bond_constraint_data, dtype=StereoBondConstraint
    )
    planar_bond_constraints = np.array(
        planar_bond_constraint_data, dtype=PlanarBondConstraint
    )
    planar_ring_5_constraints = np.array(
        planar_ring_5_constraint_data, dtype=PlanarRing5Constraint
    )
    planar_ring_6_constraints = np.array(
        planar_ring_6_constraint_data, dtype=PlanarRing6Constraint
    )

    if boltz_2:
        atom_data = [(a[0], a[3], a[5], 0.0, 1.0) for a in atom_data]
        connections = [(*c, const.bond_type_ids["COVALENT"]) for c in connections]
        bond_data = bond_data + connections
        atoms = np.array(atom_data, dtype=AtomV2)
        bonds = np.array(bond_data, dtype=BondV2)
        coords = [(x,) for x in atoms["coords"]]
        coords = np.array(coords, Coords)
        ensemble = np.array([(0, len(coords))], dtype=Ensemble)
        data = StructureV2(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            interfaces=interfaces,
            mask=mask,
            coords=coords,
            ensemble=ensemble,
        )
    else:
        bond_data = [(b[4], b[5], b[6]) for b in bond_data]
        atom_data = [(convert_atom_name(a[0]), *a[1:]) for a in atom_data]
        atoms = np.array(atom_data, dtype=Atom)
        bonds = np.array(bond_data, dtype=Bond)
        connections = np.array(connections, dtype=Connection)
        data = Structure(
            atoms=atoms,
            bonds=bonds,
            residues=residues,
            chains=chains,
            connections=connections,
            interfaces=interfaces,
            mask=mask,
        )

    # Create metadata
    struct_info = StructureInfo(num_chains=len(chains))
    chain_infos = []
    for chain in chains:
        chain_info = ChainInfo(
            chain_id=int(chain["asym_id"]),
            chain_name=chain["name"],
            mol_type=int(chain["mol_type"]),
            cluster_id=-1,
            msa_id=chain_to_msa[chain["name"]],
            num_residues=int(chain["res_num"]),
            valid=True,
            entity_id=int(chain["entity_id"]),
        )
        chain_infos.append(chain_info)

    options = InferenceOptions(
        pocket_constraints=pocket_constraints, contact_constraints=contact_constraints
    )
    record = Record(
        id=name,
        structure=struct_info,
        chains=chain_infos,
        interfaces=[],
        inference_options=options,
        templates=template_records,
        affinity=affinity_info,
    )

    residue_constraints = ResidueConstraints(
        rdkit_bounds_constraints=rdkit_bounds_constraints,
        chiral_atom_constraints=chiral_atom_constraints,
        stereo_bond_constraints=stereo_bond_constraints,
        planar_bond_constraints=planar_bond_constraints,
        planar_ring_5_constraints=planar_ring_5_constraints,
        planar_ring_6_constraints=planar_ring_6_constraints,
    )
    return Target(
        record=record,
        structure=data,
        sequences=entity_to_seq,
        residue_constraints=residue_constraints,
        templates=templates,
        extra_mols=extra_mols,
    )


def standardize(smiles: str) -> Optional[str]:
    """Standardize a molecule and return its SMILES and a flag indicating whether the molecule is valid.
    This version has exception handling, which the original in mol-finder/data doesn't have. I didn't change the mol-finder/data
    since there are a lot of other functions that depend on it and I didn't want to break them.
    """
    LARGEST_FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()

    mol = Chem.MolFromSmiles(smiles, sanitize=False)

    exclude = exclude_flag(mol, includeRDKitSanitization=False)

    if exclude:
        raise ValueError("Molecule is excluded")

    # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
    # Choose molecule with largest component
    mol = LARGEST_FRAGMENT_CHOOSER.choose(mol)
    # Standardize with ChEMBL data curation pipeline. During standardization, the molecule may be broken
    mol = standardize_mol(mol)
    smiles = Chem.MolToSmiles(mol)

    # Check if molecule can be parsed by RDKit (in rare cases, the molecule may be broken during standardization)
    if Chem.MolFromSmiles(smiles) is None:
        raise ValueError("Molecule is broken")

    return smiles


