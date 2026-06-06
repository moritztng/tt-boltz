# Vendored from github.com/Biohub/esm @ b6b0e88 (MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc.; see tt_boltz/_vendor/esm/LICENSE). Modified: absolute `esm.` imports
# rewritten to `tt_boltz._vendor.esm.` for in-tree vendoring.
from tt_boltz._vendor.esm.models.esmfold2.conformers import load_ccd
from tt_boltz._vendor.esm.models.esmfold2.constants import ELEMENT_NUMBER_TO_SYMBOL
from tt_boltz._vendor.esm.models.esmfold2.prepare_input import ChainInfo, prepare_esmfold2_input
from tt_boltz._vendor.esm.models.esmfold2.processor import ESMFold2InputBuilder, clean_esmfold2_input
from tt_boltz._vendor.esm.models.esmfold2.types import (
    MSA,
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)
from tt_boltz._vendor.esm.utils.structure.molecular_complex import (
    MolecularComplex,
    MolecularComplexMetadata,
    MolecularComplexResult,
)

__all__ = [
    "ChainInfo",
    "CovalentBond",
    "DistogramConditioning",
    "DNAInput",
    "ELEMENT_NUMBER_TO_SYMBOL",
    "ESMFold2InputBuilder",
    "LigandInput",
    "MSA",
    "Modification",
    "MolecularComplex",
    "MolecularComplexMetadata",
    "MolecularComplexResult",
    "ProteinInput",
    "RNAInput",
    "StructurePredictionInput",
    "clean_esmfold2_input",
    "load_ccd",
    "prepare_esmfold2_input",
]
