# Vendored from github.com/Biohub/esm @ b6b0e88 (MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc.; see tt_boltz/_vendor/esm/LICENSE). Modified: absolute `esm.` imports
# rewritten to `tt_boltz._vendor.esm.` for in-tree vendoring.
"""Re-exports of the canonical SPI dataclasses from input_builder.

This module exists so the HF processor and downstream code can import the
ESMFold2 input types from a single namespace without picking up internal-only
sibling utilities. The actual definitions live in
``esm.utils.structure.input_builder``.
"""

from tt_boltz._vendor.esm.utils.msa import MSA
from tt_boltz._vendor.esm.utils.parsing import FastaEntry
from tt_boltz._vendor.esm.utils.structure.input_builder import (
    CovalentBond,
    DistogramConditioning,
    DNAInput,
    LigandInput,
    Modification,
    ProteinInput,
    RNAInput,
    StructurePredictionInput,
)

__all__ = [
    "FastaEntry",
    "MSA",
    "Modification",
    "ProteinInput",
    "RNAInput",
    "DNAInput",
    "LigandInput",
    "DistogramConditioning",
    "CovalentBond",
    "StructurePredictionInput",
]
