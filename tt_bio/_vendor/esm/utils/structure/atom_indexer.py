# Vendored from github.com/Biohub/esm @ b6b0e88 (MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc.; see tt_bio/_vendor/esm/LICENSE). Modified: absolute `esm.` imports
# rewritten to `tt_bio._vendor.esm.` for in-tree vendoring.
import numpy as np

from tt_bio._vendor.esm.utils.structure.protein_structure import index_by_atom_name


class AtomIndexer:
    def __init__(self, structure, property: str, dim: int):
        self.structure = structure
        self.property = property
        self.dim = dim

    def __getitem__(self, atom_names: str | list[str]) -> np.ndarray:
        return index_by_atom_name(
            getattr(self.structure, self.property), atom_names, self.dim
        )
