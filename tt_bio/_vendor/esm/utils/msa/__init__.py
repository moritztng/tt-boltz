# Vendored from github.com/Biohub/esm @ b6b0e88 (MIT, Copyright 2026 Chan Zuckerberg
# Biohub, Inc.; see tt_bio/_vendor/esm/LICENSE). Modified: absolute `esm.` imports
# rewritten to `tt_bio._vendor.esm.` for in-tree vendoring.
from tt_bio._vendor.esm.utils.msa.msa import MSA, FastMSA, remove_insertions_from_sequence

__all__ = ["MSA", "FastMSA", "remove_insertions_from_sequence"]
