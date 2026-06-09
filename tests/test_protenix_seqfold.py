# Integration test: sequence -> build_protein_features -> on-device Protenix.fold ->
# valid structure + pLDDT, with no protenix dependency. Covers the assembled data pipeline
# + model + confidence head. Gated on the v2 checkpoint.
import os, re, sys, subprocess
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.environ.get("PROTENIX_CKPT", os.path.expanduser("~/protenix_ckpt/protenix-v2.pt"))

pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="needs the v2 checkpoint")


def test_protenix_sequence_to_structure():
    out = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "protenix_seqfold.py"),
                          "GSSGSSGQITLWQRPLVT", "8"],
                         cwd=ROOT, capture_output=True, text=True, timeout=600,
                         env={**os.environ, "PROTENIX_CKPT": CKPT}).stdout
    assert "SEQFOLD_DONE" in out, f"seqfold did not finish:\n{out[-2000:]}"
    m = re.search(r"finite=(\w+) Rg=([0-9.]+) plddt=([0-9.]+)", out)
    assert m is not None, f"no result line:\n{out[-2000:]}"
    finite, rg, plddt = m.group(1), float(m.group(2)), float(m.group(3))
    assert finite == "True", "non-finite coords"
    assert rg > 5.0, f"structure collapsed (Rg {rg})"
    assert 0.0 <= plddt <= 1.0, f"pLDDT {plddt} out of range"
