# Capture golden Protenix-v2 inference feature dicts for nucleic-acid (and mixed)
# inputs from the reference venv, so the tt-bio NA featurizer can be validated for
# exact parity without a protenix install in system python3.
# Run with the reference venv:  /home/ttuser/protenix_ref_venv/bin/python3 scripts/protenix_extract_na_gold.py
import os, pickle, torch
os.environ.setdefault("PROTENIX_ROOT_DIR", "/home/ttuser")
from protenix.data.inference.json_to_feature import SampleDictToFeatures

SAMPLES = {
    "rna": {"sequences": [{"rnaSequence": {"sequence": "GACUUA", "count": 1}}], "name": "rna"},
    "dna": {"sequences": [{"dnaSequence": {"sequence": "GATTCA", "count": 1}}], "name": "dna"},
    "prot_rna": {"sequences": [
        {"proteinChain": {"sequence": "GSSGSSG", "count": 1}},
        {"rnaSequence": {"sequence": "GACUA", "count": 1}},
    ], "name": "prot_rna"},
}

KEEP = ["restype", "residue_index", "asym_id", "entity_id", "sym_id", "token_index",
        "token_bonds", "ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars",
        "ref_mask", "atom_to_token_idx", "ref_space_uid", "atom_to_tokatom_idx",
        "distogram_rep_atom_mask", "msa", "has_deletion", "deletion_value", "profile",
        "deletion_mean"]

out = {}
for name, sample in SAMPLES.items():
    feat, atom_array, token_array = SampleDictToFeatures(sample).get_feature_dict()
    d = {}
    for k in KEEP:
        if k in feat:
            v = feat[k]
            d[k] = v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v)
    # atom names (decoded) for cross-checking the writer
    d["_atom_names"] = [a.atom_name for a in atom_array]
    d["_res_names"] = [a.res_name for a in atom_array]
    out[name] = d
    print(name, "n_token", d["restype"].shape[0], "n_atom", d["ref_pos"].shape[0],
          "keys", sorted(d.keys()), flush=True)

pickle.dump(out, open("/home/ttuser/protenix_na_gold.pkl", "wb"))
print("SAVED /home/ttuser/protenix_na_gold.pkl", flush=True)
