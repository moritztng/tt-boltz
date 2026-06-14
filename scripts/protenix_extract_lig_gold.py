# Capture golden Protenix-v2 inference feature dicts for LIGAND inputs (CCD code + SMILES,
# and a protein+ligand complex) from the reference venv -> ~/protenix_lig_gold.pkl.
# Run with:  /home/ttuser/protenix_ref_venv/bin/python3 scripts/protenix_extract_lig_gold.py
import os, pickle, torch
os.environ.setdefault("PROTENIX_ROOT_DIR", "/home/ttuser")
from protenix.data.inference.json_to_feature import SampleDictToFeatures

SAMPLES = {
    "ccd_sah": {"sequences": [{"ligand": {"ligand": "CCD_SAH", "count": 1}}], "name": "ccd_sah"},
    "smiles_tyr": {"sequences": [{"ligand": {"ligand": "N[C@@H](Cc1ccc(O)cc1)C(=O)O", "count": 1}}], "name": "smiles_tyr"},
    "prot_lig": {"sequences": [
        {"proteinChain": {"sequence": "GSSGSSG", "count": 1}},
        {"ligand": {"ligand": "CCD_SAH", "count": 1}},
    ], "name": "prot_lig"},
}

KEEP = ["restype", "residue_index", "asym_id", "entity_id", "sym_id", "token_index",
        "token_bonds", "ref_pos", "ref_element", "ref_charge", "ref_atom_name_chars",
        "ref_mask", "atom_to_token_idx", "ref_space_uid", "atom_to_tokatom_idx",
        "distogram_rep_atom_mask"]

out = {}
for name, sample in SAMPLES.items():
    try:
        feat, atom_array, token_array = SampleDictToFeatures(sample).get_feature_dict()
    except Exception as e:
        print(name, "ERR", repr(e), flush=True); continue
    d = {}
    for k in KEEP:
        if k in feat:
            v = feat[k]
            d[k] = v.detach().cpu() if torch.is_tensor(v) else torch.as_tensor(v)
    d["_atom_names"] = [a.atom_name for a in atom_array]
    d["_res_names"] = [a.res_name for a in atom_array]
    d["_mol_types"] = [a.mol_type for a in atom_array]
    out[name] = d
    print(name, "n_token", d["restype"].shape[0], "n_atom", d["ref_pos"].shape[0],
          "token_bonds_sum", float(d["token_bonds"].sum()), flush=True)

pickle.dump(out, open("/home/ttuser/protenix_lig_gold.pkl", "wb"))
print("SAVED /home/ttuser/protenix_lig_gold.pkl", flush=True)
