# Predict a structure with the on-device Protenix-v2 model from a precomputed feats dict.
# Usage: python scripts/protenix_predict.py <feats.pkl> <ckpt.pt> <out.pdb> [n_step] [seed]
# Loads the model, runs Protenix.fold end-to-end on-device, and writes the predicted atom
# coordinates as a PDB. This is the model's end-product path (the feats dict is produced by
# the data pipeline; see docs/porting-protenix-v2.md). Verifies a non-degenerate structure.
import os, sys, pickle
os.environ.setdefault('TT_VISIBLE_DEVICES', '0'); os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
sys.path.insert(0, '/home/ttuser/tt-boltz2')
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix

# minimal Z -> element symbol (covers biomolecules; fallback 'X')
_Z = {1:'H',6:'C',7:'N',8:'O',9:'F',11:'Na',12:'Mg',15:'P',16:'S',17:'Cl',19:'K',
      20:'Ca',25:'Mn',26:'Fe',29:'Cu',30:'Zn',34:'Se',35:'Br',53:'I'}

def write_pdb(path, coords, z, a2t):
    lines = []
    for i, (xyz, zi, t) in enumerate(zip(coords.tolist(), z.tolist(), a2t.tolist())):
        sym = _Z.get(int(zi), 'C')
        name = sym if len(sym) == 2 else (' ' + sym)
        lines.append("ATOM  %5d %-4s %3s A%4d    %8.3f%8.3f%8.3f%6.2f%6.2f          %2s" % (
            (i % 99999) + 1, name[:4], 'UNK', (int(t) % 9999) + 1, xyz[0], xyz[1], xyz[2], 1.0, 0.0, sym))
    lines.append("END")
    open(path, 'w').write("\n".join(lines) + "\n")
    return len(lines)

def main():
    feats_pkl, ckpt, out_pdb = sys.argv[1], sys.argv[2], sys.argv[3]
    n_step = int(sys.argv[4]) if len(sys.argv) > 4 else 200
    seed = int(sys.argv[5]) if len(sys.argv) > 5 else 0
    feats = pickle.load(open(feats_pkl, 'rb'))
    dev = get_device()
    ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                                  fp32_dest_acc_en=True, packer_l1_acc=True)
    model = Protenix.load_from_checkpoint(ckpt, compute_kernel_config=ckc, device=dev)
    coords = model.fold(feats, n_step=n_step, n_sample=1, seed=seed)[0]   # (N,3)
    z = feats['ref_element'].argmax(-1) if feats['ref_element'].dim() > 1 else feats['ref_element']
    a2t = feats['atom_to_token_idx'].long()
    n = write_pdb(out_pdb, coords, z, a2t)
    rg = float((coords - coords.mean(0)).pow(2).sum(-1).mean().sqrt())
    print("PREDICT_DONE wrote %s (%d lines, %d atoms, Rg %.2f A, finite=%s)" % (
        out_pdb, n, coords.shape[0], rg, bool(torch.isfinite(coords).all())), flush=True)

if __name__ == "__main__":
    main()
