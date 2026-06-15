# Sequence->structure integration check: build_protein_features(seq) -> Protenix.fold ->
# coords + pLDDT, entirely on-device with no protenix dependency. Verifies the assembled
# data pipeline + model produce a finite, non-collapsed structure and a sane pLDDT.
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES', '0'); os.environ.setdefault('TT_LOGGER_LEVEL', 'FATAL')
sys.path.insert(0, '/home/ttuser/tt-boltz2')
import torch, ttnn
from tt_bio.tenstorrent import get_device
from tt_bio.protenix import Protenix
from tt_bio.protenix_data import build_protein_features

CKPT = os.environ.get('PROTENIX_CKPT', '/home/ttuser/protenix_ckpt/protenix-v2.pt')
SEQ = sys.argv[1] if len(sys.argv) > 1 else 'GSSGSSGQITLWQRPLVTIKIGGQLKEALLDTGADDTV'
N_STEP = int(sys.argv[2]) if len(sys.argv) > 2 else 10

feats = build_protein_features(SEQ)
dev = get_device()
ckc = ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4,
                                             fp32_dest_acc_en=True, packer_l1_acc=True)
model = Protenix.load_from_checkpoint(CKPT, compute_kernel_config=ckc, device=dev)
coords, conf = model.fold(feats, n_step=N_STEP, n_sample=1, seed=0, return_confidence=True)
x = coords[0]
rg = float((x - x.mean(0)).pow(2).sum(-1).mean().sqrt())
print('SEQFOLD seq_len=%d n_atoms=%d finite=%s Rg=%.2f plddt=%.4f' % (
    len(SEQ), x.shape[0], bool(torch.isfinite(coords).all()), rg, conf['plddt']), flush=True)
print('SEQFOLD_DONE', flush=True)
