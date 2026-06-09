import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, torch.nn.functional as F, ttnn
from protenix_reference import (remap_outer_product_mean, remap_pair_weighted_averaging,
                                remap_transition, remap_msa_pair_stack)
from tt_bio.tenstorrent import (get_device, OuterProductMean, PairWeightedAveraging, Transition, PairformerLayer)
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
g=lambda k: ck['module.msa_module.'+k]
sub=lambda P: {k[len('module.msa_module.'+P)+1:]:v for k,v in ck.items() if k.startswith('module.msa_module.'+P)}
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
io=d['intermediates']['msa_module']; feat=io['in'][0]; z_in=io['in'][1].float(); s_inputs=io['in'][2].float(); z_gold=io['out'].float()
dev=get_device()
ckc=ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
def T(x): return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
def lin(x,w): return ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8,x=8))
# input featurization (N_msa=1)
msa=F.one_hot(feat['msa'].long(),32).float()           # (1,38,32)
ms=torch.cat([msa, feat['has_deletion'].unsqueeze(-1), feat['deletion_value'].unsqueeze(-1)],-1)  # (1,38,34)
ms=ms.unsqueeze(0)                                      # (1,1,38,34)
m=lin(T(ms),g('linear_no_bias_m.weight'))               # (1,1,38,128)
m=ttnn.add(m, lin(T(s_inputs),g('linear_no_bias_s.weight')))  # broadcast (38,128)->(1,1,38,128)
z=T(z_in.unsqueeze(0))                                   # (1,38,38,256)
for i in range(4):
    P=f'blocks.{i}.'
    opm=OuterProductMean(remap_outer_product_mean(sub(P+'outer_product_mean_msa')), ckc)
    pl=PairformerLayer(32,8,None,None,False, remap_msa_pair_stack(sub(P+'pair_stack')), ckc)
    z=ttnn.add(z, opm(m,None,None))
    has_msa_stack=any(k.startswith('blocks.%d.msa_stack.'%i) for k in (kk[len('module.msa_module.'):] for kk in ck if kk.startswith('module.msa_module.')))
    if has_msa_stack:
        pwa=PairWeightedAveraging(8,8,remap_pair_weighted_averaging(sub(P+'msa_stack.msa_pair_weighted_averaging')), ckc)
        tm=Transition(remap_transition(sub(P+'msa_stack.transition_m')), ckc)
        m=ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
        m=ttnn.add(m, ttnn.reshape(tm(m), tuple(m.shape)))
    z=pl(None,z)[1]
zo=torch.Tensor(ttnn.to_torch(z)).float().reshape(z_gold.shape)
def pcc(u,v):
    u=u.flatten().double(); v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('MSA module (4 blocks) z PCC %.5f'%pcc(zo,z_gold), flush=True)
