import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, torch.nn.functional as F, ttnn
from protenix_reference import remap_msa_pair_stack
from tt_bio.tenstorrent import get_device, PairformerLayer
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
g=lambda k: ck['module.template_embedder.'+k]
sub=lambda P: {k[len('module.template_embedder.'+P)+1:]:v for k,v in ck.items() if k.startswith('module.template_embedder.'+P)}
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
te=d['intermediates']['template_embedder']; feat=te['in'][0]; z_in=te['in'][1].float(); gold=te['out'].float()
N=z_in.shape[0]; nt=feat['template_aatype'].shape[0]
asym=feat['asym_id']; mc=(asym[:,None]==asym[None,:]).float(); pm=torch.ones(N,N)
dev=get_device()
ckc=ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
lin=lambda x,w: ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8,x=8))
# z_n = layernorm_z(z)
z_n=ttnn.layer_norm(T(z_in.unsqueeze(0)), weight=T(g('layernorm_z.weight')), bias=T(g('layernorm_z.bias')), epsilon=1e-5, compute_kernel_config=ckc)
u=None
for t in range(nt):
    dg=feat['template_distogram'][t]*mc[...,None]*pm[...,None]                 # (38,38,39)
    pb=(feat['template_pseudo_beta_mask'][t]*mc*pm).unsqueeze(-1)             # (38,38,1)
    aa=F.one_hot(feat['template_aatype'][t].long(),32).float()               # (38,32)
    aai=aa[None,:,:].expand(N,N,32); aaj=aa[:,None,:].expand(N,N,32)
    uv=feat['template_unit_vector'][t]*mc[...,None]*pm[...,None]              # (38,38,3)
    bb=(feat['template_backbone_frame_mask'][t]*mc*pm).unsqueeze(-1)         # (38,38,1)
    at=torch.cat([dg,pb,aai,aaj,uv,bb],-1)                                   # (38,38,108)
    v=ttnn.add(lin(T(at.unsqueeze(0)), g('linear_no_bias_a.weight')), lin(z_n, g('linear_no_bias_z.weight')))  # (1,38,38,64)
    for b in range(2):
        pl=PairformerLayer(32,2,None,None,False, remap_msa_pair_stack(sub(f'pairformer_stack.blocks.{b}')), ckc)
        v=pl(None,v)[1]
    v=ttnn.layer_norm(v, weight=T(g('layernorm_v.weight')), bias=T(g('layernorm_v.bias')), epsilon=1e-5, compute_kernel_config=ckc)
    u=v if u is None else ttnn.add(u,v)
u=ttnn.multiply(u, 1.0/(1e-7+nt))
u=lin(ttnn.relu(u), g('linear_no_bias_u.weight'))                            # (1,38,38,256)
out=torch.Tensor(ttnn.to_torch(u)).float().reshape(gold.shape)
def pcc(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('TemplateEmbedder z-update PCC %.5f'%pcc(out,gold), flush=True)
