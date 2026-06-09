import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN as CORE
from tt_bio.protenix import AtomTransformer
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
P='module.diffusion_module.atom_attention_decoder.'; g=lambda k: ck[P+k]
gd=pickle.load(open('/home/ttuser/protenix_atomdec_pre.pkl','rb')); kw=gd['kwargs']
a2t=kw['atom_to_token_idx'].long()
a=kw['a'].float(); a=a[0] if a.dim()==3 else a            # (38,768)
q_skip=kw['q_skip'].float(); q_skip=q_skip[0] if q_skip.dim()==3 else q_skip   # (275,128)
c_skip=kw['c_skip'].float(); c_skip=c_skip[0] if c_skip.dim()==3 else c_skip
p_skip=kw['p_skip'].float(); p_skip=p_skip[0] if p_skip.dim()==5 else p_skip   # (9,32,128,16)
cg=gd['out'].float()       # (1,275,3)
N=q_skip.shape[0]; NT=a.shape[0]
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
lin=lambda x,w: ttnn.linear(x,T(w.t().contiguous()),compute_kernel_config=ckc,core_grid=CORE)
# q = broadcast(linear_a(a)) + q_skip
a_atom_proj=lin(T(a),g('linear_no_bias_a.weight'))  # (38,128)
S=torch.zeros(N,NT); S[torch.arange(N),a2t]=1.0
q=ttnn.add(ttnn.matmul(T(S),a_atom_proj,compute_kernel_config=ckc,core_grid=CORE), T(q_skip))  # (275,128)
# atom transformer
atx=AtomTransformer(3, {k[len(P+'atom_transformer.'):]:v for k,v in ck.items() if k.startswith(P+'atom_transformer.')}, ckc)
# need mask_trunked for windowed attn — reuse from atomenc pre (same protein)
mt=pickle.load(open('/home/ttuser/protenix_atomenc_pre.pkl','rb'))['in'][8]['mask_trunked'].float()
q_out=atx(ttnn.reshape(q,(1,N,128)), ttnn.reshape(T(c_skip),(1,N,128)), T(p_skip), mt)
# coords = linear_out(layernorm_q(q_out))
qn=ttnn.layer_norm(q_out, weight=T(g('layernorm_q.weight')), epsilon=1e-5, compute_kernel_config=ckc)
coords=lin(qn, g('linear_no_bias_out.weight'))  # (1,275,3)
out=torch.Tensor(ttnn.to_torch(coords)).float().reshape(1,N,3)[:, :N]
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('atom decoder coords PCC %.5f  maxerr %.3e'%(pcc(out,cg),(out-cg).abs().max()),flush=True)
