import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from protenix_reference import remap_transition
from tt_bio.tenstorrent import get_device, Transition, CORE_GRID_MAIN as CORE
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
P='module.diffusion_module.diffusion_conditioning.'
g=lambda k: ck[P+k]; sub=lambda q:{k[len(P+q)+1:]:v for k,v in ck.items() if k.startswith(P+q)}
dg=pickle.load(open('/home/ttuser/protenix_diffusion_gold.pkl','rb'))['cond']
s_single_gold=dg['out'][0].float()                       # (1,38,384)
kw=dg['kwargs']; t_hat=dg['in'][0].float()
s_trunk=kw['s_trunk'].float(); s_inputs=kw['s_inputs'].float()
sigma=16.0
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
lin=lambda x,w: ttnn.linear(x,T(w.t().contiguous()),compute_kernel_config=ckc,core_grid=CORE)
# single_s
ss=torch.cat([s_trunk,s_inputs],-1)                       # (38, 833)
ss=ttnn.layer_norm(T(ss), weight=T(g('layernorm_s.weight')), epsilon=1e-5, compute_kernel_config=ckc)
ss=lin(ss, g('linear_no_bias_s.weight'))                  # (38,384)
# fourier noise (host, tiny)
tp=torch.log(t_hat/sigma)/4
fou=torch.cos(2*torch.pi*(tp.unsqueeze(-1)*g('fourier_embedding.w')+g('fourier_embedding.b')))  # (1,256)
nn_=ttnn.layer_norm(T(fou), weight=T(g('layernorm_n.weight')), epsilon=1e-5, compute_kernel_config=ckc)
nn_=lin(nn_, g('linear_no_bias_n.weight'))                # (1,384)
ss=ttnn.add(ss, nn_)                                      # broadcast
ss=ttnn.reshape(ss,(1,38,384))
t1=Transition(remap_transition(sub('transition_s1')), ckc); ss=ttnn.add(ss, ttnn.reshape(t1(ss),tuple(ss.shape)))
t2=Transition(remap_transition(sub('transition_s2')), ckc); ss=ttnn.add(ss, ttnn.reshape(t2(ss),tuple(ss.shape)))
out=torch.Tensor(ttnn.to_torch(ss)).float().reshape(s_single_gold.shape)
def pcc(a,b):
    a=a.flatten().double();b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('DiffusionConditioning single_s PCC %.5f'%pcc(out,s_single_gold),flush=True)
