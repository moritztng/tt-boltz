import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from protenix_reference import remap_transition
from tt_bio.tenstorrent import get_device, Transition, CORE_GRID_MAIN as CORE
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
P='module.diffusion_module.diffusion_conditioning.'
g=lambda k: ck[P+k]; sub=lambda q:{k[len(P+q)+1:]:v for k,v in ck.items() if k.startswith(P+q)}
tg=pickle.load(open('/home/ttuser/protenix_trunk_gold.pkl','rb')); z_trunk=tg['z'].float()
trk=pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl','rb')); relp=trk['relp']
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
pair_z_gold=d['intermediates']['diffusion_module']['kwargs']['pair_z'].float()
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
lin=lambda x,w: ttnn.linear(x,T(w.t().contiguous()),compute_kernel_config=ckc,core_grid=CORE)
relpe=lin(T(relp), g('relpe.linear_no_bias.weight'))          # (38,38,256)
zc=ttnn.concat([T(z_trunk), relpe], dim=-1)                    # (38,38,512)
zc=ttnn.layer_norm(zc, weight=T(g('layernorm_z.weight')), epsilon=1e-5, compute_kernel_config=ckc)
pz=lin(zc, g('linear_no_bias_z.weight'))                       # (38,38,256)
pz=ttnn.reshape(pz,(1,38,38,256))
t1=Transition(remap_transition(sub('transition_z1')), ckc); pz=ttnn.add(pz, ttnn.reshape(t1(pz),tuple(pz.shape)))
t2=Transition(remap_transition(sub('transition_z2')), ckc); pz=ttnn.add(pz, ttnn.reshape(t2(pz),tuple(pz.shape)))
out=torch.Tensor(ttnn.to_torch(pz)).float().reshape(pair_z_gold.shape)
def pcc(a,b):
    a=a.flatten().double();b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('DiffusionConditioning pair_z PCC %.5f'%pcc(out,pair_z_gold),flush=True)
