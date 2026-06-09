import os, sys, re
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN as CORE
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
P='module.diffusion_module.diffusion_transformer.'
DMS=pickle.load(open('/home/ttuser/protenix_dm_stages.pkl','rb'))
a0=DMS['dit_in'].float()[0]; s0=DMS['s_single'].float()[0]; gold=DMS['dit_out'].float()[0]
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb'))['kwargs']; pair_z=D['pair_z'].float()
import torch.nn.functional as F
z0=F.layer_norm(pair_z,(pair_z.shape[-1],))  # self.normalize
N=a0.shape[0]; nh,hd=16,48; F32=ttnn.float32
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
def T(x): return ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=F32)
def lin(x,wk,bk=None):
    w=T(ck[P+wk].float().t().contiguous()); b=T(ck[P+bk].float()) if bk else None
    return ttnn.linear(x,w,bias=b,compute_kernel_config=ckc,core_grid=CORE,dtype=F32)
def ln(x,wk):  # weight-only layernorm
    return ttnn.layer_norm(x,weight=T(ck[P+wk].float()),epsilon=1e-5,compute_kernel_config=ckc)
def ln_noaff(x): return ttnn.layer_norm(x,epsilon=1e-5,compute_kernel_config=ckc)
def adaln(a,s,pre):
    an=ln_noaff(a); sn=ttnn.multiply(ln_noaff(s),T(ck[P+pre+'layernorm_s.weight'].float()))
    sc=lin(sn,pre+'linear_s.weight',pre+'linear_s.bias'); sb=lin(sn,pre+'linear_nobias_s.weight')
    return ttnn.add(ttnn.multiply(an,sc,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID]),sb)
a_t=T(a0); s_t=T(s0); z_t=T(z0)
for b in range(24):
    A=f'blocks.{b}.attention_pair_bias.'; C=f'blocks.{b}.conditioned_transition_block.'
    an=adaln(a_t,s_t,A+'layernorm_a.')
    zb=ttnn.multiply(ln_noaff(z_t),T(ck[P+A+'layernorm_z.weight'].float()))
    bias=lin(zb,A+'linear_nobias_z.weight')                      # (N,N,16)
    bias=ttnn.permute(bias,(2,0,1))                              # (16,N,N)
    Q=lin(an,A+'attention.linear_q.weight',A+'attention.linear_q.bias'); K=lin(an,A+'attention.linear_k.weight'); V=lin(an,A+'attention.linear_v.weight')
    def heads(x): return ttnn.permute(ttnn.reshape(x,(N,nh,hd)),(1,0,2))   # (16,N,48)
    Qh=heads(Q); Kh=heads(K); Vh=heads(V)
    sc=ttnn.matmul(Qh,ttnn.permute(Kh,(0,2,1)),compute_kernel_config=ckc)  # (16,N,N)
    sc=ttnn.add(ttnn.multiply(sc,hd**-0.5),bias)
    o=ttnn.matmul(ttnn.softmax(sc,dim=-1),Vh,compute_kernel_config=ckc)    # (16,N,48)
    o=ttnn.reshape(ttnn.permute(o,(1,0,2)),(N,nh*hd))
    g=lin(an,A+'attention.linear_g.weight'); o=ttnn.multiply(o,g,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    attn=lin(o,A+'attention.linear_o.weight')
    gate=lin(s_t,A+'linear_a_last.weight',A+'linear_a_last.bias'); attn=ttnn.multiply(attn,gate,input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    ao=ttnn.add(attn,a_t)
    an2=adaln(ao,s_t,C+'adaln.')
    b1=lin(an2,C+'linear_nobias_a1.weight'); b2=lin(an2,C+'linear_nobias_a2.weight')
    bb=ttnn.multiply(b2,b1,input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
    out=lin(bb,C+'linear_nobias_b.weight'); cg=lin(s_t,C+'linear_s.weight',C+'linear_s.bias')
    a_t=ttnn.add(ttnn.multiply(out,cg,input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]),ao)
res=torch.Tensor(ttnn.to_torch(a_t)).float().reshape(N,768)
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('ttnn fp32 24-block DiT vs golden PCC %.5f'%pcc(res,gold),flush=True)
