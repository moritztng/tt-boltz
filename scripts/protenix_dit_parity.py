import os, sys, re
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from protenix_reference import remap_attention_pair_bias, remap_adaptive_layernorm
from tt_bio.tenstorrent import get_device, AdaLN, AttentionPairBias, CORE_GRID_MAIN as CORE
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
PRE='module.diffusion_module.diffusion_transformer.'
nb=1+max(int(re.search(r'blocks\.(\d+)\.',k).group(1)) for k in ck if k.startswith(PRE+'blocks.'))
dg=pickle.load(open('/home/ttuser/protenix_diffusion_gold.pkl','rb'))['dit']; kw=dg['kwargs']
a=kw['a'].float(); s=kw['s'].float(); z=kw['z'].float()           # a(1,38,768) s(1,38,384) z(1,256,38,38)
gold=dg['out'].float()
hd,nh=48,16
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
ft=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
ftt=lambda x: ttnn.from_torch(x.t(),layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
sub=lambda p:{k[len(PRE+p)+1:]:v for k,v in ck.items() if k.startswith(PRE+p)}
s2=lambda d,p:{k[len(p)+1:]:v for k,v in d.items() if k.startswith(p+'.')}
z_t=ft(z.permute(0,2,3,1).contiguous())                           # (1,38,38,256)
a_t=ft(a); s_t=ft(s)
for b in range(nb):
    apb_sd=sub(f'blocks.{b}.attention_pair_bias')
    adaln_a=AdaLN(False, remap_adaptive_layernorm(s2(apb_sd,'layernorm_a')), ckc)
    apb=AttentionPairBias(hd,nh,True,False, remap_attention_pair_bias(apb_sd), ckc)
    lalw,lalb=ftt(apb_sd['linear_a_last.weight']), ft(apb_sd['linear_a_last.bias'])
    ctb=sub(f'blocks.{b}.conditioned_transition_block')
    ctb_adaln=AdaLN(False, remap_adaptive_layernorm(s2(ctb,'adaln')), ckc)
    a1w,a2w,bw=ftt(ctb['linear_nobias_a1.weight']),ftt(ctb['linear_nobias_a2.weight']),ftt(ctb['linear_nobias_b.weight'])
    lsw,lsb=ftt(ctb['linear_s.weight']),ft(ctb['linear_s.bias'])
    an=adaln_a(a_t,s_t)
    attn=apb(an, z_t)
    gate=ttnn.linear(s_t,lalw,bias=lalb,compute_kernel_config=ckc,core_grid=CORE)
    attn=ttnn.multiply(gate,attn,input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
    attn=ttnn.add(attn, a_t)                                      # input residual
    cn=ctb_adaln(attn,s_t)
    c1=ttnn.linear(cn,a1w,compute_kernel_config=ckc,core_grid=CORE); c2=ttnn.linear(cn,a2w,compute_kernel_config=ckc,core_grid=CORE)
    cb=ttnn.multiply(c2,c1,input_tensor_b_activations=[ttnn.UnaryOpType.SILU])
    cba=ttnn.linear(cb,bw,compute_kernel_config=ckc,core_grid=CORE)
    csg=ttnn.linear(s_t,lsw,bias=lsb,compute_kernel_config=ckc,core_grid=CORE)
    ff=ttnn.multiply(csg,cba,input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID])
    a_t=ttnn.add(attn,ff)
out=torch.Tensor(ttnn.to_torch(a_t)).float().reshape(gold.shape)
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('24-block token DiT PCC %.5f'%pcc(out,gold),flush=True)
