import os
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
import pickle, torch, ttnn, sys
sys.path.insert(0,'/home/ttuser/tt-boltz2')
from tt_bio.tenstorrent import get_device, AdaLN
from tt_bio.protenix import remap_adaln
g=pickle.load(open('/home/ttuser/protenix_atomtx_gold.pkl','rb'))
W=g['weights']; mt=g['mask_trunked'].float()
N=275; nq,nk,nb,H,dh,pad_left=32,128,9,4,32,48; NP=nb*nq  # 288
dev=get_device()
ck=ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
def T(x,dt=ttnn.bfloat16): return ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=dt)
def lin(x,wkey,bkey=None):
    w=T(W[wkey].t().contiguous()); b=T(W[bkey]) if bkey else None
    return ttnn.linear(x,w,bias=b,compute_kernel_config=ck,core_grid=CORE,dtype=ttnn.bfloat16)
from tt_bio.tenstorrent import CORE_GRID_MAIN as CORE
# host-side additive mask (pair bias added on device; pad mask precomputed here as bias add)
padmask = torch.where(mt<0.5, torch.full_like(mt,-1e9), torch.zeros_like(mt))  # (9,32,128)

def block(a_tt, s_tt, p_tt, b):
    P=f'diffusion_transformer.blocks.{b}.'; apb=P+'attention_pair_bias.'
    ada_a=AdaLN(False, remap_adaln({k[len(apb+'layernorm_a.'):]:v for k,v in W.items() if k.startswith(apb+'layernorm_a.')}), ck)
    ada_kv=AdaLN(False, remap_adaln({k[len(apb+'layernorm_kv.'):]:v for k,v in W.items() if k.startswith(apb+'layernorm_kv.')}), ck)
    q_norm=ada_a(a_tt, s_tt)            # (1,275,128)
    kv_norm=ada_kv(q_norm, s_tt)
    Q=lin(q_norm, apb+'attention.linear_q.weight', apb+'attention.linear_q.bias')
    K=lin(kv_norm, apb+'attention.linear_k.weight')
    V=lin(kv_norm, apb+'attention.linear_v.weight')
    # pair bias: LN(p, weight only)->linear_nobias_z (9,32,128,16)->(9,32,128,4)
    z=ttnn.layer_norm(p_tt, weight=T(W[apb+'layernorm_z.weight']), epsilon=1e-5, compute_kernel_config=ck)
    z=lin(z, apb+'linear_nobias_z.weight')          # (9,32,128,4)
    # to torch for the windowed attention core (small); keeps parity logic explicit
    Qh=ttnn.to_torch(Q).float()[0][:N].reshape(N,H,dh)
    Kh=ttnn.to_torch(K).float()[0][:N].reshape(N,H,dh)
    Vh=ttnn.to_torch(V).float()[0][:N].reshape(N,H,dh)
    pb=ttnn.to_torch(z).float()  # (9,32,128,4)
    import torch.nn.functional as F
    qpad=F.pad(Qh,(0,0,0,0,0,NP-N)); Qb=qpad.reshape(nb,nq,H,dh).permute(0,2,1,3)
    Kp=F.pad(Kh,(0,0,0,0,pad_left,NP+nk)); Vp=F.pad(Vh,(0,0,0,0,pad_left,NP+nk))
    Kb=torch.stack([Kp[i*nq:i*nq+nk] for i in range(nb)],0).permute(0,2,1,3)
    Vb=torch.stack([Vp[i*nq:i*nq+nk] for i in range(nb)],0).permute(0,2,1,3)
    scores=torch.einsum('bhid,bhjd->bhij',Qb,Kb)/(dh**0.5)+pb.permute(0,3,1,2)+padmask.unsqueeze(1)
    attn=torch.softmax(scores,-1)
    o=torch.einsum('bhij,bhjd->bhid',attn,Vb).permute(0,2,1,3).reshape(NP,H,dh)[:N]
    o_tt=T(o.reshape(N,H*dh).unsqueeze(0))
    g_=ttnn.linear(q_norm, T(W[apb+'attention.linear_g.weight'].t().contiguous()), compute_kernel_config=ck, core_grid=CORE)
    o_tt=ttnn.multiply(o_tt, g_, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    attn_out=lin(o_tt, apb+'attention.linear_o.weight')
    gate=ttnn.linear(s_tt, T(W[apb+'linear_a_last.weight'].t().contiguous()), bias=T(W[apb+'linear_a_last.bias']), compute_kernel_config=ck, core_grid=CORE)
    attn_out=ttnn.multiply(attn_out, gate, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    a1=ttnn.add(attn_out, a_tt)
    ctb=P+'conditioned_transition_block.'
    ada_c=AdaLN(False, remap_adaln({k[len(ctb+'adaln.'):]:v for k,v in W.items() if k.startswith(ctb+'adaln.')}), ck)
    an=ada_c(a1, s_tt)
    b1=ttnn.linear(an, T(W[ctb+'linear_nobias_a1.weight'].t().contiguous()), compute_kernel_config=ck, core_grid=CORE, activation='silu')
    b2=ttnn.linear(an, T(W[ctb+'linear_nobias_a2.weight'].t().contiguous()), compute_kernel_config=ck, core_grid=CORE)
    bb=ttnn.multiply(b1,b2)
    out=lin(bb, ctb+'linear_nobias_b.weight')
    cg=ttnn.linear(s_tt, T(W[ctb+'linear_s.weight'].t().contiguous()), bias=T(W[ctb+'linear_s.bias']), compute_kernel_config=ck, core_grid=CORE)
    out=ttnn.multiply(out, cg, input_tensor_b_activations=[ttnn.UnaryOpType.SIGMOID])
    return ttnn.add(out, a1)

a=T(g['q'].unsqueeze(0)); s=T(g['c'].unsqueeze(0)); p=T(g['p'])
x=a
for b in range(3): x=block(x,s,p,b)
out=ttnn.to_torch(x).float()[0][:N]; gold=g['golden_qout'].float()
def pcc(u,v):
    u=u.flatten().double(); v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('ttnn atom-tx PCC %.6f  maxerr %.3e'%(pcc(out,gold),(out-gold).abs().max()))
