import pickle, torch, torch.nn.functional as F
torch.set_grad_enabled(False)
g=pickle.load(open('/home/ttuser/protenix_atomtx_gold.pkl','rb'))
W=g['weights']; a=g['q'].float(); s=g['c'].float(); p=g['p'].float()  # (275,128),(275,128),(9,32,128,16)
mt=g['mask_trunked'].float()  # (9,32,128)
N=a.shape[0]; nq,nk,nb=32,128,p.shape[0]; H=4; dh=32; pad_left=48
def adaln(a,s,pre):
    a_n=F.layer_norm(a,(a.shape[-1],)); s_n=F.layer_norm(s,(s.shape[-1],))*W[pre+'layernorm_s.weight']
    return torch.sigmoid(F.linear(s_n,W[pre+'linear_s.weight'],W[pre+'linear_s.bias']))*a_n + F.linear(s_n,W[pre+'linear_nobias_s.weight'])
def block(a,s,p,b):
    P=f'diffusion_transformer.blocks.{b}.'
    apb=P+'attention_pair_bias.'
    q_norm=adaln(a,s,apb+'layernorm_a.')
    kv_norm=adaln(q_norm,s,apb+'layernorm_kv.')
    Q=F.linear(q_norm,W[apb+'attention.linear_q.weight'],W[apb+'attention.linear_q.bias'])
    K=F.linear(kv_norm,W[apb+'attention.linear_k.weight'])
    V=F.linear(kv_norm,W[apb+'attention.linear_v.weight'])
    # pair bias: LayerNorm(p, weight only) -> linear_nobias_z -> (9,32,128,4)
    z=F.layer_norm(p,(p.shape[-1],))*W[apb+'layernorm_z.weight']
    bias=F.linear(z,W[apb+'linear_nobias_z.weight'])  # (9,32,128,4)
    bias=bias.permute(3,0,1,2)  # (4,9,32,128) -> per (h,block,i,j)
    # windows
    Qh=Q.reshape(N,H,dh); Kh=K.reshape(N,H,dh); Vh=V.reshape(N,H,dh)
    qpad=F.pad(Qh,(0,0,0,0,0,nb*nq-N))  # (288,H,dh)
    Qb=qpad.reshape(nb,nq,H,dh).permute(0,2,1,3)  # (nb,H,nq,dh)
    Kp=F.pad(Kh,(0,0,0,0,pad_left, nb*nq+nk)) # generous right pad
    Vp=F.pad(Vh,(0,0,0,0,pad_left, nb*nq+nk))
    Kb=torch.stack([Kp[i*nq:i*nq+nk] for i in range(nb)],0).permute(0,2,1,3) # (nb,H,nk,dh)
    Vb=torch.stack([Vp[i*nq:i*nq+nk] for i in range(nb)],0).permute(0,2,1,3)
    scores=torch.einsum('bhid,bhjd->bhij',Qb,Kb)/(dh**0.5)  # (nb,H,nq,nk)
    pb=bias.permute(1,0,2,3)  # (nb,H,nq,nk)
    padmask=(mt<0.5).unsqueeze(1)*(-1e9)  # (nb,1,nq,nk)
    scores=scores+pb+padmask
    attn=torch.softmax(scores,-1)
    o=torch.einsum('bhij,bhjd->bhid',attn,Vb)  # (nb,H,nq,dh)
    o=o.permute(0,2,1,3).reshape(nb*nq,H,dh)[:N]  # (275,H,dh)
    g_=torch.sigmoid(F.linear(q_norm,W[apb+'attention.linear_g.weight'])).reshape(N,H,dh)
    o=(o*g_).reshape(N,H*dh)
    attn_out=F.linear(o,W[apb+'attention.linear_o.weight'])
    attn_out=torch.sigmoid(F.linear(s,W[apb+'linear_a_last.weight'],W[apb+'linear_a_last.bias']))*attn_out
    a1=attn_out+a
    # ConditionedTransitionBlock
    ctb=P+'conditioned_transition_block.'
    an=adaln(a1,s,ctb+'adaln.')
    bb=F.silu(F.linear(an,W[ctb+'linear_nobias_a1.weight']))*F.linear(an,W[ctb+'linear_nobias_a2.weight'])
    out=torch.sigmoid(F.linear(s,W[ctb+'linear_s.weight'],W[ctb+'linear_s.bias']))*F.linear(bb,W[ctb+'linear_nobias_b.weight'])
    return out+a1
x=a
for b in range(nb if False else 3): x=block(x,s,p,b)
gold=g['golden_qout'].float()
def pcc(u,v):
    u=u.flatten().double(); v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('torch atom-tx PCC %.6f  maxerr %.3e'%(pcc(x,gold),(x-gold).abs().max()))
