import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from tt_bio.tenstorrent import get_device, CORE_GRID_MAIN as CORE
from tt_bio.protenix import AtomTransformer
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
P='module.diffusion_module.atom_attention_encoder.'
g=lambda k: ck[P+k]
dg=pickle.load(open('/home/ttuser/protenix_atomenc_pre.pkl','rb'))
pin=dg['in']; kw=dg['kwargs']
a2t=pin[0].long()            # (275,)
mask_trunked=pin[8]['mask_trunked'].float()   # (9,32,128)
r_l=kw['r_l'].float()[0]; s=kw['s'].float()[0]; c_l=kw['c_l'].float(); p_lm=kw['p_lm'].float()[0]  # (9,32,128,16)
a_gold=dg['out'][0].float()  # (1,38,768)
N=c_l.shape[0]; NT=int(a2t.max())+1; NQ,NK,PADL=32,128,48; NP=((N+NQ-1)//NQ)*NQ; nb=NP//NQ
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
lin=lambda x,w: ttnn.linear(x,T(w.t().contiguous()),compute_kernel_config=ckc,core_grid=CORE)
# token->atom broadcast matrix (275,38): atom a -> its token
S=torch.zeros(N,NT); S[torch.arange(N),a2t]=1.0
# c_l_aug = c_l + S @ linear_s(LN(s))
s_proj=lin(ttnn.layer_norm(T(s),weight=T(g('layernorm_s.weight')),epsilon=1e-5,compute_kernel_config=ckc), g('linear_no_bias_s.weight'))  # (38,128)
s_atom=ttnn.matmul(T(S), s_proj, compute_kernel_config=ckc, core_grid=CORE)  # (275,128)
c_la=ttnn.add(T(c_l), s_atom)
q_l=ttnn.add(c_la, lin(T(r_l), g('linear_no_bias_r.weight')))  # (275,128)
# p_lm augment with c_la windows (reuse encoder windowing inline)
def win_q(x):
    x=ttnn.to_layout(ttnn.reshape(x,(1,N,128)),ttnn.ROW_MAJOR_LAYOUT); x=ttnn.pad(x,[[0,0],[0,NP-N],[0,0]],0.0)
    return ttnn.to_layout(ttnn.reshape(x,(nb,NQ,128)),ttnn.TILE_LAYOUT)
def win_kv(x):
    x=ttnn.to_layout(ttnn.reshape(x,(1,N,128)),ttnn.ROW_MAJOR_LAYOUT); Lp=PADL+NP+NK; x=ttnn.pad(x,[[0,0],[PADL,Lp-PADL-N],[0,0]],0.0)
    bl=[ttnn.slice(x,[0,i*NQ,0],[1,i*NQ+NK,128]) for i in range(nb)]
    return ttnn.to_layout(ttnn.reshape(ttnn.concat(bl,0),(nb,NK,128)),ttnn.TILE_LAYOUT)
clq=ttnn.relu(win_q(c_la)); clk=ttnn.relu(win_kv(c_la))
cl=ttnn.unsqueeze(lin(clq,g('linear_no_bias_cl.weight')),2); cm=ttnn.unsqueeze(lin(clk,g('linear_no_bias_cm.weight')),1)
p=ttnn.add(ttnn.add(T(p_lm),cl),cm)
m=lin(ttnn.relu(p),'small_mlp.1.weight') if False else lin(ttnn.relu(p),g('small_mlp.1.weight'))
m=lin(ttnn.relu(m),g('small_mlp.3.weight')); m=lin(ttnn.relu(m),g('small_mlp.5.weight'))
p=ttnn.add(p,m)
# atom transformer (reuse validated)
atx=AtomTransformer(3, {k[len(P+'atom_transformer.'):]:v for k,v in ck.items() if k.startswith(P+'atom_transformer.')}, ckc)
q_out=atx(ttnn.reshape(q_l,(1,N,128)), ttnn.reshape(c_la,(1,N,128)), p, mask_trunked)
q=ttnn.relu(lin(q_out,g('linear_no_bias_q.weight')))  # (1,N,768)
q=ttnn.reshape(q,(N,768))
Mmat=torch.zeros(NT,N); 
for a in range(N): Mmat[a2t[a],a]=1.0
Mmat=Mmat/(Mmat.sum(-1,keepdim=True)+1e-6)
a=ttnn.matmul(T(Mmat),q,compute_kernel_config=ckc,core_grid=CORE)
out=torch.Tensor(ttnn.to_torch(a)).float()[:NT].reshape(a_gold.shape)
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('atom encoder(has_coords) a PCC %.5f'%pcc(out,a_gold),flush=True)
