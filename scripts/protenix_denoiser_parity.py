import os, sys, re
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn
from protenix_reference import remap_transition, remap_attention_pair_bias, remap_adaptive_layernorm
from tt_bio.tenstorrent import get_device, Transition, AdaLN, AttentionPairBias, CORE_GRID_MAIN as CORE
from tt_bio.protenix import AtomTransformer
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb')); kw=D['kwargs']
feat=kw['input_feature_dict']; x_noisy=kw['x_noisy'].float(); t_hat=kw['t_hat_noise_level'].float()
s_inputs=kw['s_inputs'].float(); s_trunk=kw['s_trunk'].float(); pair_z=kw['pair_z'].float()
p_lm=kw['p_lm'].float()[0]; c_l=kw['c_l'].float(); coords_g=D['out'].float()
DMS=pickle.load(open('/home/ttuser/protenix_dm_stages.pkl','rb'))
def _p(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
N=c_l.shape[0]; NT=s_inputs.shape[0]; a2t=feat['atom_to_token_idx'].long(); mt=feat['pad_info']['mask_trunked'].float()
dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
def lin(x,wk,P): return ttnn.linear(x,T(ck[P+wk].t().contiguous()),compute_kernel_config=ckc,core_grid=CORE)
NQ,NK,PADL=32,128,48; NP=((N+NQ-1)//NQ)*NQ; nb=NP//NQ
S=torch.zeros(N,NT); S[torch.arange(N),a2t]=1.0
# 1) cond single
C='module.diffusion_module.diffusion_conditioning.'; g=lambda k: ck[C+k]
ss=lin(ttnn.layer_norm(T(torch.cat([s_trunk,s_inputs],-1)),weight=T(g('layernorm_s.weight')),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_s.weight',C)
tp=torch.log(t_hat/16.0)/4; fou=torch.cos(2*torch.pi*(tp.unsqueeze(-1)*g('fourier_embedding.w')+g('fourier_embedding.b')))
nn_=lin(ttnn.layer_norm(T(fou),weight=T(g('layernorm_n.weight')),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_n.weight',C)
ss=ttnn.reshape(ttnn.add(ss,nn_),(1,NT,384))
for nm in ('transition_s1','transition_s2'):
    sub={k[len(C+nm)+1:]:v for k,v in ck.items() if k.startswith(C+nm)}; t=Transition(remap_transition(sub),ckc); ss=ttnn.add(ss,ttnn.reshape(t(ss),tuple(ss.shape)))
s_single=ss
print('s_single PCC %.4f'%_p(torch.Tensor(ttnn.to_torch(s_single)).float().reshape(NT,384), DMS['s_single'].float().reshape(NT,384)),flush=True)
# 2) atom encoder(has_coords) -> a(768), q_skip,c_skip,p_skip
E='module.diffusion_module.atom_attention_encoder.'; ge=lambda k: ck[E+k]
sp=lin(ttnn.layer_norm(T(s_trunk),weight=T(ge('layernorm_s.weight')),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_s.weight',E)
c_la=ttnn.add(T(c_l),ttnn.matmul(T(S),sp,compute_kernel_config=ckc,core_grid=CORE))
sigma=16.0
r_noisy=x_noisy/torch.sqrt(sigma**2 + t_hat**2).reshape(-1,1,1)
q_l=ttnn.add(c_la,lin(T(r_noisy[0]),'linear_no_bias_r.weight',E))
def wq(x):
    x=ttnn.to_layout(ttnn.reshape(x,(1,N,128)),ttnn.ROW_MAJOR_LAYOUT);x=ttnn.pad(x,[[0,0],[0,NP-N],[0,0]],0.0);return ttnn.to_layout(ttnn.reshape(x,(nb,NQ,128)),ttnn.TILE_LAYOUT)
def wkv(x):
    x=ttnn.to_layout(ttnn.reshape(x,(1,N,128)),ttnn.ROW_MAJOR_LAYOUT);Lp=PADL+NP+NK;x=ttnn.pad(x,[[0,0],[PADL,Lp-PADL-N],[0,0]],0.0)
    bl=[ttnn.slice(x,[0,i*NQ,0],[1,i*NQ+NK,128]) for i in range(nb)];return ttnn.to_layout(ttnn.reshape(ttnn.concat(bl,0),(nb,NK,128)),ttnn.TILE_LAYOUT)
clq=ttnn.relu(wq(c_la));clk=ttnn.relu(wkv(c_la))
p=ttnn.add(ttnn.add(T(p_lm),ttnn.unsqueeze(lin(clq,'linear_no_bias_cl.weight',E),2)),ttnn.unsqueeze(lin(clk,'linear_no_bias_cm.weight',E),1))
m=lin(ttnn.relu(p),'small_mlp.1.weight',E);m=lin(ttnn.relu(m),'small_mlp.3.weight',E);m=lin(ttnn.relu(m),'small_mlp.5.weight',E);p=ttnn.add(p,m)
atxE=AtomTransformer(3,{k[len(E+'atom_transformer.'):]:v for k,v in ck.items() if k.startswith(E+'atom_transformer.')},ckc)
q_out=atxE(ttnn.reshape(q_l,(1,N,128)),ttnn.reshape(c_la,(1,N,128)),p,mt)
a_tok=ttnn.matmul(T(S.t().contiguous()/(S.sum(0,keepdim=True).t()+1e-6)),ttnn.reshape(ttnn.relu(lin(q_out,'linear_no_bias_q.weight',E)),(N,768)),compute_kernel_config=ckc,core_grid=CORE)  # (NT,768)
q_skip=q_out; c_skip=c_la; p_skip=p
print('enc_a PCC %.4f'%_p(torch.Tensor(ttnn.to_torch(a_tok)).float().reshape(NT,768), DMS['enc_a'].float().reshape(NT,768)),flush=True)
DM='module.diffusion_module.'
a_tok=ttnn.add(a_tok, ttnn.reshape(lin(ttnn.layer_norm(ttnn.reshape(s_single,(NT,384)),weight=T(ck[DM+'layernorm_s.weight']),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_s.weight',DM),(NT,768)))
# 3) DiT-24
P='module.diffusion_module.diffusion_transformer.'; nbk=24; hd,nh=48,16
a_t=ttnn.reshape(a_tok,(1,NT,768))
print('dit_in PCC %.4f'%_p(torch.Tensor(ttnn.to_torch(a_t)).float().reshape(NT,768), DMS['dit_in'].float().reshape(NT,768)),flush=True)
pz_n=torch.nn.functional.layer_norm(pair_z,(pair_z.shape[-1],))
z_t=T(pz_n.unsqueeze(0)); s_t=s_single
sub2=lambda pp:{k[len(P+pp)+1:]:v for k,v in ck.items() if k.startswith(P+pp)}
s2=lambda d,pp:{k[len(pp)+1:]:v for k,v in d.items() if k.startswith(pp+'.')}
ftt=lambda x: ttnn.from_torch(x.t(),layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16); ft=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
for b in range(nbk):
    asd=sub2(f'blocks.{b}.attention_pair_bias'); adA=AdaLN(False,remap_adaptive_layernorm(s2(asd,'layernorm_a')),ckc); apb=AttentionPairBias(hd,nh,True,False,remap_attention_pair_bias(asd),ckc)
    lalw,lalb=ftt(asd['linear_a_last.weight']),ft(asd['linear_a_last.bias']); ctb=sub2(f'blocks.{b}.conditioned_transition_block'); adC=AdaLN(False,remap_adaptive_layernorm(s2(ctb,'adaln')),ckc)
    a1w,a2w,bw=ftt(ctb['linear_nobias_a1.weight']),ftt(ctb['linear_nobias_a2.weight']),ftt(ctb['linear_nobias_b.weight']); lsw,lsb=ftt(ctb['linear_s.weight']),ft(ctb['linear_s.bias'])
    an=adA(a_t,s_t); attn=apb(an,z_t); gate=ttnn.linear(s_t,lalw,bias=lalb,compute_kernel_config=ckc,core_grid=CORE)
    attn=ttnn.multiply(gate,attn,input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]); attn=ttnn.add(attn,a_t)
    cn=adC(attn,s_t); c1=ttnn.linear(cn,a1w,compute_kernel_config=ckc,core_grid=CORE);c2=ttnn.linear(cn,a2w,compute_kernel_config=ckc,core_grid=CORE)
    cb=ttnn.multiply(c2,c1,input_tensor_b_activations=[ttnn.UnaryOpType.SILU]);cba=ttnn.linear(cb,bw,compute_kernel_config=ckc,core_grid=CORE)
    csg=ttnn.linear(s_t,lsw,bias=lsb,compute_kernel_config=ckc,core_grid=CORE);ff=ttnn.multiply(csg,cba,input_tensor_a_activations=[ttnn.UnaryOpType.SIGMOID]);a_t=ttnn.add(attn,ff)
# 4) atom decoder
print('dit_out PCC %.4f'%_p(torch.Tensor(ttnn.to_torch(a_t)).float().reshape(NT,768), DMS['dit_out'].float().reshape(NT,768)),flush=True)
a_t=ttnn.layer_norm(a_t,weight=T(ck['module.diffusion_module.layernorm_a.weight']),epsilon=1e-5,compute_kernel_config=ckc)
DE='module.diffusion_module.atom_attention_decoder.'; gd=lambda k: ck[DE+k]
q=ttnn.add(ttnn.matmul(T(S),lin(ttnn.reshape(a_t,(NT,768)),'linear_no_bias_a.weight',DE),compute_kernel_config=ckc,core_grid=CORE),ttnn.reshape(q_skip,(N,128)))
atxD=AtomTransformer(3,{k[len(DE+'atom_transformer.'):]:v for k,v in ck.items() if k.startswith(DE+'atom_transformer.')},ckc)
qd=atxD(ttnn.reshape(q,(1,N,128)),ttnn.reshape(c_skip,(1,N,128)),p_skip,mt)
qn=ttnn.layer_norm(qd,weight=T(gd('layernorm_q.weight')),epsilon=1e-5,compute_kernel_config=ckc)
r_update=torch.Tensor(ttnn.to_torch(lin(qn,'linear_no_bias_out.weight',DE))).float().reshape(1,N,3)[:,:N]
sr=(t_hat/sigma).reshape(-1,1,1)
coords=(1.0/(1.0+sr**2))*x_noisy[:,:N] + (t_hat.reshape(-1,1,1)/torch.sqrt(1.0+sr**2))*r_update
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))
print('DENOISER-as-a-unit coords PCC %.5f  maxerr %.3e'%(pcc(coords,coords_g),(coords-coords_g).abs().max()),flush=True)
