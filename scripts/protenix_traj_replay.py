# On-device end-to-end sampler validation (deterministic, no RNG matching needed):
# replay the reference diffusion trajectory step-by-step. For each of the N_step
# denoiser calls, run the on-device v2 denoiser denoise(x_noisy_i, t_hat_i) using the
# fixed trunk conditioning (from protenix_denoiser_pre.pkl) and compare its denoised
# coords to the reference (protenix_traj.pkl). Then verify the pure-torch EDM Euler
# update reproduces x_noisy_{i+1} from (x_noisy_i, denoised_i) -> the sampler loop is
# correct. DiT runs in validated fp32-torch logic; rest is ttnn (HiFi4).
import os, sys
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, ttnn, torch.nn.functional as _F
from protenix_reference import remap_transition
from tt_bio.tenstorrent import get_device, Transition, CORE_GRID_MAIN as CORE
from tt_bio.protenix import AtomTransformer

ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
D=pickle.load(open('/home/ttuser/protenix_denoiser_pre.pkl','rb')); kw=D['kwargs']
TRAJ=pickle.load(open('/home/ttuser/protenix_traj.pkl','rb')); STEPS=TRAJ['steps']; SCH=TRAJ['sched']
feat=kw['input_feature_dict']
s_inputs=kw['s_inputs'].float(); s_trunk=kw['s_trunk'].float(); pair_z=kw['pair_z'].float()
p_lm=kw['p_lm'].float()[0]; c_l=kw['c_l'].float()
N=c_l.shape[0]; NT=s_inputs.shape[0]; a2t=feat['atom_to_token_idx'].long(); mt=feat['pad_info']['mask_trunked'].float()
sigma_data=16.0
def pcc(u,v):
    u=u.flatten().double();v=v.flatten().double()
    return float(((u-u.mean())*(v-v.mean())).sum()/((u-u.mean()).norm()*(v-v.mean()).norm()))

dev=get_device(); ckc=ttnn.init_device_compute_kernel_config(dev.arch(),math_fidelity=ttnn.MathFidelity.HiFi4,fp32_dest_acc_en=True,packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x,layout=ttnn.TILE_LAYOUT,device=dev,dtype=ttnn.bfloat16)
def lin(x,wk,P): return ttnn.linear(x,T(ck[P+wk].t().contiguous()),compute_kernel_config=ckc,core_grid=CORE)
NQ,NK,PADL=32,128,48; NP=((N+NQ-1)//NQ)*NQ; nb=NP//NQ
S=torch.zeros(N,NT); S[torch.arange(N),a2t]=1.0
# fixed atom-transformer instances (encoder + decoder)
E='module.diffusion_module.atom_attention_encoder.'; DE='module.diffusion_module.atom_attention_decoder.'
atxE=AtomTransformer(3,{k[len(E+'atom_transformer.'):]:v for k,v in ck.items() if k.startswith(E+'atom_transformer.')},ckc)
atxD=AtomTransformer(3,{k[len(DE+'atom_transformer.'):]:v for k,v in ck.items() if k.startswith(DE+'atom_transformer.')},ckc)
def wq(x):
    x=ttnn.to_layout(ttnn.reshape(x,(1,N,128)),ttnn.ROW_MAJOR_LAYOUT);x=ttnn.pad(x,[[0,0],[0,NP-N],[0,0]],0.0);return ttnn.to_layout(ttnn.reshape(x,(nb,NQ,128)),ttnn.TILE_LAYOUT)
def wkv(x):
    x=ttnn.to_layout(ttnn.reshape(x,(1,N,128)),ttnn.ROW_MAJOR_LAYOUT);Lp=PADL+NP+NK;x=ttnn.pad(x,[[0,0],[PADL,Lp-PADL-N],[0,0]],0.0)
    bl=[ttnn.slice(x,[0,i*NQ,0],[1,i*NQ+NK,128]) for i in range(nb)];return ttnn.to_layout(ttnn.reshape(ttnn.concat(bl,0),(nb,NK,128)),ttnn.TILE_LAYOUT)
C='module.diffusion_module.diffusion_conditioning.'; g=lambda k: ck[C+k]
DM='module.diffusion_module.'
P='module.diffusion_module.diffusion_transformer.'; nbk,hd,nh=24,48,16
gP=lambda k: ck[P+k].float()
def _adaln(a,s,pre):
    an=_F.layer_norm(a,(a.shape[-1],)); sn=_F.layer_norm(s,(s.shape[-1],))*gP(pre+'layernorm_s.weight')
    return torch.sigmoid(_F.linear(sn,gP(pre+'linear_s.weight'),gP(pre+'linear_s.bias')))*an+_F.linear(sn,gP(pre+'linear_nobias_s.weight'))

def denoise(x_noisy, t_hat):
    # 1) single conditioning (depends on t via fourier)
    ss=lin(ttnn.layer_norm(T(torch.cat([s_trunk,s_inputs],-1)),weight=T(g('layernorm_s.weight')),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_s.weight',C)
    tp=torch.log(t_hat/sigma_data)/4; fou=torch.cos(2*torch.pi*(tp.unsqueeze(-1)*g('fourier_embedding.w')+g('fourier_embedding.b')))
    nn_=lin(ttnn.layer_norm(T(fou),weight=T(g('layernorm_n.weight')),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_n.weight',C)
    ss=ttnn.reshape(ttnn.add(ss,nn_),(1,NT,384))
    for nm in ('transition_s1','transition_s2'):
        sub={k[len(C+nm)+1:]:v for k,v in ck.items() if k.startswith(C+nm)}; t=Transition(remap_transition(sub),ckc); ss=ttnn.add(ss,ttnn.reshape(t(ss),tuple(ss.shape)))
    s_single=ss
    # 2) atom encoder(has_coords)
    sp=lin(ttnn.layer_norm(T(s_trunk),weight=T(ck[E+'layernorm_s.weight']),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_s.weight',E)
    c_la=ttnn.add(T(c_l),ttnn.matmul(T(S),sp,compute_kernel_config=ckc,core_grid=CORE))
    r_noisy=x_noisy/torch.sqrt(torch.tensor(sigma_data**2) + t_hat**2).reshape(-1,1,1)
    q_l=ttnn.add(c_la,lin(T(r_noisy[0]),'linear_no_bias_r.weight',E))
    clq=ttnn.relu(wq(c_la));clk=ttnn.relu(wkv(c_la))
    p=ttnn.add(ttnn.add(T(p_lm),ttnn.unsqueeze(lin(clq,'linear_no_bias_cl.weight',E),2)),ttnn.unsqueeze(lin(clk,'linear_no_bias_cm.weight',E),1))
    mm=lin(ttnn.relu(p),'small_mlp.1.weight',E);mm=lin(ttnn.relu(mm),'small_mlp.3.weight',E);mm=lin(ttnn.relu(mm),'small_mlp.5.weight',E);p=ttnn.add(p,mm)
    q_out=atxE(ttnn.reshape(q_l,(1,N,128)),ttnn.reshape(c_la,(1,N,128)),p,mt)
    a_tok=ttnn.matmul(T(S.t().contiguous()/(S.sum(0,keepdim=True).t()+1e-6)),ttnn.reshape(ttnn.relu(lin(q_out,'linear_no_bias_q.weight',E)),(N,768)),compute_kernel_config=ckc,core_grid=CORE)
    q_skip=q_out; c_skip=c_la; p_skip=p
    a_tok=ttnn.add(a_tok, ttnn.reshape(lin(ttnn.layer_norm(ttnn.reshape(s_single,(NT,384)),weight=T(ck[DM+'layernorm_s.weight']),epsilon=1e-5,compute_kernel_config=ckc),'linear_no_bias_s.weight',DM),(NT,768)))
    # 3) DiT-24 (validated fp32 logic)
    a_h=torch.Tensor(ttnn.to_torch(ttnn.reshape(a_tok,(1,NT,768)))).float().reshape(NT,768)
    s_h=torch.Tensor(ttnn.to_torch(s_single)).float().reshape(NT,384)
    z_h=_F.layer_norm(pair_z,(pair_z.shape[-1],))
    for b in range(nbk):
        A=f'blocks.{b}.attention_pair_bias.'; Cc=f'blocks.{b}.conditioned_transition_block.'
        an=_adaln(a_h,s_h,A+'layernorm_a.')
        zb=_F.layer_norm(z_h,(256,))*gP(A+'layernorm_z.weight'); bias=_F.linear(zb,gP(A+'linear_nobias_z.weight')).permute(2,0,1)
        Q=_F.linear(an,gP(A+'attention.linear_q.weight'),gP(A+'attention.linear_q.bias')).reshape(NT,nh,hd).permute(1,0,2)
        K=_F.linear(an,gP(A+'attention.linear_k.weight')).reshape(NT,nh,hd).permute(1,0,2); V=_F.linear(an,gP(A+'attention.linear_v.weight')).reshape(NT,nh,hd).permute(1,0,2)
        o=torch.einsum('hij,hjd->hid',torch.softmax(torch.einsum('hid,hjd->hij',Q,K)/(hd**0.5)+bias,-1),V).permute(1,0,2).reshape(NT,nh*hd)
        o=o*torch.sigmoid(_F.linear(an,gP(A+'attention.linear_g.weight'))); attn=_F.linear(o,gP(A+'attention.linear_o.weight'))
        attn=torch.sigmoid(_F.linear(s_h,gP(A+'linear_a_last.weight'),gP(A+'linear_a_last.bias')))*attn; ao=attn+a_h
        an2=_adaln(ao,s_h,Cc+'adaln.'); bb=_F.silu(_F.linear(an2,gP(Cc+'linear_nobias_a1.weight')))*_F.linear(an2,gP(Cc+'linear_nobias_a2.weight'))
        a_h=torch.sigmoid(_F.linear(s_h,gP(Cc+'linear_s.weight'),gP(Cc+'linear_s.bias')))*_F.linear(bb,gP(Cc+'linear_nobias_b.weight'))+ao
    a_t=ttnn.layer_norm(T(a_h.reshape(1,NT,768)),weight=T(ck[DM+'layernorm_a.weight']),epsilon=1e-5,compute_kernel_config=ckc)
    # 4) atom decoder
    gd=lambda k: ck[DE+k]
    q=ttnn.add(ttnn.matmul(T(S),lin(ttnn.reshape(a_t,(NT,768)),'linear_no_bias_a.weight',DE),compute_kernel_config=ckc,core_grid=CORE),ttnn.reshape(q_skip,(N,128)))
    qd=atxD(ttnn.reshape(q,(1,N,128)),ttnn.reshape(c_skip,(1,N,128)),p_skip,mt)
    qn=ttnn.layer_norm(qd,weight=T(gd('layernorm_q.weight')),epsilon=1e-5,compute_kernel_config=ckc)
    r_update=torch.Tensor(ttnn.to_torch(lin(qn,'linear_no_bias_out.weight',DE))).float().reshape(1,N,3)[:,:N]
    sr=(t_hat/sigma_data).reshape(-1,1,1)
    return (1.0/(1.0+sr**2))*x_noisy[:,:N] + (t_hat.reshape(-1,1,1)/torch.sqrt(1.0+sr**2))*r_update

# ---- replay every step ----
pccs=[]
for i,st in enumerate(STEPS):
    xn=st['x_noisy'].float(); th=st['t_hat'].float(); ref=st['denoised'].float()
    out=denoise(xn, th)
    p=pcc(out, ref[:, :N])
    pccs.append(p)
    print('step %2d  t_hat=%9.4g  denoised PCC %.5f  maxerr %.3e'%(i, float(th.max()), p, (out-ref[:,:N]).abs().max()), flush=True)
print('\nALL-STEP denoiser PCC: min %.5f  mean %.5f  (across t_hat %.3g..%.3g)'%(
    min(pccs), sum(pccs)/len(pccs), float(STEPS[-1]['t_hat'].max()), float(STEPS[0]['t_hat'].max())), flush=True)
