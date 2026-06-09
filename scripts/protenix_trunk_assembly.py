import os, sys, re
os.environ.setdefault('TT_VISIBLE_DEVICES','0'); os.environ.setdefault('TT_LOGGER_LEVEL','FATAL')
sys.path.insert(0,'/home/ttuser/tt-boltz2'); sys.path.insert(0,'/home/ttuser/tt-boltz2/tests')
import pickle, torch, torch.nn.functional as F, ttnn
from protenix_reference import (remap_pairformer_block, remap_msa_pair_stack, remap_outer_product_mean,
                                remap_pair_weighted_averaging, remap_transition)
from tt_bio.tenstorrent import (get_device, Pairformer, PairformerLayer, OuterProductMean,
                                PairWeightedAveraging, Transition)
from tt_bio.protenix import TrunkInput
ck=torch.load('/home/ttuser/protenix_ckpt/protenix-v2.pt',map_location='cpu',weights_only=True); ck=ck.get('model',ck)
d=pickle.load(open('/home/ttuser/protenix_ref_out.pkl','rb'))
feat=d['intermediates']['template_embedder']['in'][0]
s_inputs=d['intermediates']['input_embedder']['out'].float()           # golden (38,449)
relp=d['intermediates'].get('input_embedder')  # need relp/token_bonds: pull from trunkin gold
tg=pickle.load(open('/home/ttuser/protenix_trunkin_gold.pkl','rb'))
relp=tg['relp']; token_bonds=tg['token_bonds']
dk=d['intermediates']['diffusion_module']['kwargs']; s_gold=dk['s_trunk'].float(); z_gold=dk['pair_z'].float()
N=38
dev=get_device()
ckc=ttnn.init_device_compute_kernel_config(dev.arch(), math_fidelity=ttnn.MathFidelity.HiFi4, fp32_dest_acc_en=True, packer_l1_acc=True)
T=lambda x: ttnn.from_torch(x, layout=ttnn.TILE_LAYOUT, device=dev, dtype=ttnn.bfloat16)
mm=lambda P: {k[len('module.'+P)+1:]:v for k,v in ck.items() if k.startswith('module.'+P)}
g=lambda k: ck['module.'+k]
def lin(x,w): return ttnn.linear(x, T(w.t().contiguous()), compute_kernel_config=ckc, core_grid=ttnn.CoreGrid(y=8,x=8))
def ln(x,wk,bk=None): return ttnn.layer_norm(x, weight=T(g(wk)), bias=(T(g(bk)) if bk else None), epsilon=1e-5, compute_kernel_config=ckc)

# trunk init (TrunkInput, validated)
ti_keys=['linear_no_bias_sinit','linear_no_bias_zinit1','linear_no_bias_zinit2','linear_no_bias_token_bond','relative_position_encoding']
ti_sd={k:v for k,v in {kk[len('module.'):]:vv for kk,vv in ck.items() if kk.startswith('module.')}.items() if any(k.startswith(p) for p in ti_keys)}
ti=TrunkInput(ti_sd, ckc)
s_init, z_init = ti(T(s_inputs), T(relp), T(token_bonds.unsqueeze(-1)))   # (38,384),(38,38,256)

# --- instantiate modules ONCE ---
# pairformer 48
nb_pf=1+max(int(re.search(r'pairformer_stack\.blocks\.(\d+)\.',k).group(1)) for k in ck if 'pairformer_stack.blocks.' in k)
comb={}
for i in range(nb_pf):
    for k,v in remap_pairformer_block(mm(f'pairformer_stack.blocks.{i}')).items(): comb[f'layers.{i}.{k}']=v
PF=Pairformer(nb_pf,32,8,384//16,16,True,comb,ckc)
# template (2 pair-only blocks), build feature concat once
asym=feat['asym_id']; mc=(asym[:,None]==asym[None,:]).float(); pm=torch.ones(N,N)
nt=feat['template_aatype'].shape[0]
te_at=[]
for t in range(nt):
    dg=feat['template_distogram'][t]*mc[...,None]*pm[...,None]; pb=(feat['template_pseudo_beta_mask'][t]*mc*pm).unsqueeze(-1)
    aa=F.one_hot(feat['template_aatype'][t].long(),32).float(); aai=aa[None].expand(N,N,32); aaj=aa[:,None].expand(N,N,32)
    uv=feat['template_unit_vector'][t]*mc[...,None]*pm[...,None]; bb=(feat['template_backbone_frame_mask'][t]*mc*pm).unsqueeze(-1)
    te_at.append(torch.cat([dg,pb,aai,aaj,uv,bb],-1))
TPL=[[PairformerLayer(32,2,None,None,False, remap_msa_pair_stack(mm(f'template_embedder.pairformer_stack.blocks.{b}')), ckc) for b in range(2)]]
# msa components per block
msa=F.one_hot(feat['msa'].long(),32).float()
ms=torch.cat([msa, feat['has_deletion'].unsqueeze(-1), feat['deletion_value'].unsqueeze(-1)],-1).unsqueeze(0)
nb_msa=4
MSA=[]
for i in range(nb_msa):
    P=f'msa_module.blocks.{i}.'
    opm=OuterProductMean(remap_outer_product_mean(mm(P+'outer_product_mean_msa')), ckc)
    pl=PairformerLayer(32,8,None,None,False, remap_msa_pair_stack(mm(P+'pair_stack')), ckc)
    has=any(k.startswith('module.'+P+'msa_stack.') for k in ck)
    pwa=tm=None
    if has:
        pwa=PairWeightedAveraging(8,8,remap_pair_weighted_averaging(mm(P+'msa_stack.msa_pair_weighted_averaging')), ckc)
        tm=Transition(remap_transition(mm(P+'msa_stack.transition_m')), ckc)
    MSA.append((opm,pwa,tm,pl))

def template(z):  # z (1,38,38,256) -> z-update
    zn=ln(z,'template_embedder.layernorm_z.weight','template_embedder.layernorm_z.bias')
    u=None
    for t in range(nt):
        v=ttnn.add(lin(T(te_at[t].unsqueeze(0)), g('template_embedder.linear_no_bias_a.weight')),
                   lin(zn, g('template_embedder.linear_no_bias_z.weight')))
        for pl in TPL[0]: v=pl(None,v)[1]
        v=ln(v,'template_embedder.layernorm_v.weight','template_embedder.layernorm_v.bias')
        u=v if u is None else ttnn.add(u,v)
    u=ttnn.multiply(u,1.0/(1e-7+nt))
    return lin(ttnn.relu(u), g('template_embedder.linear_no_bias_u.weight'))

def msa_mod(z, m):
    for (opm,pwa,tm,pl) in MSA:
        z=ttnn.add(z, opm(m,None,None))
        if pwa is not None:
            m=ttnn.add(m, ttnn.reshape(pwa(m, ttnn.clone(z)), tuple(m.shape)))
            m=ttnn.add(m, ttnn.reshape(tm(m), tuple(m.shape)))
        z=pl(None,z)[1]
    return z

m_feat=ttnn.add(lin(T(ms), g('msa_module.linear_no_bias_m.weight')), lin(T(s_inputs), g('msa_module.linear_no_bias_s.weight')))
z=ttnn.mul(z_init,0.0); s=ttnn.mul(s_init,0.0)
z3=ttnn.reshape(z,(1,N,N,256))
for cyc in range(10):
    zc=lin(ln(z3,'layernorm_z_cycle.weight','layernorm_z_cycle.bias'), g('linear_no_bias_z_cycle.weight'))
    z3=ttnn.add(ttnn.reshape(z_init,(1,N,N,256)), zc)
    z3=ttnn.add(z3, template(z3))
    z3=msa_mod(z3, m_feat)
    sc=lin(ln(s,'layernorm_s.weight','layernorm_s.bias'), g('linear_no_bias_s.weight'))
    s=ttnn.add(s_init, sc)
    s,z3=PF(ttnn.reshape(s,(1,N,384)), z3)
    s=ttnn.reshape(s,(N,384))
    print('cycle',cyc,'done',flush=True)
so=torch.Tensor(ttnn.to_torch(s)).float().reshape(s_gold.shape); zo=torch.Tensor(ttnn.to_torch(z3)).float().reshape(z_gold.shape)
def pcc(a,b):
    a=a.flatten().double(); b=b.flatten().double()
    return float(((a-a.mean())*(b-b.mean())).sum()/((a-a.mean()).norm()*(b-b.mean()).norm()))
print('FULL TRUNK (10 cycles)  s PCC %.5f  z PCC %.5f'%(pcc(so,s_gold),pcc(zo,z_gold)),flush=True)
