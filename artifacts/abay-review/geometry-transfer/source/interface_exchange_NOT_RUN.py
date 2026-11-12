"""Paired-state calibration of independently trained robot interfaces.

No controller retraining. All maps are fitted only on 512 training states;
closed-loop evaluation uses fresh reset seeds. Execute through experiment.py.
"""
from pathlib import Path
import json
import os
import time
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from borrowed_recovery.experiment import emit, wilson
from borrowed_recovery.geometry import (collect, GemmaSingleToken, evaluate, loss, sha)
from borrowed_recovery.gemma_weights import load_layer


class SharedPolicy(nn.Module):
    def __init__(self, checkpoint, substrate):
        super().__init__()
        self.encoder=nn.Linear(21,5376)
        self.input_norm=nn.LayerNorm(5376)
        self.decoder=nn.Linear(5376,5)
        self.substrate=substrate
        self.load_state_dict(checkpoint['adapter'],strict=False)
    def hidden(self,x):
        return F.layer_norm(self.substrate(self.input_norm(self.encoder(x))),(5376,))
    def forward(self,x):return self.decoder(self.hidden(x))


class Exchange(nn.Module):
    def __init__(self, source, weight, bias):
        super().__init__()
        self.source=source
        self.register_buffer('weight',torch.from_numpy(weight.astype(np.float32)))
        self.register_buffer('bias',torch.from_numpy(bias.astype(np.float32)))
    def forward(self,x):return self.source.hidden(x) @ self.weight + self.bias


@torch.no_grad()
def hidden_batch(model,x,batch=128):
    return np.concatenate([model.hidden(x[i:i+batch]).cpu().numpy() for i in range(0,len(x),batch)])


def basis(h,rank):
    mean=h.mean(0)
    centered=h-mean
    # Thin SVD; basis vectors are columns, sample coordinates are H @ V.
    _,s,vt=np.linalg.svd(centered,full_matrices=False)
    return mean,vt[:rank].T,s


def fit_maps(a,b,wa,ba,rank,seed):
    ma,va,sa=basis(a,rank);mb,vb,sb=basis(b,rank)
    xa=(a-ma) @ va;xb=(b-mb) @ vb
    u,s,vt=np.linalg.svd(xb.T @ xa,full_matrices=False)
    rotation=u@vt;scale=float(s.sum()/np.square(xb).sum())
    aligned=vb @ (scale*rotation) @ (va.T @ wa)
    aligned_bias=ma@wa+ba-mb@aligned
    perm=np.random.default_rng(seed).permutation(len(xa))
    u,ss,vt=np.linalg.svd(xb.T @ xa[perm],full_matrices=False)
    shuffled_scale=float(ss.sum()/np.square(xb).sum())
    shuffled=vb @ (shuffled_scale*(u@vt)) @ (va.T @ wa)
    shuffled_bias=ma@wa+ba-mb@shuffled
    projected=va @ (va.T @ wa)
    projected_bias=ma@wa+ba-ma@projected
    # Functional baseline: ridge predicts A's action logits, using the same states and B basis.
    target=a@wa+ba;tm=target.mean(0)
    lam=1e-3*float(np.trace(xb.T@xb)/rank)
    coeff=np.linalg.solve(xb.T@xb+lam*np.eye(rank),xb.T@(target-tm))
    ridge=vb@coeff;ridge_bias=tm-mb@ridge
    info={'rank':rank,'paired_states':len(a),'procrustes_scale':scale,
          'shuffled_scale':shuffled_scale,'ridge_lambda':lam,
          'a_retained_variance':float(np.square(sa[:rank]).sum()/np.square(sa).sum()),
          'b_retained_variance':float(np.square(sb[:rank]).sum()/np.square(sb).sum())}
    return {'aligned_exchange':(aligned,aligned_bias),'shuffled_alignment':(shuffled,shuffled_bias),
            'a_rank_control':(projected,projected_bias),'ridge_exchange':(ridge,ridge_bias)},info


def representations(a,b):
    an=a/(np.linalg.norm(a,axis=1,keepdims=True)+1e-12)
    bn=b/(np.linalg.norm(b,axis=1,keepdims=True)+1e-12)
    ac=a-a.mean(0);bc=b-b.mean(0)
    ga=ac@ac.T;gb=bc@bc.T
    cka=float((ga*gb).sum()/np.sqrt(np.square(ga).sum()*np.square(gb).sum()))
    return {'paired_raw_cosine_mean':float(np.einsum('ij,ij->i',an,bn).mean()),
            'shuffled_raw_cosine_mean':float(np.einsum('ij,ij->i',an,bn[::-1]).mean()),
            'linear_cka':cka}


def run(cfg):
    began=time.monotonic();out=Path('outputs');out.mkdir(exist_ok=True)
    torch.set_num_threads(cfg['cpu_threads'])
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    source=Path(cfg['parent_policy_dir'])
    paths={name:source/f'policy-pretrained-s{seed}.pt' for name,seed in [('a',cfg['decoder_seed']),('b',cfg['encoder_seed'])]}
    checkpoints={name:torch.load(path,map_location='cpu',weights_only=False) for name,path in paths.items()}
    emit('interface_configuration',config=cfg,run_id=os.environ.get('ORX_RUN_ID'),commit=os.environ.get('ORX_COMMIT'),
         source_hashes={name:sha(path) for name,path in paths.items()},device=str(device))
    a_ck,b_ck=checkpoints['a'],checkpoints['b']
    assert torch.equal(a_ck['mean'],b_ck['mean']) and torch.equal(a_ck['std'],b_ck['std'])
    arrays,data_meta=collect(a_ck['config'])
    mean=a_ck['mean'].to(device);std=a_ck['std'].to(device)
    state,weight_meta=load_layer(cfg['cache_root'])
    substrate=GemmaSingleToken(state,'pretrained',cfg['decoder_seed'],torch.bfloat16).to(device)
    del state
    policies={name:SharedPolicy(ck,substrate).to(device).eval() for name,ck in checkpoints.items()}
    idx=np.random.default_rng(cfg['alignment_seed']).choice(len(arrays['train_x']),cfg['alignment_states'],replace=False)
    fit_x=(torch.from_numpy(arrays['train_x'][idx]).to(device)-mean)/std
    validation_x=(torch.from_numpy(arrays['validation_x']).to(device)-mean)/std
    hidden={name:hidden_batch(p,fit_x).astype(np.float64) for name,p in policies.items()}
    heldout={name:hidden_batch(p,validation_x).astype(np.float64) for name,p in policies.items()}
    wa=policies['a'].decoder.weight.detach().cpu().numpy().T.astype(np.float64)
    ba=policies['a'].decoder.bias.detach().cpu().numpy().astype(np.float64)
    wb=policies['b'].decoder.weight.detach().cpu().numpy().T.astype(np.float64)
    bb=policies['b'].decoder.bias.detach().cpu().numpy().astype(np.float64)
    maps,map_info=fit_maps(hidden['a'],hidden['b'],wa,ba,cfg['alignment_rank'],cfg['alignment_seed']+1)
    emit('alignment_fit',**map_info,validation_geometry=representations(heldout['a'],heldout['b']))
    np.savez_compressed(out/'interface-maps.npz',fit_indices=idx,
                        **{name+'_'+kind:value for name,pair in maps.items() for kind,value in zip(['weight','bias'],pair)})
    variants={'own_a':policies['a'],'own_b':policies['b'],'raw_exchange':Exchange(policies['b'],wa,ba).to(device)}
    initial_path=source/f'adapter-pretrained-s{cfg["decoder_seed"]}-step0.pt'
    initial=torch.load(initial_path,map_location='cpu',weights_only=True)
    emit('component_swap_configuration',initial_adapter_sha256=sha(initial_path),seed=cfg['decoder_seed'])
    initial_policy=SharedPolicy({'adapter':initial},substrate).to(device).eval()
    initial_hidden=hidden_batch(initial_policy,validation_x).astype(np.float64)
    wi=initial_policy.decoder.weight.detach().cpu().numpy().T.astype(np.float64)
    bi=initial_policy.decoder.bias.detach().cpu().numpy().astype(np.float64)
    component_predictions={
        'initial_both':initial_hidden@wi+bi,
        'initial_encoder_trained_decoder':initial_hidden@wa+ba,
        'trained_encoder_initial_decoder':heldout['a']@wi+bi,
    }
    for name,(weight,bias) in maps.items():
        variants[name]=Exchange(policies['a'] if name=='a_rank_control' else policies['b'],weight,bias).to(device)
    variants['initial_both']=initial_policy
    variants['initial_encoder_trained_decoder']=Exchange(initial_policy,wa,ba).to(device)
    variants['trained_encoder_initial_decoder']=Exchange(policies['a'],wi,bi).to(device)
    # Offline checks reuse hidden activations, including a direct native-head consistency check.
    teacher=heldout['a']@wa+ba
    offline=[]
    for name in variants:
        if name=='own_a':pred=teacher
        elif name=='own_b':pred=heldout['b']@wb+bb
        elif name=='raw_exchange':pred=heldout['b']@wa+ba
        elif name in component_predictions:pred=component_predictions[name]
        else:
            w,b=maps[name];pred=heldout['a' if name=='a_rank_control' else 'b']@w+b
        result={'variant':name,'continuous_rmse_to_a':float(np.sqrt(np.mean((np.clip(pred[:,:4],-1,1)-np.clip(teacher[:,:4],-1,1))**2))),
                'gripper_disagreement_to_a':float(np.mean((pred[:,4]>0)!=(teacher[:,4]>0))),
                'expert_imitation_loss':float(loss(torch.from_numpy(pred.astype(np.float32)),torch.from_numpy(arrays['validation_y'])))}
        offline.append(result);emit('interface_offline',**result)
    rows=[];results=[];physical={}
    for name,model in variants.items():
        emit('interface_evaluation_started',variant=name)
        episodes=evaluate(model,mean,std,device,cfg,cfg['test_start'],cfg['test_episodes'],'id','interface-'+name)
        for r in episodes:
            key=r['seed']
            if key in physical:assert physical[key]==r['initial_hash']
            physical[key]=r['initial_hash']
        k=sum(r['success'] for r in episodes);n=len(episodes)
        result={'variant':name,'successes':k,'n':n,'success_rate':k/n,'wilson95':wilson(k,n),
                'mean_goal_error_m':float(np.mean([r['goal_error_m'] for r in episodes]))}
        rows.extend({'variant':name,**r} for r in episodes);results.append(result)
        emit('interface_evaluation',**result)
        (out/'interface-results.json').write_text(json.dumps(results,indent=2))
        (out/'interface-episodes.json').write_text(json.dumps(rows,indent=2))
    summary={'config':cfg,'parent_dataset':data_meta,'map_info':map_info,'offline':offline,'results':results,
             'source_hashes':{name:sha(path) for name,path in paths.items()},'wall_time_s':time.monotonic()-began}
    (out/'interface-summary.json').write_text(json.dumps(summary,indent=2));emit('interface_summary',**summary)
