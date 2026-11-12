"""Learned robot control with frozen Gemma geometry and controlled nulls.

Execute only through the fixed experiment entry point. All split definitions,
optimization budgets, model variants and test conditions come from config.json.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import time

import numpy as np
import ogbench
import torch
from torch import nn
from torch.nn import functional as F
from ogbench.manipspace.oracles.markov.cube_markov import CubeMarkovOracle

from borrowed_recovery.experiment import emit, target_for_oracle, wilson


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):
            h.update(block)
    return h.hexdigest()


def make_env(cfg):
    return ogbench.make_env_and_datasets(cfg['environment'],env_only=True,
                                       permute_blocks=False,visualize_info=False)


def reset(env,seed,condition='id'):
    """Fixed task1 geometry with goal jitter; OOD moves the goal farther in x."""
    rng = np.random.default_rng(seed+880000)
    if condition=='id':
        delta = rng.uniform(-.03,.03,2)
    elif condition=='shift':
        delta = np.array([rng.choice([-1.,1.])*rng.uniform(.06,.10),rng.uniform(-.03,.03)])
    else:
        raise ValueError(condition)
    task = dict(task_name='task1_goal_jitter',
                init_xyzs=np.array([[.425,0.,.02],[.425,-.1,.02]]),
                goal_xyzs=np.array([[.425,0.,.02],[.425+delta[0],.1+delta[1],.02]]))
    env.unwrapped.task_infos[0] = task
    np.random.seed(seed)
    env.action_space.seed(seed)
    obs,info=env.reset(seed=seed,options={'render_goal':False})
    info,idx,target=target_for_oracle(env,info)
    assert idx==1
    return obs,info,target


def features(info,target):
    ey=float(info['proprio/effector_yaw'][0])
    cy=float(info['privileged/block_1_yaw'][0])
    return np.concatenate([
        info['proprio/joint_pos'],info['proprio/effector_pos'],[math.sin(ey),math.cos(ey)],
        info['proprio/gripper_opening'],info['proprio/gripper_contact'],
        info['privileged/block_1_pos'],[math.sin(cy),math.cos(cy)],target,
    ]).astype(np.float32)


def collect(cfg):
    spec={k:cfg[k] for k in ['environment','train_start','train_demos','validation_start',
                            'validation_demos','data_horizon','demonstration_noise']}
    spec['features']='21_raw_state_fields_goal_jitter_v1'
    digest=hashlib.sha256(json.dumps(spec,sort_keys=True).encode()).hexdigest()[:16]
    root=Path(cfg['cache_root'])/'datasets'/digest
    root.mkdir(parents=True,exist_ok=True)
    path=root/'demonstrations.npz'
    metadata=root/'metadata.json'
    if path.exists() and metadata.exists():
        meta=json.loads(metadata.read_text())
        assert meta['spec']==spec and meta['sha256']==sha(path)
        emit('dataset_cache',**meta)
    else:
        env=make_env(cfg)
        arrays={};stats={}
        try:
            for split,start,n,noise in [('train',cfg['train_start'],cfg['train_demos'],cfg['demonstration_noise']),
                                       ('validation',cfg['validation_start'],cfg['validation_demos'],0.)]:
                xs=[];ys=[];episodes=[];successes=0
                for seed in range(start,start+n):
                    obs,info,target=reset(env,seed)
                    oracle=CubeMarkovOracle(env=env)
                    np.random.seed(seed+400000)
                    oracle.reset(obs,info)
                    rng=np.random.default_rng(seed+500000)
                    success=False
                    for step in range(cfg['data_horizon']):
                        action=oracle.select_action(obs,info)
                        xs.append(features(info,target));ys.append(action.copy());episodes.append(seed)
                        execute=action.copy()
                        if noise:
                            execute[:4]+=rng.normal(0,noise,4)
                        obs,_,term,trunc,next_info=env.step(np.clip(execute,-1,1))
                        info,_,_=target_for_oracle(env,next_info)
                        success=bool(info['success'])
                        if success or term or trunc:break
                    successes+=int(success)
                    if (seed-start+1)%32==0:
                        emit('dataset_progress',split=split,episodes=seed-start+1,
                             successes=successes,transitions=len(xs))
                arrays[split+'_x']=np.asarray(xs,dtype=np.float32)
                arrays[split+'_y']=np.asarray(ys,dtype=np.float32)
                arrays[split+'_episode']=np.asarray(episodes,dtype=np.int64)
                stats[split]={'episodes':n,'successes':successes,'transitions':len(xs)}
        finally:
            env.close()
        assert arrays['train_x'].shape[1]==21
        np.savez_compressed(path,**arrays)
        meta={'spec':spec,'sha256':sha(path),'stats':stats}
        metadata.write_text(json.dumps(meta,indent=2))
        emit('dataset_created',**meta)
    arrays=dict(np.load(path))
    assert not set(arrays['train_episode']) & set(arrays['validation_episode'])
    return arrays,meta


class MLP(nn.Module):
    def __init__(self,n):
        super().__init__()
        self.network=nn.Sequential(nn.Linear(n,381),nn.GELU(),nn.Linear(381,381),nn.GELU(),nn.Linear(381,5))
    def forward(self,x):return self.network(x)


def rms(x,weight=None):
    y=x.float()*torch.rsqrt(x.float().square().mean(-1,keepdim=True)+1e-6)
    if weight is not None:y=y*weight.float()
    return y.to(x.dtype)


class GemmaSingleToken(nn.Module):
    """Exact single-token algebra of the public L26 block, excluding unused Q/K.

    With one token and no KV cache softmax is exactly one; Q/K/RoPE cannot
    affect this block's output. This is explicitly not an attention-copy test.
    """
    def __init__(self,state,mode,seed,dtype):
        super().__init__()
        gen=torch.Generator(device='cpu').manual_seed(seed+770000)
        self.weights=nn.ParameterDict()
        self.stats={}
        for name,w in state.items():
            if name.startswith(('self_attn.q_','self_attn.k_')):continue
            v=w.to(dtype).clone()
            if v.ndim==2:
                mean=float(v.float().mean());sd=float(v.float().std())
                if mode=='random':
                    v=(torch.randn(v.shape,generator=gen)*sd+mean).to(dtype)
                elif mode=='scrambled':
                    # Independent row and column permutations preserve every singular value.
                    ri=torch.randperm(v.shape[0],generator=gen)
                    ci=torch.randperm(v.shape[1],generator=gen)
                    v=v[ri][:,ci].contiguous()
                self.stats[name]={'shape':list(v.shape),'source_mean':mean,'source_std':sd,
                                  'variant_mean':float(v.float().mean()),'variant_std':float(v.float().std())}
            self.weights[name.replace('.','__')]=nn.Parameter(v,requires_grad=False)
        self.dtype=dtype
    def w(self,name):return self.weights[name.replace('.','__')]
    def forward(self,x,ablate=None,return_features=False):
        x=x.to(self.dtype)
        v=F.linear(rms(x,self.w('input_layernorm.weight')),self.w('self_attn.v_proj.weight'))
        v=rms(v.reshape(-1,16,256)).repeat_interleave(2,dim=1)
        if ablate is not None:
            mask=torch.ones(32,device=v.device,dtype=v.dtype);mask[ablate]=0
            v=v*mask[None,:,None]
        attention=F.linear(v.flatten(1),self.w('self_attn.o_proj.weight'))
        h=x+rms(attention,self.w('post_attention_layernorm.weight'))
        u=rms(h,self.w('pre_feedforward_layernorm.weight'))
        gate=F.gelu(F.linear(u,self.w('mlp.gate_proj.weight')),approximate='tanh')
        neurons=gate*F.linear(u,self.w('mlp.up_proj.weight'))
        output=h+rms(F.linear(neurons,self.w('mlp.down_proj.weight')),self.w('post_feedforward_layernorm.weight'))
        output=output*self.w('layer_scalar')
        if return_features:return output.float(),neurons.float()
        return output.float()


class FrozenPolicy(nn.Module):
    def __init__(self,n,state,mode,seed,dtype):
        super().__init__()
        self.encoder=nn.Linear(n,5376)
        self.input_norm=nn.LayerNorm(5376)
        self.decoder=nn.Linear(5376,5)
        self.substrate=GemmaSingleToken(state,mode,seed,dtype)
    def forward(self,x):
        z=self.substrate(self.input_norm(self.encoder(x)))
        return self.decoder(F.layer_norm(z,(5376,)))
    def fingerprint(self,x):
        z,neurons=self.substrate(self.input_norm(self.encoder(x)),return_features=True)
        z=F.layer_norm(z,(5376,))
        return z,neurons,self.decoder(z)


def loss(pred,target):
    return F.mse_loss(pred[:,:4],target[:,:4])+.25*F.binary_cross_entropy_with_logits(pred[:,4],(target[:,4]>0).float())


@torch.no_grad()
def validation_loss(model,x,y,batch=128):
    model.eval();total=0.
    for i in range(0,len(x),batch):total+=float(loss(model(x[i:i+batch]),y[i:i+batch]))*len(x[i:i+batch])
    return total/len(x)


@torch.no_grad()
def evaluate(model,mean,std,device,cfg,seed_start,n,condition,tag):
    """Batch policy inference across independently seeded CPU physics worlds."""
    envs=[make_env(cfg) for _ in range(min(cfg['eval_parallel'],n))]
    rows=[]
    try:
        for begin in range(0,n,len(envs)):
            active=[];infos={};targets={};steps={};hashes={};traces={}
            for slot,seed in enumerate(range(seed_start+begin,min(seed_start+n,seed_start+begin+len(envs)))):
                _,info,target=reset(envs[slot],seed,condition)
                infos[slot]=info;targets[slot]=target;steps[slot]=0;active.append(slot)
                hashes[slot]=hashlib.sha256(info['qpos'].tobytes()).hexdigest()
                traces[slot]=[]
            while active:
                xx=np.stack([features(infos[s],targets[s]) for s in active])
                xx=(torch.from_numpy(xx).to(device)-mean)/std
                pred=model(xx).float().cpu().numpy()
                actions=np.clip(pred,-1,1);actions[:,4]=np.where(pred[:,4]>0,1.,-1.)
                finished=[]
                for j,s in enumerate(active):
                    if begin==0 and s==0 and steps[s]%3==0:
                        traces[s].append({'step':steps[s],'cube':infos[s]['privileged/block_1_pos'].tolist(),
                                          'effector':infos[s]['proprio/effector_pos'].tolist(),
                                          'target':targets[s].tolist(),'action':actions[j].tolist()})
                    _,_,term,trunc,info=envs[s].step(actions[j])
                    steps[s]+=1;infos[s]=info;success=bool(info['success'])
                    if success or term or trunc or steps[s]>=cfg['eval_horizon']:
                        row={'condition':condition,'seed':seed_start+begin+s,'success':success,'steps':steps[s],
                             'goal_error_m':float(np.linalg.norm(info['privileged/block_1_pos']-targets[s])),
                             'initial_hash':hashes[s]}
                        rows.append(row);finished.append(s)
                        if traces[s]:
                            Path('outputs',f'trace-{tag}-{condition}.json').write_text(json.dumps({'outcome':row,'frames':traces[s]}))
                active=[s for s in active if s not in finished]
    finally:
        for env in envs:env.close()
    return rows


def expert_evaluation(cfg):
    env=make_env(cfg);out=[]
    try:
        for condition,start in [('id',cfg['test_start']),('shift',cfg['shift_start'])]:
            for seed in range(start,start+cfg['expert_test_episodes']):
                obs,info,target=reset(env,seed,condition)
                oracle=CubeMarkovOracle(env=env);np.random.seed(seed+400000);oracle.reset(obs,info)
                success=False
                for step in range(cfg['eval_horizon']):
                    action=oracle.select_action(obs,info)
                    obs,_,term,trunc,next_info=env.step(action)
                    info,_,_=target_for_oracle(env,next_info);success=bool(info['success'])
                    if success or term or trunc:break
                out.append({'condition':condition,'seed':seed,'success':success,'steps':step+1})
    finally:env.close()
    emit('expert_control',rows=out)
    return out


def run(cfg):
    out=Path('outputs');out.mkdir(exist_ok=True)
    torch.set_num_threads(cfg.get('cpu_threads',4))
    device=torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    versions={k:importlib.metadata.version(k) for k in ['torch','numpy','ogbench','mujoco','safetensors']}
    metadata={'config':cfg,'run_id':os.environ.get('ORX_RUN_ID'),'commit':os.environ.get('ORX_COMMIT'),
              'device':str(device),'versions':versions,'python':platform.python_version()}
    emit('geometry_configuration',**metadata)
    arrays,data_meta=collect(cfg)
    x=torch.from_numpy(arrays['train_x']).to(device);y=torch.from_numpy(arrays['train_y']).to(device)
    mean=x.mean(0);std=x.std(0).clamp_min(.05)
    x=(x-mean)/std
    vx=(torch.from_numpy(arrays['validation_x']).to(device)-mean)/std
    vy=torch.from_numpy(arrays['validation_y']).to(device)
    experts=expert_evaluation(cfg)
    state=None;weight_meta=None
    if any(mode!='mlp' for mode in cfg['modes']):
        from borrowed_recovery.gemma_weights import load_layer
        state,weight_meta=load_layer(cfg['cache_root'])
        emit('pretrained_weights',**weight_meta)
    results=[];all_rows=[];all_curves=[]
    for seed in cfg['training_seeds']:
        for mode in cfg['modes']:
            torch.manual_seed(seed);np.random.seed(seed)
            began=time.monotonic()
            model=MLP(21) if mode=='mlp' else FrozenPolicy(21,state,mode,seed,torch.bfloat16)
            model=model.to(device)
            trainable=[p for p in model.parameters() if p.requires_grad]
            emit('model_started',mode=mode,training_seed=seed,trainable=sum(p.numel() for p in trainable),
                 frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad))
            opt=torch.optim.AdamW(trainable,lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
            sampler=np.random.default_rng(seed+620000)
            best_loss=float('inf');best_step=0;best=None;curve=[];fps={}
            panel=vx[:128]
            for step in range(cfg['train_steps']+1):
                if step in cfg['checkpoints']:
                    val=validation_loss(model,vx,vy,cfg['batch_size'])
                    record={'mode':mode,'training_seed':seed,'step':step,'validation_loss':val,
                            'elapsed_s':time.monotonic()-began}
                    curve.append(record);emit('learning_checkpoint',**record)
                    if val<best_loss:
                        best_loss=val;best_step=step
                        best={k:v.detach().cpu().clone() for k,v in model.state_dict().items() if not k.startswith('substrate.')}
                    if mode!='mlp':
                        with torch.no_grad():
                            z,neu,pred=model.fingerprint(panel)
                            fps[str(step)]={'hidden':z.cpu().numpy(),'neurons':neu.cpu().numpy(),'actions':pred.cpu().numpy()}
                if step==cfg['train_steps']:break
                model.train()
                idx=torch.as_tensor(sampler.integers(0,len(x),cfg['batch_size']),device=device)
                pred=model(x[idx]);ll=loss(pred,y[idx]);assert torch.isfinite(ll),'nonfinite training loss'
                opt.zero_grad(set_to_none=True);ll.backward();torch.nn.utils.clip_grad_norm_(trainable,1.0);opt.step()
                if (step+1)%cfg['progress_interval']==0:
                    emit('training_progress',mode=mode,training_seed=seed,step=step+1,
                         minibatch_loss=float(ll.detach()),elapsed_s=time.monotonic()-began)
            model.load_state_dict(best,strict=(mode=='mlp'));model.eval()
            tag=f'{mode}-s{seed}'
            torch.save({'adapter':best,'mean':mean.cpu(),'std':std.cpu(),'mode':mode,'seed':seed,
                        'best_step':best_step,'config':cfg},out/f'policy-{tag}.pt')
            if fps:
                np.savez_compressed(out/f'fingerprints-{tag}.npz',**{f'{s}_{kind}':value for s,d in fps.items() for kind,value in d.items()})
            per_condition=[]
            for condition,start in [('id',cfg['test_start']),('shift',cfg['shift_start'])]:
                rows=evaluate(model,mean,std,device,cfg,start,cfg['test_episodes'],condition,tag)
                k=sum(r['success'] for r in rows);n=len(rows)
                result={'mode':mode,'training_seed':seed,'condition':condition,'successes':k,'n':n,
                        'success_rate':k/n,'wilson95':wilson(k,n),'mean_steps':float(np.mean([r['steps'] for r in rows])),
                        'mean_goal_error_m':float(np.mean([r['goal_error_m'] for r in rows]))}
                per_condition.append(result);emit('policy_evaluation',**result)
                all_rows.extend({'mode':mode,'training_seed':seed,**r} for r in rows)
            results.extend(per_condition);all_curves.extend(curve)
            emit('model_completed',mode=mode,training_seed=seed,best_step=best_step,best_validation_loss=best_loss,
                 wall_time_s=time.monotonic()-began)
            (out/'geometry-results.json').write_text(json.dumps(results,indent=2))
            (out/'geometry-episodes.json').write_text(json.dumps(all_rows,indent=2))
            (out/'learning-curves.json').write_text(json.dumps(all_curves,indent=2))
            del model,opt,best,trainable;gc.collect()
            if device.type=='mps':torch.mps.empty_cache()
    hashes={}
    for r in all_rows:
        key=(r['condition'],r['seed'])
        if key in hashes:assert hashes[key]==r['initial_hash'],'unpaired physical state'
        hashes[key]=r['initial_hash']
    summary={**metadata,'dataset':data_meta,'weights':weight_meta,'results':results,'expert_controls':experts,
             'checks':{'disjoint_episode_splits':True,'paired_eval_states':True,'frozen_excluded_from_optimizer':True}}
    (out/'geometry-summary.json').write_text(json.dumps(summary,indent=2));emit('geometry_summary',**summary)
