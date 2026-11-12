"""Offline arithmetic on recorded experiments; no training or policy inference."""
from pathlib import Path
import json
import csv
import sys
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=Path('/Users/aravindkannappan/.local/share/openresearch/local-runs')
MAIN='ed2527c3-cc8f-4df8-a7ee-3a68cbdb6e5d'

def write_csv(name,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with (HERE/name).open('w') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)

def audit(path,best_step):
    data=np.load(path)
    result={'model':path.stem.replace('fingerprints-',''),'panel_states':len(data['panel_indices']),
            'selected_step':best_step}
    per_neuron=[]
    for endpoint in [best_step,1000]:
        a=data['0_neurons'].astype(np.float64);b=data[f'{endpoint}_neurons'].astype(np.float64)
        an=np.linalg.norm(a,axis=0);bn=np.linalg.norm(b,axis=0)
        valid=(an>1e-10)&(bn>1e-10)
        raw=np.full(a.shape[1],np.nan);raw[valid]=np.einsum('ij,ij->j',a[:,valid],b[:,valid])/(an[valid]*bn[valid])
        ac=a-a.mean(0);bc=b-b.mean(0)
        acn=np.linalg.norm(ac,axis=0);bcn=np.linalg.norm(bc,axis=0)
        valid_c=(acn>1e-10)&(bcn>1e-10)
        centered=np.full(a.shape[1],np.nan)
        centered[valid_c]=np.einsum('ij,ij->j',ac[:,valid_c],bc[:,valid_c])/(acn[valid_c]*bcn[valid_c])
        ratio=bn[valid]/an[valid]
        result[str(endpoint)]={'raw_cosine_median':float(np.nanmedian(raw)),
            'centered_cosine_median':float(np.nanmedian(centered)),
            'centered_cosine_below_zero_fraction':float(np.mean(centered[valid_c]<0)),
            'centered_cosine_below_half_fraction':float(np.mean(centered[valid_c]<.5)),
            'response_norm_ratio_median':float(np.median(ratio)),
            'response_norm_ratio_p10_p90':np.quantile(ratio,[.1,.9]).tolist(),
            'eligible_neurons':int(valid_c.sum()),'total_neurons':a.shape[1]}
        if endpoint==best_step:
            per_neuron=[{'neuron':i,'raw_cosine':float(raw[i]),'centered_cosine':float(centered[i]),
                         'response_norm_ratio':float(bn[i]/an[i]) if an[i]>1e-10 else None} for i in range(a.shape[1])]
    return result,per_neuron

def main():
    source=Path(sys.argv[1]) if len(sys.argv)>1 else HERE/'data'
    results=json.loads((source/'geometry-results.json').read_text())
    episodes=json.loads((source/'geometry-episodes.json').read_text())
    curves=json.loads((source/'learning-curves.json').read_text())
    summary=[];pairs=[];audit_rows=[]
    if not list(source.glob('fingerprints-pretrained-*.npz')) and (HERE/'neuron-audit.json').exists():
        audit_rows=json.loads((HERE/'neuron-audit.json').read_text())
    for condition in ['id','shift']:
        for mode in ['pretrained','random','scrambled','mlp']:
            rr=[r for r in results if r['condition']==condition and r['mode']==mode]
            if not rr:continue
            rates=[r['success_rate'] for r in rr]
            summary.append({'condition':condition,'mode':mode,'training_seeds':len(rr),
                            'mean_success_rate':float(np.mean(rates)),'min_seed_rate':min(rates),
                            'max_seed_rate':max(rates),'successes':sum(r['successes'] for r in rr),
                            'rollouts':sum(r['n'] for r in rr)})
        for training_seed in [42,1337,2024]:
            base={r['seed']:r for r in episodes if r['mode']=='pretrained' and r['training_seed']==training_seed and r['condition']==condition}
            for mode in ['random','scrambled','mlp']:
                control={r['seed']:r for r in episodes if r['mode']==mode and r['training_seed']==training_seed and r['condition']==condition}
                if not base or set(base)!=set(control):continue
                assert all(base[s]['initial_hash']==control[s]['initial_hash'] for s in base)
                both=sum(base[s]['success'] and control[s]['success'] for s in base)
                only_a=sum(base[s]['success'] and not control[s]['success'] for s in base)
                only_b=sum(not base[s]['success'] and control[s]['success'] for s in base)
                pairs.append({'condition':condition,'training_seed':training_seed,'comparison':'pretrained vs '+mode,
                              'both_success':both,'pretrained_only':only_a,'control_only':only_b,
                              'both_fail':len(base)-both-only_a-only_b,'n':len(base),'rate_difference':(only_a-only_b)/len(base)})
    for path in sorted(source.glob('fingerprints-pretrained-*.npz')):
        seed=int(path.stem.rsplit('s',1)[1]);curve=[r for r in curves if r['mode']=='pretrained' and r['training_seed']==seed]
        if not curve:continue
        best=min(curve,key=lambda r:r['validation_loss'])['step']
        item,neurons=audit(path,best);audit_rows.append(item)
        write_csv(path.stem+'-neuron-audit.csv',neurons)
    write_csv('success-by-seed.csv',results);write_csv('paired-outcomes.csv',pairs)
    write_csv('success-summary.csv',summary);write_csv('learning-curves.csv',curves)
    (HERE/'neuron-audit.json').write_text(json.dumps(audit_rows,indent=2))
    (HERE/'analysis-summary.json').write_text(json.dumps({'source_directory':str(source),'results':summary,'paired':pairs,'neuron_audit':audit_rows},indent=2))
    print(json.dumps({'results':summary,'neuron_audit':audit_rows},indent=2))

if __name__=='__main__':main()
