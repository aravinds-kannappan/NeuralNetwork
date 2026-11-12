from pathlib import Path
import json,csv,shutil,hashlib,zipfile,html
R=Path('/Users/aravindkannappan/.local/share/openresearch/files/neuralnetworks/abay-review/geometry-transfer')
S=Path('/Users/aravindkannappan/.local/share/openresearch/local-runs/fb3dd5e8-79fb-4457-ba32-7fbf23077a3c/repo/outputs')
P=Path.cwd()
for folder in ['data','models','source']: (R/folder).mkdir(exist_ok=True)
for name in ['geometry-results.json','geometry-episodes.json','learning-curves.json']:
    shutil.copy2(S/name,R/'data'/name)
results=json.loads((S/'geometry-results.json').read_text());episodes=json.loads((S/'geometry-episodes.json').read_text());curves=json.loads((S/'learning-curves.json').read_text())
def csvout(name,rows):
    with (R/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
csvout('success-by-seed.csv',results);csvout('learning-curves.csv',curves)
pairs=[]
for condition in ['id','shift']:
    base={r['seed']:r for r in episodes if r['mode']=='pretrained' and r['training_seed']==42 and r['condition']==condition}
    for mode in ['random','scrambled','mlp']:
        control={r['seed']:r for r in episodes if r['mode']==mode and r['training_seed']==42 and r['condition']==condition}
        assert set(base)==set(control)
        assert all(base[s]['initial_hash']==control[s]['initial_hash'] for s in base)
        both=sum(base[s]['success'] and control[s]['success'] for s in base)
        only_a=sum(base[s]['success'] and not control[s]['success'] for s in base)
        only_b=sum(not base[s]['success'] and control[s]['success'] for s in base)
        pairs.append(dict(condition=condition,training_seed=42,comparison='pretrained vs '+mode,both_success=both,pretrained_only=only_a,control_only=only_b,both_fail=len(base)-both-only_a-only_b,n=len(base),oracle_union=both+only_a+only_b))
csvout('paired-outcomes.csv',pairs)
for mode,seed in sorted({(r['mode'],r['training_seed']) for r in results}):shutil.copy2(S/f'policy-{mode}-s{seed}.pt',R/'models'/f'policy-{mode}-s{seed}.pt')
shutil.copy2(S/'adapter-pretrained-s42-step0.pt',R/'models'/'adapter-pretrained-s42-step0.pt')
for name in ['borrowed_recovery','pyproject.toml','uv.lock','.python-version','GEOMETRY_PROTOCOL.md','config.json']:
    src=P/name;dest=R/'source'/name
    if src.is_dir():shutil.copytree(src,dest,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    else:shutil.copy2(src,dest)
# Keep the proposed intervention visibly separate from code that was executed.
shutil.copy2('/tmp/interface_exchange.py',R/'source'/'interface_exchange_NOT_RUN.py')
provenance={'status':'stopped_at_user_request','planned_training_seeds':[42,1337,2024],
 'complete_comparison_seeds':[42],'additional_pretrained_seeds':[1337],
 'completed_learned_policy_rollouts':sum(r['n'] for r in results),
 'runs':[{'id':'ed2527c3-cc8f-4df8-a7ee-3a68cbdb6e5d','commit':'e77752e1eb36dcf625d13d876d3ea26340f2f54c','status':'terminated after three completed models'},
         {'id':'fb3dd5e8-79fb-4457-ba32-7fbf23077a3c','commit':'55c942f','status':'cancelled at user request after two more completed models'},
         {'id':'b9238468-c01d-4d1f-a879-2ea48577baa8','commit':'2b400eaf0edec9ddf10c4f0675b56e5e82369569','status':'completed 256-demonstration feasibility baseline'}],
 'not_run':['interface exchange','paired-state alignment','encoder/decoder component swaps','multi-arm coordination','learned expert routing','recursive self-improvement'],
 'model':{'repository':'google/gemma-4-31B','revision':'5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89','layer':26},
 'files':{str(p.relative_to(R)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['data','models','source'] for p in (R/folder).rglob('*') if p.is_file()}}
(R/'provenance.json').write_text(json.dumps(provenance,indent=2))
(R/'analysis-summary.json').write_text(json.dumps({'status':'partial; stopped at user request','results':results,'paired':pairs,'neuron_audit':json.loads((R/'neuron-audit.json').read_text())},indent=2))
(R/'success-summary.csv').unlink(missing_ok=True)
readme='''# Recorded robot-geometry experiments

Status: stopped at the user's request. No further compute is running.

Five trained policies have complete evaluations: pretrained/random/scrambled/MLP seed 42, plus pretrained seed 1337. This is ONE complete comparison seed, not a three-seed study. See report.html, data/, success-by-seed.csv, paired-outcomes.csv and provenance.json.

models/ contains selected trained adapters and the initial seed-42 pretrained adapter. Frozen model weights are not included in the archive. The source downloader retrieves the pinned public Gemma layer (about 958 MB). These adapters operate on 21 privileged simulator features; they are not language agents.

source/config.json preserves the run configuration and original local paths for provenance. To run on a different machine, choose a local cache_root and remove resume_from, result_cache and max_models_per_run (or deliberately configure new batches). Reduce training_seeds/modes if reproducing only a subset. The fixed execution recipe is:

    uv run --locked python -m borrowed_recovery.experiment

Do not launch this merely to read the results. Training uses substantial laptop memory and compute. The source's result-cache resume settings preserve prior evaluations; a fresh replication must remove them.

source/interface_exchange_NOT_RUN.py is a proposed follow-up, not an executed experiment and not wired into the entry point. It was only syntax-checked. It has no findings.

Neuron audit: analyze_results.py performs arithmetic on recorded fingerprints. The full fingerprints remain in the original run outputs; this compact archive includes the audit and per-neuron values instead. It does not execute the original lithium-neuron experiment.
'''
(R/'README.txt').write_text(readme)
# Format only the complete matched seed, keeping the partial second seed separate.
names={'pretrained':'Pretrained Gemma','random':'Random linear matrices','scrambled':'Scrambled matrices','mlp':'Small trained MLP'}
rows=''
for mode in names:
    rs={r['condition']:r for r in results if r['mode']==mode and r['training_seed']==42}
    rows+=f"<tr><td>{names[mode]}</td><td>{rs['id']['successes']}/60 ({rs['id']['success_rate']:.1%})</td><td>{rs['shift']['successes']}/60 ({rs['shift']['success_rate']:.1%})</td></tr>"
css='''*{box-sizing:border-box}body{margin:0;background:#f5f5ef;color:#18313b;font:17px/1.65 system-ui,sans-serif}main{max-width:940px;margin:auto;padding:42px 28px 70px}header{border-top:6px solid #166e68;padding-top:24px}h1{font-size:clamp(30px,5vw,48px);line-height:1.12;letter-spacing:-.035em;margin:14px 0}h2{font-size:25px;line-height:1.3;margin-top:36px}h3{font-size:19px}a{color:#126a66}p{margin:12px 0}.small,figcaption{font-size:14px;color:#536771}.note{padding:18px 22px;background:#e5efea;border-left:4px solid #166e68;margin:22px 0}.limit{background:#f6edde;border-color:#9a6a31}.scroll{overflow:auto}table{width:100%;border-collapse:collapse;background:white}th,td{text-align:left;padding:12px;border-bottom:1px solid #d3dddd}th{background:#e5efea}li{margin:12px 0}code{overflow-wrap:anywhere;font-size:13px}.downloads{display:flex;gap:10px;flex-wrap:wrap}.downloads a{background:#18313b;color:white;padding:9px 14px;border-radius:5px;text-decoration:none;font-size:14px}figure{margin:24px 0}img{max-width:100%;height:auto;border-radius:8px}footer{border-top:1px solid #d3dddd;margin-top:32px;padding-top:16px}details{border-top:1px solid #d3dddd;padding:12px 0}summary{font-weight:650;cursor:pointer}'''
body=f'''<header><p class="small">NEURALNETWORKS · SAVED EXPERIMENT FINDINGS · NOVEMBER 2026</p><h1>Borrowed Geometry:<br>what the robot tests actually found</h1><p>Five learned policies, 600 completed evaluation rollouts, and one complete matched comparison seed.</p></header>
<div class="note"><strong>Compute stopped.</strong> The remaining experiment was cancelled at your request. Results below were saved before cancellation. The planned three-seed comparison is incomplete; the interface-exchange and alignment tests were not run.</div>
<div class="downloads"><a href="reproduction.zip">Code, results &amp; trained adapters</a><a href="success-by-seed.csv">Measured success counts</a><a href="provenance.json">Run provenance</a></div>
<h2>1. Strongest supported findings</h2>
<p><strong>Frozen layers can support learned robot control in this setup, but the completed comparison does not demonstrate an advantage specific to pretrained matrix structure.</strong> With 16 training demonstrations, pretrained Gemma and a randomized-linear-matrix control both succeeded on 30/60 familiar-target episodes. They both dropped to 9/60 on shifted targets. <a href="data/geometry-results.json">Recorded results</a></p>
<div class="scroll"><table><thead><tr><th>Training seed 42</th><th>Familiar targets</th><th>Shifted targets</th></tr></thead><tbody>{rows}</tbody></table></div>
<p class="small">All methods use the same demonstration data and test reset seeds. This is one training seed, not a statistically established ranking. For pretrained familiar-target success, the within-policy 95% Wilson interval is 37.7–62.3%; it does not quantify variation across training seeds.</p>
<p><strong>A second pretrained controller achieved 25/60 familiar and 8/60 shifted successes.</strong> Its matched random, scrambled, and MLP controls did not finish. It is reported separately and is not averaged into an unequal comparison. <a href="success-by-seed.csv">All completed policies</a></p>
<p><strong>Equal success rates concealed different strengths.</strong> On familiar targets, the pretrained and random controllers both succeeded on 18 episodes; each uniquely succeeded on another 12. Their union is 42/60 (70%). On shifted targets, the union is 17/60. These are hindsight upper bounds for a perfect selector, not measured routing or recovery performance. <a href="paired-outcomes.csv">Paired episode counts</a></p>
<p><strong>Learning occurred with very stable neuron response patterns.</strong> For the pretrained seed-42 controller, the median mean-centered cosine between initial and selected-checkpoint neuron responses was 0.99797 across the same 128 states. The median response-norm ratio was 0.99839. Parameter hashes confirmed the block remained frozen. <a href="neuron-audit.json">Audit</a> · <a href="fingerprints-pretrained-s42-neuron-audit.csv">Per-neuron measurements</a></p>
<div class="note"><strong>Interpretation, not proof:</strong> a wide frozen feature bank and the learned interface may explain much of the benefit in this setting. Stable responses are consistent with the decoder learning to use existing features. Neither explanation has yet been isolated by the proposed component-swap tests.</div>
<p><strong>A lower validation loss did not imply better closed-loop control.</strong> The selected small MLP had validation imitation loss 0.15258, lower than pretrained Gemma’s 0.17830, yet familiar-target success was 11/60 versus 30/60. This is one comparison, but it directly motivates evaluating behavior rather than selecting robotics claims from embedding similarity or prediction loss alone. <a href="learning-curves.csv">Validation losses</a> · <a href="success-by-seed.csv">Behavior</a></p>
<h2>2. What the controls establish—and what they do not</h2>
<p>The randomized control replaces each linear matrix with Gaussian entries matched to its mean and standard deviation. The scrambled control independently permutes each matrix’s rows and columns, preserving its singular values and element values while disrupting relationships between matrices. Both retain pretrained normalization vectors and the layer scalar. Thus, they are not controls for removing every effect of pretraining. <a href="source/borrowed_recovery/geometry.py">Executed implementation</a></p>
<p>The small MLP has 155,834 trainable parameters; the frozen-layer interfaces have 155,909. The latter also use 412,898,305 frozen parameters. Trainable counts are matched closely, but total capacity and compute are not. The frozen variants’ advantage over the MLP in this seed cannot uniquely identify language-derived geometry.</p>
<p>A separate feasibility run trained the small MLP on 256 demonstrations for 2,000 steps and reached 39/40 successes in both conditions. That run used different evaluation resets and more data/training. It shows the task is learnable; it is not a matched sample-efficiency comparison. <a href="provenance.json">Baseline run ID</a></p>
<h2>3. The connection to your robot team</h2>
<p><a href="https://arxiv.org/abs/2605.00333">Abay’s Borrowed Geometry paper</a> reports robot control using a frozen pretrained block with learned interfaces. These experiments exercise that architectural idea using actual Gemma weights and learned robot policies. The specific pretrained-weight advantage remains unresolved here.</p>
<p><strong>The most concrete team connection is complementary failure.</strong> Two controllers with identical aggregate success rates solved different physical episodes. That gives a measurable reason to study a router or backup controller. A successful system must detect which controller is struggling and switch without losing progress; a 70% hindsight union does not demonstrate that ability.</p>
<p><strong>A second connection is interface compatibility.</strong> Sharing frozen weights may still leave separately trained encoders and decoders using incompatible representations. A paired-state alignment experiment was prepared, but not run. No result about shared memory, mixture-of-experts routing, multiple arms, or recursive self-improvement is claimed.</p>
<div class="note limit"><strong>This is an extension, not a replication.</strong> We used behavior cloning on task-specific expert demonstrations and 21 privileged state features. Abay’s reported robotics setting uses offline reinforcement learning on a larger play dataset. Our adapters, training duration, and public model revision differ. The linked original robotics implementation and trained adapters were not available to verify. These results do not reproduce or refute his reported score.</div>
<h2>4. Most consequential open questions and small next tests</h2>
<ul><li><strong>Is the frozen-layer benefit reproducible?</strong> Finish matched random/scrambled/MLP controls for seeds 1337 and 2024. This is the first priority before claiming a method ranking.</li><li><strong>Does pretraining matter beyond scale and architecture?</strong> Add normalization-reset and same-width frozen-random feature controls; keep data and model-selection rules fixed.</li><li><strong>Where did the robot behavior get learned?</strong> Keep the trained decoder but restore the initial input interface, then perform the reverse swap. Measure completion on the same held-out reset states.</li><li><strong>Can complementary errors become real recovery?</strong> Compare a fixed-budget progress-triggered switch against each individual policy and a random-switch control. Report actual completion; retain the oracle union only as an upper bound.</li><li><strong>Can agents exchange representations?</strong> Swap two learned interfaces around the same frozen block. Fit an alignment on paired training states, with shuffled-pair and simple regression controls; evaluate on fresh physics resets.</li></ul>
<p class="small">All five items are proposed follow-ups. No additional experiment was launched after the stop request.</p>
<h2>Five specific questions for Abay</h2><ol>
<li>Can you share the exact model revision, original robot adapters, preprocessing, and data split so we can isolate why our randomized control can match the pretrained block in this setting?</li>
<li>Does the claimed geometry advantage survive controls that preserve singular values and normalization scales, and how sensitive is it to expert demonstrations versus offline play data?</li>
<li>Have you measured whether separately trained interfaces around one frozen layer can exchange hidden states or action decoders successfully?</li>
<li>For the lithium-neuron checkpoints, can we compare identical examples with signed activations, raw norms, and fixed coordinate identity while separating upstream-input changes from neuron-weight changes?</li>
<li>What signal could predict which controller will fail early enough for a useful handover, given that lower imitation loss did not predict better closed-loop success here?</li></ol>
<h2>A recorded robot rollout</h2><figure><img src="recorded-gemma-policy.gif" alt="Saved MuJoCo replay of the pretrained Gemma controller moving a cube"><figcaption>First predeclared familiar-target test reset, seed 30000; pretrained controller seed 42. Success in 44 control steps. Rendered from saved configurations, with one frame every three steps and illustrative playback timing. This was not selected for a favorable outcome. <a href="recorded-gemma-provenance.json">Provenance</a></figcaption></figure>
<h2>Methods and evidence</h2><details open><summary>What actually ran</summary><p>OGBench two-cube task 1 in MuJoCo, one arm; 16 training demonstrations with noisy execution and clean expert labels, 32 validation demonstrations, 1,000 AdamW steps, batch 128, learning rate 0.0003. The lowest validation loss among steps 0, 100, 250, 500, 750, and 1000 selected the evaluated checkpoint. Familiar targets jitter ±3 cm; shifted targets move 6–10 cm in x. Each condition uses 60 paired reset states and a 150-step horizon. No per-method tuning was performed.</p><p>The 32 validation trajectories are additional data beyond the 16 training demonstrations. The 128-state neuron panel is sampled from those validation trajectories, so its states are correlated and are not independent semantic concepts. Signed gated-MLP activations are compared at fixed neuron coordinates; the surrounding input interface can change immediate neuron inputs despite fixed physical states.</p></details>
<details><summary>Model identity, exactness, and scope</summary><p><a href="https://huggingface.co/google/gemma-4-31B/tree/5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89">Public Gemma 4 31B revision 5bbc2fb1…</a>, layer 26. The extracted block matched the official Transformers implementation exactly in the logged numerical check. Frozen-parameter hashes were unchanged after each completed frozen-model training run.</p><p>The block receives one token and no KV cache, making attention softmax equal to one. Q/K cannot influence this computation; this is not a test of an attention-copy mechanism. The public model is multimodal. We did not test Abay’s original lithium-neuron model/checkpoints, a real robot, multi-arm coordination, a learned router, collision safety, or recursive self-improvement.</p></details>
<details><summary>Stop status and reproducibility</summary><p>The initial run ended after three complete models. A resumable run completed the MLP seed 42 and pretrained seed 1337 before cancellation at your request. Random seed 1337 training was incomplete and has no reported evaluation. Five trained policy adapters, selected-checkpoint metadata, raw completed episode records, and source are preserved.</p><p>Read <a href="README.txt">README.txt</a> before reproducing anything. The archive excludes the approximately 958 MB frozen layer, which the supplied downloader can retrieve. The prepared interface-exchange script is clearly marked NOT_RUN and is not presented as tested code.</p></details>
<footer><p><a href="reproduction.zip">Download the evidence and code</a> · <a href="data/geometry-episodes.json">Raw episode records</a> · <a href="paired-outcomes.csv">Paired outcomes</a> · <a href="neuron-audit.json">Neuron audit</a> · <a href="provenance.json">Run IDs and checksums</a></p><p class="small">Observed measurements, interpretations, and unrun proposals are separated throughout. No further computation is needed to read this report.</p></footer>'''
(R/'report.html').write_text('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Borrowed Geometry: saved robot findings</title><style>'+css+'</style></head><body><main>'+body+'</main></body></html>')
# Remove the unpopulated multi-seed template to avoid presenting planned outcomes as results.
(R/'report-template.html').unlink(missing_ok=True)
with zipfile.ZipFile(R/'reproduction.zip','w',zipfile.ZIP_DEFLATED,compresslevel=1) as z:
    for p in R.rglob('*'):
        if p.is_file() and p.name!='reproduction.zip' and p.name not in ['orx-chart.html','orx_chart.py']:
            z.write(p,p.relative_to(R))
# Lightweight checks only: report links and completed counts.
from html.parser import HTMLParser
class Links(HTMLParser):
    def __init__(self):super().__init__();self.links=[]
    def handle_starttag(self,tag,attrs):
        for k,v in attrs:
            if k in ['href','src'] and not v.startswith(('http','#')):self.links.append(v)
p=Links();p.feed((R/'report.html').read_text())
assert all((R/link).exists() for link in p.links)
assert len(results)==10 and sum(r['n'] for r in results)==600
print(json.dumps({'report':str(R/'report.html'),'completed_rollouts':600,'complete_comparison_training_seeds':1,'local_links_checked':len(p.links),'archive_bytes':(R/'reproduction.zip').stat().st_size,'paired':pairs},indent=2))
