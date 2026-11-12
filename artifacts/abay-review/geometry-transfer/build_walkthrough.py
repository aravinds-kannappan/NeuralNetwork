"""Build a self-contained research walkthrough from saved results. No inference or training."""
from pathlib import Path
import base64
import hashlib
import html
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parent
results=json.loads((ROOT/'data/geometry-results.json').read_text())
episodes=json.loads((ROOT/'data/geometry-episodes.json').read_text())
audit=json.loads((ROOT/'neuron-audit.json').read_text())[0]
curves=json.loads((ROOT/'data/learning-curves.json').read_text())
assert sum(r['n'] for r in results)==600
assert {r['mode'] for r in results if r['training_seed']==42}=={'pretrained','random','scrambled','mlp'}
paired={}
for condition in ['id','shift']:
    p={r['seed']:r for r in episodes if r['training_seed']==42 and r['mode']=='pretrained' and r['condition']==condition}
    r={r['seed']:r for r in episodes if r['training_seed']==42 and r['mode']=='random' and r['condition']==condition}
    assert len(p)==len(r)==60 and p.keys()==r.keys()
    assert all(p[k]['initial_hash']==r[k]['initial_hash'] for k in p)
    paired[condition]=[{'seed':k,'p':{field:p[k][field] for field in ['success','steps','goal_error_m']},'r':{field:r[k][field] for field in ['success','steps','goal_error_m']}} for k in sorted(p)]
def image_data(name,mime):return 'data:'+mime+';base64,'+base64.b64encode((ROOT/name).read_bytes()).decode()
D={'results':results,'paired':paired,'audit':audit,'curves':curves,
   'replayGif':image_data('recorded-gemma-policy.gif','image/gif'),
   'replayStill':image_data('recorded-gemma-final.png','image/png')}
chapters=[('words','·','The terms, on one robot'),('ch0','0','What the robot learned'),('ch1','1','Testing the frozen layer'),('ch2','2','The same score, different failures'),('ch3','3','Following neuron responses'),('ch4','4','From one controller to a team'),('ch5','5','Small tests that would matter'),('ch6','6','Five questions for Abay'),('ch7','7','What we know, and the files')]
toc=''.join(f'<li><a href="#{key}"><span class="number">{num}</span>{title}</a></li>' for key,num,title in chapters)
body='''
<a class="skip" href="#main">Skip to the article</a>
<div class="site">
<nav class="contents" aria-label="Chapters"><p class="contents-label">A robot-control study</p><ol>__TOC__</ol><p class="minor">Saved results only.<br>One complete comparison seed.<br>Five evaluated policies.<br>No experiment runs from this page.</p></nav>
<main class="article" id="main">
<header class="article-header">
<nav class="article-nav" aria-label="Article"><a href="report-first-edition.html">Earlier report</a><a href="#files">The files</a><a href="reproduction.zip" download>Download</a><button type="button" class="plain-button" id="theme-toggle" aria-pressed="false">Dark theme</button></nav>
<h1>One frozen layer,<br>five robot policies</h1>
<p class="lede">We trained a simulated robot arm to move a cube using a fixed piece of a pretrained neural network. Then we changed that piece to ask what was actually helping.</p>
<p>The starting point was Abay’s <a href="https://www.alphaxiv.org/pdf/2605.00333">Borrowed Geometry</a>: can a network reuse structure learned before it ever sees the new task? Your longer-term question goes further: could several specialists inside a robot share useful information and cover for one another when something goes wrong?</p>
<p>This page follows the completed tests, explains what each measurement means, and separates observations from possible explanations. Three observations guide the story:</p>
<ol class="opening-list"><li>The original pretrained layer and a control with randomized linear weights achieved the same success rate in the one complete comparison seed.</li><li>Those two controllers succeeded on different attempts. A shared score hid potentially useful differences.</li><li>The pretrained controller’s neuron response patterns stayed very similar during training on a fixed panel of robot states.</li></ol>
<p class="status-note"><strong>The study stopped early.</strong> Five policies completed 600 evaluation episodes. Only one training seed has all four comparison methods. The remaining compute was cancelled at your request. The proposed team and alignment tests were not run. <a href="provenance.json">Run record</a></p>
<p class="reading-guide"><strong>New to the topic?</strong> Start with the terms below; each chapter explains the question before the measurement. <strong>Already familiar?</strong> Use the sidebar, data links, and expandable method notes. <button class="plain-button" id="technical-toggle" type="button" aria-expanded="false">Open technical notes</button></p>
<p class="dateline">A walkthrough of saved local experiments · November 2026. Written with an AI research assistant. This is an exploratory study, not an independently reviewed replication.</p>
</header>

<section class="chapter" id="words">
<h2><span class="number">·</span>The terms, on one robot</h2>
<p>Imagine the arm at the start of an attempt. It has to pick up a cube and move it to a target. These seven terms describe the pieces involved.</p>
<p><strong>Controller, or policy.</strong> A rule that turns what the robot knows now into its next action. Our learned controller reads numbers describing the arm, cube, and target, then produces movement and gripper commands.</p>
<p><strong>Representation.</strong> An internal vector—a list of numbers—that the network uses to process a situation. Two controllers can describe the same physical situation using different internal vectors. Sharing the same vector length does not establish that the vectors mean the same thing.</p>
<p><strong>Frozen layer and learned interface.</strong> “Frozen” means we keep that part’s numerical weights fixed. We train a small input interface to put robot measurements into the layer, and an output decoder to turn its response into actions. The interfaces can learn even though the middle part stays fixed.</p>
<p><strong>Demonstration.</strong> An example of how to act. A scripted expert generated robot trajectories. We trained the controller to imitate the expert’s commands. This procedure is called <em>behavior cloning</em>.</p>
<p><strong>Training seed and test episode.</strong> A training seed sets the random choices used to train one controller. A test episode is one attempt by that controller to move the cube. Sixty test episodes from one trained controller are not sixty independently trained controllers.</p>
<p><strong>Checkpoint.</strong> The saved weights at a particular training step. We selected a checkpoint using separate validation examples, then tested it on robot episodes that were not used for training or selection.</p>
<p><strong>Control.</strong> A comparison designed to test an explanation. For example, if a randomized layer works as well as the original one, we must ask whether the original pretrained matrix structure was responsible for the score.</p>
<aside class="aside-note"><p><strong>What does “geometry” mean here?</strong> How a network arranges and transforms vectors of numbers. The research question is whether structure from earlier learning helps a new task. It does not simply mean the physical shape of the robot arm.</p></aside>
<p class="caption-note">The definitions above describe this experiment. Exact input fields and model operations are in <a href="source/borrowed_recovery/geometry.py">the executed code</a>.</p>
</section>

<section class="chapter" id="ch0">
<h2><span class="number">0</span>What the robot learned</h2>
<p><strong>The task:</strong> move a cube to a target in a physics simulator. Each control step starts with a measurement of the current situation and ends with an action. The robot gets another measurement after it moves, and repeats the process.</p>
<p>We borrowed one internal block, layer 26, from the public Gemma 4 31B model. We kept its weights fixed and trained the small interfaces around it. The diagram shows the computation that actually ran. Select a part to see its job.</p>
<figure class="figure">
<div class="figure-panel">
<p class="figure-title">One robot-control step</p><p class="figure-subtitle">The middle block is fixed; the input interface and action decoder learn.</p>
<div class="flow" role="group" aria-label="Parts of the robot controller">
<button type="button" data-flow="state" aria-pressed="true">Robot state<span class="flow-kind">21 measurements</span></button>
<button type="button" data-flow="encoder" class="learned" aria-pressed="false">Input interface<span class="flow-kind">learned</span></button>
<button type="button" data-flow="frozen" aria-pressed="false">Gemma layer<span class="flow-kind">frozen</span></button>
<button type="button" data-flow="decoder" class="learned" aria-pressed="false">Action decoder<span class="flow-kind">learned</span></button>
<button type="button" data-flow="robot" aria-pressed="false">Robot moves<span class="flow-kind">next state</span></button>
</div>
<div class="readout" id="flow-readout" aria-live="polite"><strong>Robot state.</strong> The simulator supplies 21 numbers about the arm, gripper, cube, and target. The controller is not interpreting camera images.</div>
</div>
<figcaption>Figure 1. The learned interfaces connect robot measurements to a fixed Gemma block and back to actions. This is an architecture diagram, not a claim that a whole language model is reasoning about the robot. <a class="data-link" href="source/borrowed_recovery/geometry.py">code</a></figcaption>
</figure>
<p>We used 16 training demonstrations and 32 separate validation demonstrations. Each evaluated controller got 60 attempts at familiar targets and 60 at shifted targets. Familiar targets stay near the demonstrated region; shifted targets move farther sideways.</p>
<details class="technical"><summary>Technical detail: input features, optimization, and checkpoint selection</summary>
<p>The observation is a 21-dimensional privileged state vector: joint positions, effector position and yaw encoding, gripper opening/contact, cube position and yaw encoding, and target position. Features are normalized with training-set means and standard deviations, with a standard-deviation floor of 0.05. There is no camera perception model.</p>
<p>The policy uses a 21→5,376 linear encoder, affine LayerNorm, the frozen block, non-affine output LayerNorm, and a 5,376→5 linear decoder. The loss is continuous-action MSE plus 0.25 times binary cross entropy for gripper direction.</p>
<p>AdamW ran for 1,000 steps at learning rate 0.0003, batch size 128, weight decay 0.0001, and gradient clipping 1.0. Validation checkpoints were steps 0, 100, 250, 500, 750, and 1000. The minimum validation loss selected the evaluated policy. Test success did not select the checkpoint.</p>
<p>Training resets: 1000–1015. Validation resets: 2000–2031. Familiar test resets: 30000–30059. Shifted test resets: 40000–40059. The additional validation trajectories mean this is not a strict 16-episode total-data budget. <a href="source/config.json">Configuration</a> · <a href="source/GEOMETRY_PROTOCOL.md">Protocol</a></p>
</details>
<details class="technical"><summary>Technical detail: which Gemma computation was verified?</summary>
<p>The pinned model is <a href="https://huggingface.co/google/gemma-4-31B/tree/5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89">google/gemma-4-31B revision 5bbc2fb1…</a>, layer 26. The extracted block matched the official implementation on the logged three-input numerical check. Frozen parameter hashes were unchanged for each completed frozen-model training run.</p>
<p>The input contains one state token and no past-token cache. Attention has only one token to attend to, so its softmax is one and Q/K cannot affect the output. This is not a test of an attention-copy mechanism. The public model revision is multimodal; these experiments cannot attribute robot control to text-only pretraining. <a href="source/borrowed_recovery/gemma_weights.py">Extraction and numerical check</a></p>
</details>
</section>

<section class="chapter" id="ch1">
<h2><span class="number">1</span>Testing the frozen layer</h2>
<p><strong>The question:</strong> does the original pretrained matrix structure help more than a changed version of the same block?</p>
<p>We compared four controllers. The pretrained version uses the original matrices. A randomized version replaces their entries while matching their average and spread. A scrambled version rearranges their rows and columns. A small conventional neural network provides another baseline.</p>
<p><span class="tag">Observed</span>The original and randomized versions tied in the complete comparison seed. Both completed half of the familiar-target episodes, and far fewer shifted-target episodes. Switch the target condition below to see the difference.</p>
<figure class="figure" id="comparison">
<div class="figure-panel">
<p class="figure-title">Completed attempts, out of 60</p><p class="figure-subtitle">One matched training seed: 42. All four policies start from the same test configurations.</p>
<div class="controls"><div class="segmented" role="group" aria-label="Target condition for success comparison"><button type="button" data-group="scores" data-value="id" aria-pressed="true">Familiar targets</button><button type="button" data-group="scores" data-value="shift" aria-pressed="false">Shifted targets</button></div><label class="option"><input type="checkbox" id="show-intervals">Episode uncertainty</label></div>
<div id="success-bars" class="bars" aria-live="polite"></div>
<div class="scale" aria-hidden="true"><span>0</span><span>15</span><span>30</span><span>45</span><span>60</span></div><p class="axis-label">Successful episodes</p>
<p id="success-note" class="dynamic-note"></p>
<p id="interval-note" class="dynamic-note" hidden>Whiskers are 95% Wilson intervals over reset episodes for a fixed trained policy. They do not measure variation across training seeds or uncertainty in a difference between methods.</p>
<noscript><p class="no-js">JavaScript is disabled. Familiar-target successes were 30, 30, 26, and 11 out of 60 for pretrained, randomized, scrambled, and small-network policies. Shifted-target successes were 9, 9, 8, and 9. <a href="success-by-seed.csv">All measurements</a>.</p></noscript>
</div>
<figcaption>Figure 2. A single matched comparison does not establish a pretrained-weight advantage. Matching scores also does not prove that the two methods are equivalent. The three-seed comparison was not completed. <a class="data-link" href="data/geometry-results.json">data</a></figcaption>
</figure>
<p><span class="tag">Interpretation</span>A large fixed collection of transformations may help this low-data task even when its original matrix structure is disrupted. That is an explanation to investigate, not a mechanism we have isolated.</p>
<p>The second pretrained controller, trained with seed 1337, completed 25/60 familiar-target and 8/60 shifted-target episodes. Its comparison controls did not finish. We keep it separate rather than averaging two pretrained seeds against one control seed. <a href="success-by-seed.csv">Per-seed results</a></p>
<details class="technical"><summary>Technical detail: what the controls preserve</summary>
<p><strong>Randomized linear matrices.</strong> Each matrix is replaced with Gaussian weights matched to its original mean and standard deviation. Pretrained normalization vectors and the layer scalar remain. This is not a completely random model.</p>
<p><strong>Scrambled matrices.</strong> Independent row and column permutations preserve each matrix’s singular values, element values, and norms while disrupting how directions align between matrices. Normalization vectors and the layer scalar remain unchanged.</p>
<p><strong>Small MLP.</strong> The fully trained 21→381→381→5 network has 155,834 trainable parameters. Frozen-layer policies have 155,909 trainable parameters plus 412,898,305 frozen parameters. We approximately match trainable counts, not total capacity or compute. Shared hyperparameters were not tuned separately for each method. <a href="source/borrowed_recovery/geometry.py">Executed controls</a></p>
</details>
<aside class="aside-note"><p><strong>Why the original paper’s score is not the comparison here.</strong> <a href="https://www.alphaxiv.org/pdf/2605.00333">Borrowed Geometry</a> reports a different robotics setup using offline reinforcement learning on play data. We used imitation of task-specific demonstrations, different interfaces, and a public model revision that does not exactly match the paper’s stated dimensions. We could not retrieve the original robot implementation or trained adapters. These results extend the question; they do not reproduce or refute that result.</p></aside>
</section>

<section class="chapter" id="ch2">
<h2><span class="number">2</span>The same score, different failures</h2>
<p><strong>The question:</strong> if two controllers succeed equally often, do they solve the same attempts?</p>
<p>This matters for your robot-team idea. A backup can be useful when it handles situations the first controller cannot. Looking only at the overall score hides that difference.</p>
<p><span class="tag">Observed</span>On familiar targets, both controllers succeeded on 18 episodes. The pretrained controller uniquely solved 12 more, and the randomized controller uniquely solved another 12. The grid shows the actual paired attempts. Select a square to inspect it.</p>
<figure class="figure">
<div class="figure-panel">
<p class="figure-title">Which controller solved each attempt?</p><p class="figure-subtitle">One square is one shared test reset, ordered by reset seed. P = pretrained; R = randomized.</p>
<div class="controls"><div class="segmented" role="group" aria-label="Target condition for paired episodes"><button type="button" data-group="episodes" data-value="id" aria-pressed="true">Familiar targets</button><button type="button" data-group="episodes" data-value="shift" aria-pressed="false">Shifted targets</button></div></div>
<div class="legend"><span><b class="swatch both">B</b>Both</span><span><b class="swatch onlyp">P</b>Pretrained only</span><span><b class="swatch onlyr">R</b>Randomized only</span><span><b class="swatch neither">–</b>Neither</span></div>
<div class="episode-grid" id="episode-grid" role="group" aria-label="Sixty paired robot episodes"></div>
<div class="counts"><span>Both <strong id="count-both">18</strong></span><span>P only <strong id="count-onlyp">12</strong></span><span>R only <strong id="count-onlyr">12</strong></span><span>Neither <strong id="count-neither">18</strong></span></div>
<div id="episode-readout" class="readout" aria-live="polite"></div><p id="union-note" class="dynamic-note"></p>
</div>
<figcaption>Figure 3. Complementary failures create a reason to investigate selecting between specialists. They do not demonstrate a working selector or safe mid-action takeover. Each pair’s initial physical-state hash matches. <a class="data-link" href="data/geometry-episodes.json">episode data</a> <a class="data-link" href="paired-outcomes.csv">paired counts</a></figcaption>
</figure>
<p><span class="tag">Interpretation</span>At least one controller succeeded on 42/60 familiar-target episodes. A perfect selector with advance knowledge could choose those successes. A real system would have to detect trouble early, choose a better controller, and handle the state left behind by the first one. We have not tested any of those steps.</p>
<details class="technical"><summary>Technical detail: why the union is not an ensemble result</summary>
<p>The union is computed after observing both complete rollouts from the same initial state. A switched controller would usually take over from a different, partly progressed state. It also needs a selection rule without access to the future outcome. Therefore, 42/60 is an oracle selection upper bound for these separately evaluated rollouts, not a measured recovery rate.</p>
<p>All these policies share one training dataset. Reusing reset seeds improves pairing, but does not create independent training replications. The episode grid is descriptive evidence from training seed 42. <a href="data/geometry-episodes.json">Saved rollouts</a></p>
</details>
</section>

<section class="chapter" id="ch3">
<h2><span class="number">3</span>Following neuron responses</h2>
<p><strong>The question:</strong> how much do a neuron’s responses change while the surrounding robot interface is trained?</p>
<p>A neuron produces a number for each input. We took the same 128 robot states at different checkpoints and collected each neuron’s 128 responses. Comparing those lists gives us a response “fingerprint.” A cosine close to 1 means the lists point in nearly the same direction.</p>
<p>Direction is only part of the story. If every response doubled, cosine could stay at 1. We therefore also compared response magnitude, and used a mean-centered cosine to reduce the effect of a common offset.</p>
<figure class="figure">
<div class="figure-panel">
<p class="figure-title">The same states, the same neuron coordinates</p><p class="figure-subtitle">Pretrained controller, training seed 42 · 128 fixed validation states · 21,504 signed MLP neuron responses.</p>
<div class="controls"><div class="segmented" role="group" aria-label="Neuron checkpoint comparison"><button type="button" data-group="audit" data-value="100" aria-pressed="true">Evaluated checkpoint · 100</button><button type="button" data-group="audit" data-value="1000" aria-pressed="false">Last checkpoint · 1000</button></div></div>
<div class="audit-numbers"><div><span class="metric-name">Median raw cosine</span><output id="audit-raw">0.99802</output><small>Direction of each neuron’s response vector.</small></div><div><span class="metric-name">Median centered cosine</span><output id="audit-centered">0.99797</output><small>Direction after removing each vector’s mean.</small></div><div><span class="metric-name">Median response-norm ratio</span><output id="audit-norm">0.99839×</output><small>1× means the same overall response magnitude.</small></div></div>
<p id="audit-note" class="dynamic-note"></p><p id="audit-range" class="dynamic-note"></p>
</div>
<figcaption>Figure 4. Response patterns were very stable at the checkpoint used for the robot evaluation. These are medians over neurons, not proof that every neuron was unchanged or retained a particular semantic role. <a class="data-link" href="neuron-audit.json">data</a> <a class="data-link" href="fingerprints-pretrained-s42-neuron-audit.csv">individual neurons</a></figcaption>
</figure>
<p><span class="tag">Observed</span>The median centered cosine from initialization to the selected checkpoint was about 0.998. The frozen block’s parameter hash was unchanged. At the same time, the surrounding interfaces had been trained and the resulting controller completed robot tasks.</p>
<p><span class="tag">Interpretation</span>This is consistent with learning to use existing features. It does not establish that the decoder alone caused the successful behavior. Restoring the initial encoder while keeping the trained decoder would help test that explanation; we did not run that intervention.</p>
<aside class="aside-note"><p><strong>Connection to Abay’s neuron study.</strong> A changing response does not, by itself, tell us which part of a network changed or whether a neuron gained a new semantic role. Identity, inputs, response scale, and behavior need separate checks. Our measurements are from Gemma robot controllers, not the lithium neuron or Abay’s original checkpoints. <a href="https://abaybektursun.github.io/already-clean-audit/#words">His definitions and checkpoint study</a></p></aside>
<details class="technical"><summary>Technical detail: what exactly was compared?</summary>
<p>For neuron j, the response vector contains its signed gated-MLP activation on each of the same 128 validation states. Raw cosine compares those vectors directly. Centered cosine compares each after subtracting its own mean; here it is Pearson correlation across states. The norm ratio is ‖a<sub>later</sub>‖₂ / ‖a<sub>initial</sub>‖₂. We do not take a cosine between two scalar activations.</p>
<p>Neuron indices refer to fixed coordinates in the same extracted block. No positive/negative clipping or cross-model neuron matching is used. Hashes verify unchanged block parameters, but the learned encoder and input normalization can change the immediate inputs to that block even when physical states are held fixed.</p>
<p>The panel states come from validation trajectories and are correlated. We did not assign semantic labels to the neurons. Step 100 was selected by validation loss; step 1000 had worse validation loss and was not used for the reported robot scores. <a href="analyze_results.py">Audit code</a> · <a href="data/learning-curves.json">Checkpoint losses</a></p>
</details>
<h3>Prediction loss and behavior gave different rankings</h3>
<p>The small network’s selected validation imitation loss was lower than pretrained Gemma’s: 0.15258 versus 0.17830. Yet its familiar-target success was also lower: 11/60 versus 30/60. Predicting expert commands on saved states and controlling a moving robot are different evaluations. This single comparison illustrates why the behavioral test matters. <a href="learning-curves.csv">Loss records</a> · <a href="success-by-seed.csv">Success records</a></p>
</section>

<section class="chapter" id="ch4">
<h2><span class="number">4</span>From one controller to a team</h2>
<p>The work so far contains one simulated arm and separately evaluated controllers. The connection to your project is a testable next step: build specialists whose strengths differ, then measure whether they can exchange useful information and recover from each other’s failures.</p>
<p>One specialist might handle grasping; another might assess obstacles; a third might decide whether an action should continue. A shared layer or memory would be a component of that design. It would not, on its own, show that the specialists understand one another’s internal representations.</p>
<p><strong>A proposed compatibility test:</strong> take two independently trained input interfaces around the same frozen layer. Feed the second interface’s representation to the first controller’s decoder. Measure whether control still works. Then fit an alignment using paired examples of the same robot states, and test whether that repairs any lost performance.</p>
<aside class="aside-note"><p><strong>Prepared, not executed.</strong> The interface-exchange and alignment experiment exists as a draft script. It has no experimental results. We also did not train a mixture-of-experts router, evaluate multiple arms, or test collision safety. <a href="source/interface_exchange_NOT_RUN.py">Unrun draft</a></p></aside>
<details class="technical"><summary>Technical detail: alignment, expert routing, and self-improvement</summary>
<p>The draft uses controllers with training seeds 42 and 1337, a fixed shared pretrained block, and 512 paired training states. It proposes rank-64 scaled Procrustes alignment, a shuffled-pair negative control, an own-controller rank-projection control, and a ridge-regression baseline. Fresh closed-loop evaluations would test whether alignment improves behavior, not only similarity.</p>
<p>A mixture-of-experts connection would be a gating or routing mechanism that chooses between specialists. In the proposed robot system, that choice must be evaluated under a shared action/time budget and realistic partial failures. The current oracle union is not such a router.</p>
<p>Sharing information or adapting during a task is not recursive self-improvement. That longer-term claim would require a system that helps change its own capabilities and demonstrates durable improvement under independent evaluation. No such loop was implemented here.</p>
</details>
<h3>One saved rollout</h3>
<p>This replay shows what the learned controller actually did on the first predeclared familiar-target test reset. It was not selected because it looked good. The arm completed that episode in 44 control steps.</p>
<figure class="figure">
<div class="figure-panel"><img id="recorded-replay" class="replay-image" src="__STILL__" width="600" height="450" alt="Saved final frame of the Gemma-controlled robot arm moving a cube in MuJoCo"><div class="replay-row"><button type="button" id="replay-toggle" class="plain-button" aria-pressed="false">Play saved rollout</button><span id="replay-status" class="caption-note">Saved final frame. Playback uses previously recorded states.</span></div></div>
<figcaption>Figure 5. Pretrained controller, training seed 42, test reset 30000. Frames were recorded every three control steps. Playback timing is illustrative; playing this image does not run the model or simulator. <a class="data-link" href="recorded-gemma-provenance.json">record</a></figcaption>
</figure>
</section>

<section class="chapter" id="ch5">
<h2><span class="number">5</span>Small tests that would matter</h2>
<p>These are proposed experiments, ordered by what they could change about our interpretation. None was launched after the stop request.</p>
<ol class="tests">
<li><strong>Finish the matched seeds.</strong> Complete the missing randomized, scrambled, and small-network runs for the remaining seeds. This tests whether the observed pattern is stable across training randomness. Repeating test episodes alone cannot answer that question.</li>
<li><strong>Separate fixed features from pretrained structure.</strong> Add controls that reset the normalization scales and compare fixed random features at the same width. If the advantage survives those changes, a more general feature-bank explanation becomes more plausible.</li>
<li><strong>Swap the learned components.</strong> Keep the trained decoder and restore the initial input interface; then do the reverse. Evaluate the same held-out physical resets. This tests which learned component the behavior depends on.</li>
<li><strong>Turn complementary failures into a real handover.</strong> Compare a progress-triggered switch with each individual controller and a random-switch control, using the same total action budget. Measure actual completion from the state at handover.</li>
<li><strong>Test representation exchange before building a larger team.</strong> Compare direct interface exchange, paired-state alignment, shuffled alignment, and simple regression on fresh robot resets. Better cosine or action prediction alone would not establish better coordination.</li>
</ol>
</section>

<section class="chapter" id="ch6">
<h2><span class="number">6</span>Five questions for Abay</h2>
<ol class="question-list">
<li>Can you share the exact model revision, original robot adapters, input preprocessing, and data split so we can isolate why our randomized control can match the pretrained block in this setting?</li>
<li>Does the claimed geometry advantage survive controls that preserve matrix singular values and normalization scales, and how sensitive is it to expert demonstrations versus offline play data?</li>
<li>Have separately trained interfaces around one frozen layer been tested for exchanging hidden representations or action decoders, and does an alignment repair any failure?</li>
<li>For the lithium-neuron checkpoints, can we compare identical examples with signed activations, raw norms, and fixed coordinate identity while separating upstream-input changes from neuron-weight changes?</li>
<li>What signal might predict failure early enough for a useful handover, given that lower imitation loss did not predict better closed-loop success in our completed comparison?</li>
</ol>
</section>

<section class="chapter" id="ch7">
<h2><span class="number">7</span>What we know, and the files</h2>
<p>We have working learned controllers and controlled comparisons on recorded physical resets. The evidence supports more precise next questions. It does not yet establish that pretrained geometry is the cause of the observed control performance, or that a team of agents can recover from faults.</p>
<div class="table-scroll"><table class="status-table"><thead><tr><th>Evidence</th><th>What is complete</th><th>Limit</th></tr></thead><tbody><tr><td>Four-method comparison</td><td>Training seed 42</td><td>One complete training replicate</td></tr><tr><td>Additional pretrained policy</td><td>Training seed 1337</td><td>Matched controls incomplete</td></tr><tr><td>Fixed-state neuron audit</td><td>Pretrained seed 42</td><td>No semantic role assignment or causal swap</td></tr><tr><td>Interface exchange and alignment</td><td>Draft only</td><td>No measured outcome</td></tr><tr><td>Robot-team recovery</td><td>Proposed next study</td><td>No trained router or multi-arm test</td></tr></tbody></table></div>
<details class="technical"><summary>Technical detail: feasibility baseline, uncertainty, and stop history</summary>
<p>An earlier small-network feasibility run used 256 demonstrations and 2,000 optimization steps, achieving 39/40 successes in each condition. It used different test reset seeds and more data/training. It establishes that this task is learnable, not a matched sample-efficiency comparison. Its run ID is recorded in the provenance file.</p>
<p>The first comparison run stopped after three complete models. A resumable run completed the small network for seed 42 and pretrained Gemma for seed 1337, then was cancelled at the user’s request. The random seed-1337 training was incomplete and has no reported evaluation.</p>
<p>We retain completed model outcomes from interrupted runs, but do not present the sweep as completed. All policies share one demonstration dataset; training-seed and episode-seed variation are different sources of uncertainty. The simulator supplies privileged state, so no claim about camera perception or physical-robot transfer follows.</p>
<p><a href="provenance.json">Run IDs, source commits, model revision, and file checksums</a></p>
</details>
<h3 id="files">Read the evidence without running anything</h3>
<ul class="files"><li><a href="reproduction.zip" download>Code, data, and trained adapters</a></li><li><a href="success-by-seed.csv">Success counts by seed</a></li><li><a href="data/geometry-episodes.json">Every completed episode</a></li><li><a href="paired-outcomes.csv">Paired success and failure counts</a></li><li><a href="learning-curves.csv">Checkpoint validation losses</a></li><li><a href="neuron-audit.json">Neuron response audit</a></li><li><a href="source/GEOMETRY_PROTOCOL.md">Experimental protocol</a></li><li><a href="source/borrowed_recovery/geometry.py">Executed robot-policy code</a></li><li><a href="provenance.json">Experiment provenance</a></li><li><a href="README.txt">Reproduction notes</a></li></ul>
<p class="caption-note">This page embeds its measurements, interface code, and replay. It works offline. The linked downloads and source files are included in the accompanying archive; none of the page’s controls starts training or simulation.</p>
</section>
<footer><p>Presentation inspired by <a href="https://abaybektursun.github.io/already-clean-audit/#words">Abay’s checkpoint walkthrough</a>: terms first, chapters, interactive evidence, and links to the underlying files. The text, implementation, and experiments here are separate.</p><p>Related research: <a href="https://www.alphaxiv.org/pdf/2605.00333">Borrowed Geometry</a> · <a href="https://github.com/abaybektursun/already-clean-audit">Already Clean audit repository</a>. The earlier report remains available <a href="report-first-edition.html">unchanged</a>.</p></footer>
</main></div>
'''
body=body.replace('__TOC__',toc).replace('__STILL__',D['replayStill'])
css=(ROOT/'walkthrough.css').read_text();js=(ROOT/'walkthrough.js').read_text()
payload=json.dumps(D,separators=(',',':')).replace('<','\\u003c')
page='<!doctype html><html lang="en" data-theme="light"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta name="description" content="A beginner-readable, technically documented walkthrough of saved robot-control experiments inspired by Borrowed Geometry."><title>One frozen layer, five robot policies</title><style>'+css+'</style></head><body>'+body+'<script id="saved-data" type="application/json">'+payload+'</script><script>'+js+'</script></body></html>'
if not (ROOT/'report-first-edition.html').exists():shutil.copy2(ROOT/'report.html',ROOT/'report-first-edition.html')
(ROOT/'report.html').write_text(page)
manifest={'change':'presentation and explanation only; no experiments launched',
          'reference':'https://abaybektursun.github.io/already-clean-audit/#words',
          'completed_policy_rollouts':600,'complete_comparison_training_seeds':[42],
          'inputs':{str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'data/geometry-results.json',ROOT/'data/geometry-episodes.json',ROOT/'neuron-audit.json']},
          'html_bytes':len(page.encode()),'builder':'build_walkthrough.py'}
(ROOT/'presentation-manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(manifest,indent=2))
