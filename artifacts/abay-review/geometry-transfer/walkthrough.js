'use strict';
(() => {
  const D = JSON.parse(document.getElementById('saved-data').textContent);
  const names = {pretrained:'Pretrained Gemma',random:'Randomized matrices',scrambled:'Scrambled matrices',mlp:'Small trained network'};
  const modes = Object.keys(names);
  const pct = n => (100*n).toFixed(1)+'%';
  let scoreCondition='id', episodeCondition='id', selectedEpisode=null;
  function pressed(group,value){document.querySelectorAll(`[data-group="${group}"]`).forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.value===value)));}
  function scores(){
    const rows=D.results.filter(r=>r.training_seed===42 && r.condition===scoreCondition);
    const target=document.getElementById('success-bars');target.replaceChildren();
    modes.forEach(mode=>{
      const r=rows.find(r=>r.mode===mode);
      const row=document.createElement('div');row.className='bar-row '+mode;
      const label=document.createElement('div');label.className='bar-label';
      const name=document.createElement('span');name.textContent=names[mode];
      const value=document.createElement('output');value.textContent=`${r.successes}/${r.n} · ${pct(r.success_rate)}`;
      label.append(name,value);
      const track=document.createElement('div');track.className='bar-track';track.setAttribute('role','img');
      track.setAttribute('aria-label',`${names[mode]}: ${r.successes} successes out of ${r.n}. Episode-level 95 percent Wilson interval ${pct(r.wilson95[0])} to ${pct(r.wilson95[1])}. One training seed.`);
      const fill=document.createElement('div');fill.className='bar-fill';fill.style.width=100*r.success_rate+'%';
      const whisker=document.createElement('div');whisker.className='whisker';whisker.style.left=100*r.wilson95[0]+'%';whisker.style.width=100*(r.wilson95[1]-r.wilson95[0])+'%';
      track.append(fill,whisker);row.append(label,track);target.append(row);
    });
    const note=scoreCondition==='id'?'Familiar goals: targets vary within the region used for demonstrations.':'Shifted goals: targets move farther sideways than in the demonstrations.';
    document.getElementById('success-note').textContent=note+' The four rows are matched at training seed 42.';
    pressed('scores',scoreCondition);
  }
  document.querySelectorAll('[data-group="scores"]').forEach(b=>b.addEventListener('click',()=>{scoreCondition=b.dataset.value;scores();}));
  document.getElementById('show-intervals').addEventListener('change',e=>{
    document.getElementById('success-bars').classList.toggle('intervals',e.target.checked);
    document.getElementById('interval-note').hidden=!e.target.checked;
  });
  const flows={
    state:['Robot state','The simulator supplies 21 numbers about the arm, gripper, cube, and target. These measurements are available directly; the controller is not interpreting camera images.'],
    encoder:['Input interface · learned','A small trainable network converts those 21 measurements into a representation with 5,376 numbers. Learning changes how robot states enter the frozen layer.'],
    frozen:['Gemma layer · fixed','One internal block from the pretrained model transforms that representation. Its weights are held fixed. The controls replace its linear matrices while keeping its shape and normalization scales.'],
    decoder:['Action decoder · learned','A trainable output layer converts the representation into five action outputs: four continuous controls and a gripper command. This is one possible place where new robot behavior can be learned.'],
    robot:['Robot and next state','The simulator executes the command, then supplies the next state. Success is measured by completing the cube-moving task within 150 control steps.']
  };
  document.querySelectorAll('[data-flow]').forEach(b=>b.addEventListener('click',()=>{
    document.querySelectorAll('[data-flow]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));
    const [title,description]=flows[b.dataset.flow];const r=document.getElementById('flow-readout');r.replaceChildren();
    const strong=document.createElement('strong');strong.textContent=title+'. ';r.append(strong,document.createTextNode(description));
  }));
  function episodes(){
    const rows=D.paired[episodeCondition],grid=document.getElementById('episode-grid');grid.replaceChildren();
    const counts={both:0,onlyp:0,onlyr:0,neither:0};
    if(selectedEpisode===null || !rows.some(r=>r.seed===selectedEpisode))selectedEpisode=rows[0].seed;
    rows.forEach(r=>{
      const type=r.p.success?(r.r.success?'both':'onlyp'):(r.r.success?'onlyr':'neither');counts[type]++;
      const marker={both:'B',onlyp:'P',onlyr:'R',neither:'–'}[type];
      const button=document.createElement('button');button.type='button';button.className='episode '+type;button.textContent=marker;
      button.dataset.seed=r.seed;button.setAttribute('aria-pressed',String(r.seed===selectedEpisode));
      const description=`Reset ${r.seed}. Pretrained ${r.p.success?'succeeded':'failed'}; randomized ${r.r.success?'succeeded':'failed'}.`;
      button.setAttribute('aria-label',description);button.title=description;
      button.addEventListener('click',()=>{selectedEpisode=r.seed;document.querySelectorAll('.episode').forEach(x=>x.setAttribute('aria-pressed',String(Number(x.dataset.seed)===selectedEpisode)));episodeReadout(r);});
      grid.append(button);
    });
    Object.entries(counts).forEach(([key,value])=>document.getElementById('count-'+key).textContent=value);
    const union=counts.both+counts.onlyp+counts.onlyr;
    document.getElementById('union-note').textContent=`At least one controller succeeded on ${union}/60 episodes (${pct(union/60)}). This is a hindsight upper bound for a perfect selector, not an achieved team score.`;
    episodeReadout(rows.find(r=>r.seed===selectedEpisode));pressed('episodes',episodeCondition);
  }
  function episodeReadout(r){
    const e=document.getElementById('episode-readout');e.replaceChildren();
    const title=document.createElement('strong');title.textContent=`Reset ${r.seed}. `;
    e.append(title,document.createTextNode(`Pretrained: ${r.p.success?'success':'failure'} after ${r.p.steps} steps. Randomized: ${r.r.success?'success':'failure'} after ${r.r.steps} steps. Both started from the same recorded physical configuration.`));
  }
  document.querySelectorAll('[data-group="episodes"]').forEach(b=>b.addEventListener('click',()=>{episodeCondition=b.dataset.value;selectedEpisode=null;episodes();}));
  function audit(step){
    const a=D.audit[step];pressed('audit',step);
    document.getElementById('audit-raw').textContent=a.raw_cosine_median.toFixed(5);
    document.getElementById('audit-centered').textContent=a.centered_cosine_median.toFixed(5);
    document.getElementById('audit-norm').textContent=a.response_norm_ratio_median.toFixed(5)+'×';
    document.getElementById('audit-range').textContent=`Middle 80% of neuron norm ratios: ${a.response_norm_ratio_p10_p90[0].toFixed(3)}–${a.response_norm_ratio_p10_p90[1].toFixed(3)}×.`;
    document.getElementById('audit-note').textContent=step==='100'?'Initialization → step 100, the checkpoint selected for this controller’s robot evaluation. The median response patterns changed very little.':'Initialization → step 1000, the last saved training checkpoint. Its validation loss was worse, so it was not used for the reported robot success scores.';
  }
  document.querySelectorAll('[data-group="audit"]').forEach(b=>b.addEventListener('click',()=>audit(b.dataset.value)));
  let playing=false;
  document.getElementById('replay-toggle').addEventListener('click',e=>{
    playing=!playing;document.getElementById('recorded-replay').src=playing?D.replayGif:D.replayStill;
    e.currentTarget.textContent=playing?'Stop replay':'Play saved rollout';e.currentTarget.setAttribute('aria-pressed',String(playing));
    document.getElementById('replay-status').textContent=playing?'Playing saved frames; no simulation is running.':'Saved final frame. Playback uses previously recorded states.';
  });
  const theme=document.getElementById('theme-toggle');
  theme.addEventListener('click',()=>{
    const dark=document.documentElement.dataset.theme!=='dark';document.documentElement.dataset.theme=dark?'dark':'light';theme.textContent=dark?'Light theme':'Dark theme';theme.setAttribute('aria-pressed',String(dark));
  });
  const tech=document.getElementById('technical-toggle');
  tech.addEventListener('click',()=>{
    const open=tech.getAttribute('aria-expanded')!=='true';document.querySelectorAll('details.technical').forEach(x=>x.open=open);
    tech.setAttribute('aria-expanded',String(open));tech.textContent=open?'Close technical notes':'Open technical notes';
  });
  const observer=new IntersectionObserver(entries=>{entries.forEach(e=>{if(e.isIntersecting){document.querySelectorAll('.contents a').forEach(a=>a.removeAttribute('aria-current'));const link=document.querySelector(`.contents a[href="#${e.target.id}"]`);if(link)link.setAttribute('aria-current','true');}});},{rootMargin:'-5% 0px -70% 0px',threshold:0});
  document.querySelectorAll('section.chapter').forEach(s=>observer.observe(s));
  scores();episodes();audit('100');
})();
