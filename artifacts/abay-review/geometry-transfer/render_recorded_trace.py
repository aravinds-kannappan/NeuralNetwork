"""Render recorded MuJoCo configurations; does not execute a policy or collect results."""
import json
from pathlib import Path
import mujoco
import numpy as np
from PIL import Image,ImageDraw
from borrowed_recovery.geometry import reset
import ogbench

HERE=Path(__file__).resolve().parent
SOURCE=Path('/Users/aravindkannappan/.local/share/openresearch/local-runs/ed2527c3-cc8f-4df8-a7ee-3a68cbdb6e5d/repo/outputs')
trace=json.loads((SOURCE/'trace-pretrained-s42-id.json').read_text())
env=ogbench.make_env_and_datasets('cube-double-play-singletask-task1-v0',env_only=True,permute_blocks=False,visualize_info=False,width=600,height=450)
reset(env,trace['outcome']['seed'],'id')
frames=[]
try:
    for row in trace['frames']:
        env.unwrapped._data.qpos[:]=np.array(row['qpos'])
        mujoco.mj_forward(env.unwrapped._model,env.unwrapped._data)
        pixels=env.render()
        frame=Image.fromarray(pixels).convert('RGB')
        draw=ImageDraw.Draw(frame)
        draw.rectangle((0,0,frame.width,28),fill='#13242e')
        draw.text((8,8),f'Recorded Gemma policy | reset {trace["outcome"]["seed"]} | step {row["step"]}',fill='white')
        frames.append(frame)
    frames[0].save(HERE/'recorded-gemma-policy.gif',save_all=True,append_images=frames[1:],duration=150,loop=0)
    frames[-1].save(HERE/'recorded-gemma-final.png')
    (HERE/'recorded-gemma-provenance.json').write_text(json.dumps({'source_run':'ed2527c3-cc8f-4df8-a7ee-3a68cbdb6e5d','source_trace':str(SOURCE/'trace-pretrained-s42-id.json'),'outcome':trace['outcome'],'rendered_frames':len(frames),'selection':'first predeclared test reset, not selected for outcome'},indent=2))
    print(json.dumps({'outcome':trace['outcome'],'frames':len(frames),'image_size':frames[0].size}))
finally:env.close()
