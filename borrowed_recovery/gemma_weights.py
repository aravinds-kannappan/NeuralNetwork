"""Fetch only one pinned public Gemma layer, preserving its source tensor bytes."""
from __future__ import annotations
import gc
import hashlib
import json
from pathlib import Path
import struct
import time

import requests
import torch
from safetensors.torch import load_file

MODEL='google/gemma-4-31B'
REVISION='5bbc2fb1c1b2c611d06e3d9f23c170ba21659d89'
PREFIX='model.language_model.layers.26.'
SHARD='model-00001-of-00002.safetensors'
BASE=f'https://huggingface.co/{MODEL}/resolve/{REVISION}/'


def emit(kind,**data):print(json.dumps({'kind':kind,**data}),flush=True)


def file_sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def byte_range(session,start,end):
    # Unique query avoids a proxy serving one cached partial response for another range.
    url=BASE+SHARD+f'?download=true&slice={start}-{end}'
    for attempt in range(3):
        try:
            r=session.get(url,headers={'Range':f'bytes={start}-{end}'},timeout=(30,90))
            r.raise_for_status()
            assert r.status_code==206,f'range not honored: {r.status_code}'
            assert r.headers.get('Content-Range','').startswith(f'bytes {start}-{end}/')
            assert len(r.content)==end-start+1
            return r.content
        except (requests.RequestException,AssertionError):
            if attempt==2:raise
            time.sleep(2)


def load_layer(cache_root):
    root=Path(cache_root)/'models'/('gemma4-l26-'+REVISION[:12]);root.mkdir(parents=True,exist_ok=True)
    target=root/'layer.safetensors';manifest=root/'manifest.json'
    if target.exists() and manifest.exists():
        meta=json.loads(manifest.read_text())
        assert meta['revision']==REVISION and meta['sha256']==file_sha(target)
    else:
        session=requests.Session()
        header_len=struct.unpack('<Q',byte_range(session,0,7))[0]
        header=json.loads(byte_range(session,8,7+header_len))
        chosen={k[len(PREFIX):]:v for k,v in header.items() if k.startswith(PREFIX)}
        assert len(chosen)==14
        start=min(v['data_offsets'][0] for v in chosen.values())
        end=max(v['data_offsets'][1] for v in chosen.values())
        assert sum(v['data_offsets'][1]-v['data_offsets'][0] for v in chosen.values())==end-start
        local={k:{**v,'data_offsets':[v['data_offsets'][0]-start,v['data_offsets'][1]-start]} for k,v in chosen.items()}
        local['__metadata__']={'model':MODEL,'revision':REVISION,'layer':'26'}
        head=json.dumps(local,separators=(',',':')).encode();head+=b' '*((-len(head))%8)
        partial=target.with_suffix('.partial')
        with partial.open('wb') as f:
            f.write(struct.pack('<Q',len(head)));f.write(head)
            chunk=32*1024*1024
            for relative in range(0,end-start,chunk):
                count=min(chunk,end-start-relative)
                raw=byte_range(session,8+header_len+start+relative,8+header_len+start+relative+count-1)
                f.write(raw)
                emit('weight_download',bytes_complete=relative+count,total_bytes=end-start)
        partial.replace(target)
        config_response=session.get(BASE+'config.json',timeout=30);config_response.raise_for_status()
        config=config_response.json();(root/'config.json').write_text(json.dumps(config,indent=2))
        meta={'model':MODEL,'revision':REVISION,'layer':26,'sha256':file_sha(target),
              'parameters':sum(__import__('math').prod(v['shape']) for v in chosen.values()),
              'tensor_shapes':{k:v['shape'] for k,v in chosen.items()},'text_config':config['text_config'],
              'source_url':f'https://huggingface.co/{MODEL}/tree/{REVISION}',
              'scope':'Public multimodal Gemma revision; exact Abay checkpoint revision unverified.'}
        manifest.write_text(json.dumps(meta,indent=2))
    return load_file(str(target)),meta


@torch.no_grad()
def verify_native(state,meta,device):
    """Compare the extracted single-token algebra with the official implementation."""
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
    from borrowed_recovery.geometry import GemmaSingleToken
    cfg=Gemma4TextConfig(**meta['text_config']);cfg._attn_implementation='eager'
    with torch.device('meta'):
        native=Gemma4TextDecoderLayer(cfg,layer_idx=26)
    native.load_state_dict(state,strict=True,assign=True)
    native=native.to(device=device,dtype=torch.bfloat16).eval()
    custom=GemmaSingleToken(state,'pretrained',0,torch.bfloat16).to(device).eval()
    torch.manual_seed(87721)
    x=torch.randn(3,1,5376,device=device,dtype=torch.bfloat16)
    pos=(torch.ones(3,1,256,device=device,dtype=torch.bfloat16),torch.zeros(3,1,256,device=device,dtype=torch.bfloat16))
    actual=native(x.clone(),position_embeddings=pos,shared_kv_states={}).float().squeeze(1)
    expected=custom(x[:,0]).float()
    diff=actual-expected
    result={'max_abs_error':float(diff.abs().max()),
            'relative_rms_error':float(diff.square().mean().sqrt()/actual.square().mean().sqrt()),
            'dtype':'bfloat16','shape':list(actual.shape)}
    assert result['relative_rms_error']<.01,result
    emit('native_equivalence',**result)
    del native,custom,x;gc.collect()
    if device.type=='mps':torch.mps.empty_cache()
    return result
