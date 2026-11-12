"""Copy already completed run outputs and committed code into a portable bundle.

This is evidence collection, not an experiment launcher. Run paths are recorded
explicitly so a source mismatch fails rather than silently choosing a newer run.
"""
from pathlib import Path
import csv
import hashlib
import json
import shutil
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = Path('/Users/aravindkannappan/.local/share/openresearch')
RUNS = {
    'baseline':'dcdf3af0-76f9-45f2-bfc5-09ccf2ca1820',
    'paired':'39f2f1ea-51df-4e9a-8db7-dd150a708280',
    'false-freshness':'db0ed781-2a03-44a5-95c2-cb5098a17073',
}
CODE = ['README.md','PROTOCOL.md','pyproject.toml','uv.lock','.python-version',
        'config.json','borrowed_recovery/__init__.py','borrowed_recovery/experiment.py']

def main():
    data = HERE/'data'
    data.mkdir(exist_ok=True)
    all_rows, all_traces, manifest = [], [], []
    with zipfile.ZipFile(HERE/'reproduction.zip','w',zipfile.ZIP_DEFLATED) as bundle:
        for name, run_id in RUNS.items():
            repo = ROOT/'local-runs'/run_id/'repo'
            log = ROOT/'run-logs'/f'{run_id}.log'
            records = [json.loads(line) for line in log.read_text().splitlines() if line.startswith('{')]
            summary = next(x for x in records if x.get('kind')=='summary')
            saved = json.loads((repo/'outputs/summary.json').read_text())
            assert {k:v for k,v in summary.items() if k!='kind'} == saved
            assert summary['run_id'] == run_id
            rows = json.loads((repo/'outputs/episodes.json').read_text())
            assert len(rows) == saved['episodes']
            assert [x for x in records if x.get('kind')=='episode'] == [{'kind':'episode',**r} for r in rows]
            traces = json.loads((repo/'outputs/traces.json').read_text())
            shutil.copyfile(repo/'outputs/summary.json',data/f'{name}-summary.json')
            for row in rows:
                all_rows.append({'study':name,'run_id':run_id,**row})
            for trace in traces:
                if name != 'baseline':
                    all_traces.append({'study':name,'run_id':run_id,**trace})
            source_hashes = {}
            for filename in CODE:
                source = repo/filename
                source_hashes[filename] = hashlib.sha256(source.read_bytes()).hexdigest()
                bundle.write(source,f'{name}/{filename}')
            bundle.write(log,f'{name}/run.log')
            for filename in ['episodes.json','summary.json','traces.json']:
                bundle.write(repo/'outputs'/filename,f'{name}/recorded/{filename}')
            manifest.append({'study':name,'run_id':run_id,'commit':saved['commit'],
                             'episodes':len(rows),'config':saved['config'],
                             'versions':saved['versions'],'source_sha256':source_hashes})
        bundle.writestr('REPRODUCE.txt',
            'Each study folder is an independently reproducible code snapshot.\n'
            'Install uv if it is not available; then, inside the desired study folder:\n'
            '  uv run --locked python -m borrowed_recovery.experiment\n'
            'This creates outputs/. Recorded original results are in recorded/.\n'
            'Python 3.12 and the locked dependencies are required. Network is needed\n'
            'only to obtain dependencies. The simulation itself runs locally, headless.\n'
            'No learned weights or offline OGBench datasets are downloaded.\n'
            'Floating-point and dependency-platform differences may affect exact replay\n'
            'on another machine. The recorded checks apply to the original local host.\n')
    (data/'episodes.json').write_text(json.dumps(all_rows,indent=2))
    (data/'traces.json').write_text(json.dumps(all_traces,separators=(',',':')))
    (data/'provenance.json').write_text(json.dumps(manifest,indent=2))
    with (data/'episodes.csv').open('w',newline='') as f:
        writer = csv.DictWriter(f,fieldnames=list(all_rows[0]))
        writer.writeheader(); writer.writerows(all_rows)
    print(json.dumps({'scored_episodes':len(all_rows),'comparative_traces':len(all_traces),'runs':RUNS}))

if __name__ == '__main__':
    main()
