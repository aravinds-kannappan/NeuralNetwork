"""Offline arithmetic on published activation values; does not load or run a model."""
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parent

def cosine(a, b):
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(x*x for x in b))
    return sum(x*y for x,y in zip(a,b)) / (na*nb) if na and nb else None

def centered_cosine(a, b):
    ma, mb = statistics.mean(a), statistics.mean(b)
    return cosine([x-ma for x in a], [x-mb for x in b])

def main():
    data = json.loads((ROOT/'checkpoint_cosine_input.json').read_text())
    ex, steps = data['example'], data['snapshot_steps']
    vals = ex['activation_own_direction_by_snapshot']
    onset = ex['birth_snapshot']['banked']
    a, b = vals[onset-1], vals[onset]
    med0, med1 = statistics.median(a), statistics.median(b)
    q0, q1 = ex['threshold_by_snapshot'][onset-1:onset+1]
    output = {
        'kind': 'offline reanalysis of published rounded values; no model experiment',
        'source_url': data['source_url'], 'scope': data['scope'],
        'pole': ex['id'], 'n_examples': len(a),
        'steps': [steps[onset-1], steps[onset]],
        'vectors': [a,b], 'cosine': cosine(a,b),
        'centered_cosine': centered_cosine(a,b),
        'leave_one_example_out_cosines': [cosine(a[:j]+a[j+1:], b[:j]+b[j+1:]) for j in range(len(a))],
        'median_activations': [med0,med1], 'percentile_thresholds': [q0,q1],
        'median_relative_change': med1/med0-1, 'threshold_relative_change': q1/q0-1,
        'median_over_threshold': [med0/q0, med1/q1],
        'onset_by_percentile_rule': steps[onset],
        'onset_by_raw_ratio_rule': steps[ex['birth_by_raw_ratio_reading']],
        'limitations': [
            'Four examples selected from the final checkpoint; no random or held-out panel.',
            'Positive-pole magnitudes with zero clipping, not full signed neuron activations.',
            'The CRUD example dominates these vectors; a high cosine does not establish semantic stability.',
            'This is not the lithium neuron and gives no evidence about causal importance.'
        ]
    }
    (ROOT/'checkpoint_cosine_demo.json').write_text(json.dumps(output, indent=2)+'\n')
    print(json.dumps(output, indent=2))

if __name__ == '__main__':
    main()
