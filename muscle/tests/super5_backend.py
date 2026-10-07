#!/usr/bin/env python3
"""Bounded Super5 CUDA integration: outputs, partition/path hashes, coverage."""
import argparse
import json
import pathlib
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--binary', required=True)
p.add_argument('--input', required=True)
p.add_argument('--out', required=True)
p.add_argument('--threads', type=int, default=1)
p.add_argument('--cluster-min-ea', type=float)
p.add_argument('--require-stage', action='append', default=[])
p.add_argument('--require-ppp', action='store_true')
p.add_argument('--perm', default='none', choices=['none', 'all'])
a = p.parse_args()
out = pathlib.Path(a.out)
out.mkdir(parents=True, exist_ok=True)
base = [a.binary, '-super5', a.input, '-threads', str(a.threads),
        '-perturb', '1', '-randseed', '1', '-perm', a.perm]
if a.cluster_min_ea is not None:
    base += ['-super4_minea1', str(a.cluster_min_ea)]
traces = {}
for backend in ['cpu', 'gpu']:
    trace = out / (backend + '.jsonl')
    target = out / (backend + '.@.afa' if a.perm == 'all' else backend + '.afa')
    cmd = base + ['-backend', backend, '-output', str(target), '-stage_trace', str(trace)]
    if backend == 'gpu':
        cmd += ['-gpu_verify']
    with (out / (backend + '.log')).open('w') as log:
        subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=90)
    traces[backend] = [json.loads(x) for x in trace.read_text().splitlines() if x.strip()]
cpu_files = sorted(out.glob('cpu.*.afa')) if a.perm == 'all' else [out / 'cpu.afa']
assert cpu_files, 'no CPU output'
assert len(cpu_files) == (4 if a.perm == 'all' else 1), 'unexpected output count'
for cpu in cpu_files:
    gpu = out / cpu.name.replace('cpu.', 'gpu.', 1)
    assert cpu.read_bytes() == gpu.read_bytes(), f'byte mismatch: {cpu.name}'
def partitions(records):
    return [r['stage'] for r in records if '.partition:' in r.get('stage', '')]
assert any(r['stage'].startswith('super5.uclust.partition:') for r in traces['cpu']), 'missing UClust partition evidence'
assert any(r['stage'].startswith('super5.eacluster.partition:') for r in traces['cpu']), 'missing EACluster partition evidence'
assert partitions(traces['cpu']) == partitions(traces['gpu']), 'cluster membership/path hashes differ'
commits = [r for r in traces['gpu'] if r.get('stage') in ['super5.pair.commit','pair_backend.commit']]
gpu = [r for r in commits if r.get('backend') == 'cuda-pair-v1']
assert gpu, 'no actual CUDA commit'
for stage in a.require_stage:
    assert any(r.get('stage') == stage and r.get('backend') == 'cuda-pair-v1' for r in traces['gpu']), 'missing GPU stage: '+stage
if a.require_ppp:
    assert any(r.get('stage') == 'pair_backend.commit' and r.get('backend') == 'cuda-pair-v1' for r in traces['gpu']), 'missing cluster PPP CUDA commit'
assert all(not r.get('fallback_reason') for r in commits), 'unexpected fallback'
report = {'byte_equal': True, 'partition_paths_equal': True,
          'cuda_commits': len(gpu), 'outputs': len(cpu_files),
          'stages': sorted(set(r['stage'] for r in traces['gpu'] if r['stage'].startswith('super5.')))}
(out / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report), flush=True)
