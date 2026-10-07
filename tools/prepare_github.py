#!/usr/bin/env python3
"""Create a public source archive without operational records or credentials."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='muscle_gpu-github')
parser.add_argument('--omit-large-example', action='store_true', help='Omit optional 27MiB RdRp structure example for web-friendly publication')
args = parser.parse_args()
if not args.name or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in args.name):
    raise SystemExit('Export name must contain only letters, digits, underscore or hyphen')
root = Path(__file__).resolve().parents[1]
dist = root / 'dist'
dist.mkdir(exist_ok=True)
stage = dist / args.name
if stage.exists():
    raise SystemExit('Output directory already exists; choose a fresh export directory before rebuilding.')
stage.mkdir()
excluded = {'.git', '.DS_Store', '__pycache__', 'build', '.tools', 'lit'}
for name in ('muscle', 'docs', '.github', 'tools'):
    for source in sorted((root / name).rglob('*')):
        relative = source.relative_to(root)
        if not source.is_file() or any(p in excluded for p in relative.parts):
            continue
        if source.suffix.lower() in ('.pdf', '.pyc'):
            continue
        if args.omit_large_example and relative.as_posix() == 'muscle/test_data/rdrp/rdrp.mega.gz':
            continue
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
for name in ('README.md', 'LICENSE', 'NOTICE.md', 'THIRD_PARTY_NOTICES.md', 'CITATION.cff', '.gitignore'):
    shutil.copy2(root / name, stage / name)
# Retain the received source provenance without publishing machine details.
provenance = stage / 'docs' / 'provenance'
provenance.mkdir(exist_ok=True)
source_manifest = root / 'planning/evidence/source_manifest.sha256'
if not source_manifest.exists():
    source_manifest = root / 'docs/provenance/source_manifest.sha256'
shutil.copy2(source_manifest, provenance)
files = {str(p.relative_to(stage)): hashlib.sha256(p.read_bytes()).hexdigest()
         for p in sorted(stage.rglob('*')) if p.is_file()}
(stage / 'docs/release_tree.json').write_text(json.dumps(
    {'schema': 1, 'files': files, 'note': 'Excludes this manifest itself.'}, indent=2) + '\n')
archive = dist / (args.name + '.tar.gz')
with tarfile.open(archive, 'w:gz') as output:
    output.add(stage, arcname=stage.name)
print(f'Prepared {len(files)} files plus manifest: {archive}')
