#!/usr/bin/env python3
"""Write the first N records of the supplied RdRp FASTA, without duplication."""
import argparse
import hashlib
from pathlib import Path
p = argparse.ArgumentParser()
p.add_argument('--source', default='muscle/test_data/rdrp/rdrp.fa')
p.add_argument('--count', type=int, required=True)
p.add_argument('--output', required=True)
a = p.parse_args()
records = []
for line in Path(a.source).read_text().splitlines():
    if line.startswith('>'):
        records.append([line, ''])
    elif line.strip():
        if not records:
            raise SystemExit('Sequence before FASTA header')
        records[-1][1] += line.strip()
if not 1 <= a.count <= len(records):
    raise SystemExit('Invalid record count')
output = Path(a.output)
output.write_text(''.join(h+'\n'+s+'\n' for h,s in records[:a.count]))
print(f'{a.count} records; sha256={hashlib.sha256(output.read_bytes()).hexdigest()}')
