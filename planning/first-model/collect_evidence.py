# SPDX-License-Identifier: Apache-2.0
"""Record the isolated CPU dependency spike; does not implement project features.
Runs from repo root; OSV query sends only public package names and versions.
"""
import hashlib
import importlib.metadata as metadata
import json
from pathlib import Path
import platform
import urllib.request
from datetime import datetime, timezone

import torch

OUT = Path(__file__).resolve().parent
report = dict(collected_at=datetime.now(timezone.utc).isoformat(),
              python=platform.python_version(), platform=platform.platform(),
              torch=torch.__version__, cuda_available=torch.cuda.is_available(),
              tokenizer_files={}, packages=[])
for p in Path('.venv/compat-tokenizer').iterdir():
    report['tokenizer_files'][p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
for d in sorted(metadata.distributions(), key=lambda d: d.metadata['Name'].lower()):
    md = d.metadata
    item = dict(name=md['Name'], version=d.version,
                license_expression=md.get('License-Expression'),
                license_metadata_first_line=(md.get('License') or '').split('\n')[0][:250],
                license_classifiers=[x for x in md.get_all('Classifier', []) if 'License' in x],
                installed_license_files=[])
    for f in d.files or []:
        s = str(f)
        if '.dist-info/' in s and any(v in Path(s).name.lower() for v in ['license', 'licence', 'notice']):
            actual = d.locate_file(f)
            if actual.is_file():
                item['installed_license_files'].append(dict(path=s,
                    sha256=hashlib.sha256(actual.read_bytes()).hexdigest(), bytes=actual.stat().st_size))
    report['packages'].append(item)
report['numpy_bundled_native_files'] = [str(f) for f in metadata.distribution('numpy').files if 'numpy.libs/' in str(f)]
(OUT / 'environment-evidence.json').write_text(json.dumps(report, indent=2) + '\n')
packages = [p for p in report['packages'] if p['name'].lower() not in ['pip', 'setuptools', 'wheel']]
query = {'queries': [{'package': {'name': p['name'], 'ecosystem': 'PyPI'},
                      'version': p['version'].split('+')[0]} for p in packages]}
req = urllib.request.Request('https://api.osv.dev/v1/querybatch', data=json.dumps(query).encode(),
                             headers={'Content-Type': 'application/json'})
with urllib.request.urlopen(req, timeout=30) as response:
    result = json.load(response)
assert len(result['results']) == len(packages)
security = dict(queried_at=datetime.now(timezone.utc).isoformat(),
    source='https://api.osv.dev/v1/querybatch',
    scope='CPU Python versions only; matches need triage, absence is not a security guarantee. Excludes installer tools, bundled native libraries and GPU runtime.',
    packages=[dict(name=p['name'], version=p['version'], matches=r.get('vulns', []))
              for p, r in zip(packages, result['results'])])
(OUT / 'security-advisory-matches.json').write_text(json.dumps(security, indent=2) + '\n')
for p in security['packages']:
    if p['matches']:
        print(p['name'], p['version'], [v['id'] for v in p['matches']])
print('Queried', len(packages), 'packages;', sum(bool(p['matches']) for p in security['packages']), 'with advisory matches')
