"""Download public model files via mirror GETs, avoiding broken HEAD metadata."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import os
import time
from urllib.parse import quote


def fetch(args, report, attempts=12, sleep=time.sleep):
    for attempt in range(1, attempts + 1):
        result = subprocess.run(
            ['curl', '-fL', '--connect-timeout', '30', '--write-out', '\n%{http_code}', *args],
            stdout=subprocess.PIPE, text=True)
        body, _, status = result.stdout.rpartition('\n')
        if result.returncode == 0:
            return body
        transient = result.returncode in {5, 6, 7, 18, 28, 35, 52, 55, 56, 92} or (
            result.returncode == 22 and status in {'408', '429', '500', '502', '503', '504'})
        if not transient or attempt == attempts:
            report(f'ATTENTION curl={result.returncode}, HTTP={status}, attempt={attempt}/{attempts}; partial preserved')
            raise RuntimeError('Download failed; see download_status.md')
        delay = min(30 * 2 ** (attempt - 1), 600)
        report(f'RETRY {attempt}/{attempts}: curl={result.returncode}, HTTP={status}; wait {delay}s')
        sleep(delay)


repo, directory = sys.argv[1:]
root = Path(directory)
endpoint = 'https://hf-mirror.com'
attempts = int(os.getenv('DOWNLOAD_MAX_ATTEMPTS', '12'))
if not 1 <= attempts <= 100:
    raise ValueError('DOWNLOAD_MAX_ATTEMPTS must be 1..100')


def report(message):
    line = f'{time.strftime("%F %T %Z")}: {message}'
    print(line, flush=True)
    with (root / 'download_status.md').open('a') as handle:
        handle.write(f'- {line}\n')


def failure_hook(kind, value, tb):
    report(f'ATTENTION: {value}; existing files preserved')
    sys.__excepthook__(kind, value, tb)


sys.excepthook = failure_hook
manifest = root / 'mirror_manifest.json'
if manifest.exists():
    info = json.loads(manifest.read_text())
    if info['id'].lower() != repo.lower():
        raise ValueError('Manifest repository mismatch')
else:
    info = json.loads(fetch(['--max-time', '90', f'{endpoint}/api/models/{repo}?blobs=true'], report, attempts))
    temp = root / 'mirror_manifest.json.tmp'
    temp.write_text(json.dumps(info))
    temp.replace(manifest)
revision = info['sha']
report(f'START repo={repo} revision={revision}; attempts_per_file={attempts}')
source = os.getenv('DOWNLOAD_SOURCE', 'hf-mirror')
ms_files = {}
if source == 'modelscope':
    listing = json.loads(fetch(['--max-time', '90',
        f'https://modelscope.cn/api/v1/models/{repo}/repo/files?Revision=master&Recursive=true'], report, attempts))
    ms_files = {item['Path']: item for item in listing['Data']['Files'] if item['Type'] == 'blob'}
    report('SOURCE modelscope.cn; weight hashes must match pinned HF manifest')
elif source != 'hf-mirror':
    raise ValueError(f'Unsupported source: {source}')
for entry in info['siblings']:
    name = entry['rfilename']
    if name.startswith(('consolidated', 'images/')):
        continue
    relative = Path(name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError(f'Unsafe remote path: {name}')
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    lfs = entry.get('lfs') or {}
    size = lfs.get('size', entry.get('size'))
    sha = lfs.get('sha256')
    if size is None or (name.endswith('.safetensors') and not sha):
        raise ValueError(f'Missing integrity metadata: {name}')

    def valid(path):
        if not path.is_file() or size is None or path.stat().st_size != size:
            return False
        if sha:
            digest = hashlib.sha256()
            with path.open('rb') as handle:
                for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b''):
                    digest.update(chunk)
            return digest.hexdigest() == sha
        return True

    if valid(target):
        report(f'SKIP verified {name}')
        continue
    partial = target.with_name(target.name + '.mirror-partial')
    report(f'DOWNLOAD {name} expected_bytes={size}')
    if partial.exists() and partial.stat().st_size >= size and not valid(partial):
        raise RuntimeError(f'Invalid full-sized partial: {name}; preserved for diagnosis')
    if not valid(partial):
        url = f'{endpoint}/{repo}/resolve/{revision}/{quote(name, safe="/")}'
        if source == 'modelscope' and name.endswith('.safetensors'):
            item = ms_files.get(name, {})
            if item.get('Sha256') != sha or item.get('Size') != size:
                raise ValueError(f'ModelScope weight differs from pinned HF manifest: {name}')
            url = f'https://modelscope.cn/models/{repo}/resolve/{item["Revision"]}/{quote(name, safe="/")}'
        fetch([
                    '--speed-limit', '1024', '--speed-time', '120', '-C', '-',
                    '-o', str(partial),
                    url], report, attempts)
    if not valid(partial):
        raise RuntimeError(f'Integrity check failed for {name}; partial preserved')
    partial.replace(target)
    report(f'VERIFIED {name}')
report('DOWNLOAD_COMPLETE')
