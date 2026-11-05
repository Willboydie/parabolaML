"""Run ledger for the stage 2 experiments.

Each call writes one JSON record and appends one row to the shared CSV. The record
holds the config, the command, the seeds, the code version and the output files.
"""
import csv
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LEDGER = ROOT / 'results' / 'stage2' / 'ledger.csv'
COLUMNS = ('run_id', 'stage', 'model', 'miss_mm', 'catch_rate', 'record')


def _git():
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
        dirty = subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return dict(commit=None, dirty=None)
    return dict(commit=commit, dirty=bool(dirty), diff_files=dirty.splitlines())


def _versions():
    import numpy
    out = dict(python=platform.python_version(), numpy=numpy.__version__,
               system=platform.platform(), processor=platform.processor())
    try:
        import torch
        out['torch'] = torch.__version__
        out['cuda'] = torch.version.cuda
        out['gpu'] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except ImportError:
        pass
    try:
        import pybullet
        out['pybullet'] = pybullet.__version__ if hasattr(pybullet, '__version__') else 'unknown'
    except ImportError:
        pass
    return out


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def start_run(stage, model, config, command, seeds, note=''):
    stamp = time.strftime('%Y%m%d-%H%M%S')
    run_id = f'{stamp}-{stage}'
    record = dict(run_id=run_id, stage=stage, model=model, note=note,
                  started=time.strftime('%Y-%m-%dT%H:%M:%S'),
                  config=config, command=command, seeds=seeds, git=_git(), versions=_versions())
    path = ROOT / 'results' / 'stage2' / 'runs' / f'{run_id}.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2, default=str))
    return run_id, path, record


def finish_run(path, record, metrics, outputs):
    record['ended'] = time.strftime('%Y-%m-%dT%H:%M:%S')
    record['metrics'] = metrics
    record['outputs'] = []
    for item in outputs:
        item = Path(item)
        if item.exists():
            record['outputs'].append(dict(path=str(item), bytes=item.stat().st_size, sha256=file_hash(item)))
    Path(path).write_text(json.dumps(record, indent=2, default=str))
    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.exists()
    with LEDGER.open('a', newline='') as handle:
        writer = csv.DictWriter(handle, COLUMNS)
        if new:
            writer.writeheader()
        writer.writerow(dict(run_id=record['run_id'], stage=record['stage'], model=record['model'],
                             miss_mm=metrics.get('miss_mm', ''), catch_rate=metrics.get('catch_rate', ''),
                             record=str(path)))
    return record
