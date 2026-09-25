#!/usr/bin/env python3
import json
from pathlib import Path

ROOT = Path('/home/dyf/code/distill/MAES')
RESULT = ROOT / 'results/vllm_acp_align128_completion'
REPORT = ROOT / 'docs/acp_coco_mmvet_results.md'
CONFIGS = [
    ('Kimi-VL-A3B-Instruct', 'kimi-vl-a3b', 'p30', '0.3'),
    ('Qwen3-VL-30B-A3B-Instruct', 'qwen3-vl-30b-a3b', 'p30', '0.3'),
    ('InternVL3.5-30B-A3B-HF', 'internvl3_5-30b-a3b', 'p30', '0.3'),
    ('Kimi-VL-A3B-Instruct', 'kimi-vl-a3b', 'p50', '0.5'),
    ('Qwen3-VL-30B-A3B-Instruct', 'qwen3-vl-30b-a3b', 'p50', '0.5'),
    ('InternVL3.5-30B-A3B-HF', 'internvl3_5-30b-a3b', 'p50', '0.5'),
]

def coco(run):
    files = sorted((run / 'tasks/coco2017_cap_val_local').glob('**/*_results.json'))
    metrics = None
    for path in reversed(files):
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            continue
        task_results = data.get('results')
        if isinstance(task_results, dict) and 'coco2017_cap_val_local' in task_results:
            metrics = task_results['coco2017_cap_val_local']
            break
    if metrics is None:
        return '-', '-'
    samples = sorted((run / 'tasks/coco2017_cap_val_local').glob('**/*_samples_*.jsonl'))
    count = sum(1 for _ in samples[-1].open()) if samples else '-'
    return f"{metrics['coco_CIDEr,none']:.4f}", str(count)

def mmvet(run):
    path = run / 'local_judge/mmvet_summary.json'
    if not path.is_file():
        return '-', '-'
    data = json.loads(path.read_text())
    return f"{data['score']:.4f}", str(data['num_scored'])

lines = [
    '# ACP COCO and MMVet completion results', '',
    'Protocol: ACP `first_attr_coverage`, non-modality-aware, `gateup_act`, align/min=128, '
    '`largest_channel`; padded EP4 on 4 x H20; deterministic 50% subset with seed 42.', '',
    '| Model | p | COCO CIDEr | COCO samples | MMVet judge score | MMVet samples |',
    '| --- | ---: | ---: | ---: | ---: | ---: |',
]
for model, tag, ratio_tag, ratio in CONFIGS:
    run = RESULT / tag / f'ep4-{ratio_tag}-padded/run'
    cider, coco_n = coco(run)
    vet, vet_n = mmvet(run)
    lines.append(f'| {model} | {ratio} | {cider} | {coco_n} | {vet} | {vet_n} |')
lines += ['', f'Raw results: `{RESULT}`.', '']
REPORT.write_text('\n'.join(lines))
