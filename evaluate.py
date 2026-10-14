"""
Evaluate results.json files produced by demo.py.

Usage:
    # Single file
    python evaluate.py results/vqa-rad_results.json

    # Compare multiple runs (e.g. ablation across CFG settings)
    python evaluate.py results/vqa-rad_base.json results/vqa-rad_cfg.json results/vqa-rad_full.json
"""

import argparse
import json
import sys
from pathlib import Path


def load_results(path: str) -> dict:
    with open(path) as f:
        data = json.load(f)

    # Support both old flat-list format and new {meta, summary, results} format
    if isinstance(data, list):
        results = data
        meta = {}
    else:
        results = data.get('results', [])
        meta = data.get('meta', {})

    return {'meta': meta, 'results': results, 'path': path}


def compute_metrics(results: list[dict]) -> dict:
    total = len(results)
    if total == 0:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}

    correct = sum(1 for r in results if r.get('correct', False))

    # Closed (yes/no) vs open breakdown — if 'question_type' field exists
    closed = [r for r in results if r.get('question_type', '').lower() == 'closed']
    open_q = [r for r in results if r.get('question_type', '').lower() == 'open']

    metrics = {
        'total': total,
        'correct': correct,
        'accuracy': round(correct / total, 4),
    }

    if closed:
        c_correct = sum(1 for r in closed if r.get('correct', False))
        metrics['closed_total'] = len(closed)
        metrics['closed_accuracy'] = round(c_correct / len(closed), 4)

    if open_q:
        o_correct = sum(1 for r in open_q if r.get('correct', False))
        metrics['open_total'] = len(open_q)
        metrics['open_accuracy'] = round(o_correct / len(open_q), 4)

    # Average predictive entropy
    pes = [r['predictive_entropy'] for r in results if 'predictive_entropy' in r]
    if pes:
        metrics['mean_predictive_entropy'] = round(sum(pes) / len(pes), 4)

    return metrics


def format_run(path: str, meta: dict, metrics: dict) -> str:
    name = Path(path).stem
    dataset = meta.get('dataset', '?')
    cfg = meta.get('cfg', '?')
    lora = 'LoRA' if meta.get('use_lora') else 'base'
    patch = '+attn' if meta.get('use_attn_patch') else ''
    tag = f"{dataset} | γ={cfg} | {lora}{patch}"

    lines = [
        f"  File   : {name}",
        f"  Config : {tag}",
        f"  Total  : {metrics['total']}",
        f"  Correct: {metrics['correct']}",
        f"  Acc    : {metrics['accuracy']:.4f}  ({metrics['accuracy']*100:.2f}%)",
    ]
    if 'closed_accuracy' in metrics:
        lines.append(f"  Closed : {metrics['closed_accuracy']:.4f}  (n={metrics['closed_total']})")
    if 'open_accuracy' in metrics:
        lines.append(f"  Open   : {metrics['open_accuracy']:.4f}  (n={metrics['open_total']})")
    if 'mean_predictive_entropy' in metrics:
        lines.append(f"  Mean PE: {metrics['mean_predictive_entropy']:.4f}")

    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description='Evaluate Expert-CFG results JSON files')
    parser.add_argument('files', nargs='+', help='Path(s) to results.json')
    args = parser.parse_args()

    all_metrics = []
    for path in args.files:
        try:
            run = load_results(path)
        except FileNotFoundError:
            print(f'[ERROR] File not found: {path}', file=sys.stderr)
            continue
        except json.JSONDecodeError as e:
            print(f'[ERROR] JSON parse error in {path}: {e}', file=sys.stderr)
            continue

        metrics = compute_metrics(run['results'])
        all_metrics.append((path, run['meta'], metrics))
        print('─' * 50)
        print(format_run(path, run['meta'], metrics))

    if len(all_metrics) > 1:
        print('\n' + '═' * 50)
        print('Comparison (accuracy)')
        print('─' * 50)
        header = f"{'Run':<30}  {'Dataset':<10}  {'Acc':>6}  {'CFG':>5}  {'LoRA':>4}  {'Patch':>5}"
        print(header)
        for path, meta, m in all_metrics:
            name = Path(path).stem[:28]
            dataset = meta.get('dataset', '?')[:10]
            cfg = str(meta.get('cfg', '?'))
            lora = 'Y' if meta.get('use_lora') else 'N'
            patch = 'Y' if meta.get('use_attn_patch') else 'N'
            print(f"{name:<30}  {dataset:<10}  {m['accuracy']:>6.4f}  {cfg:>5}  {lora:>4}  {patch:>5}")

    print('─' * 50)


if __name__ == '__main__':
    main()
