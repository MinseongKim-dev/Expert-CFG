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


def _is_correct(result: dict) -> bool:
    """Smart match matching demo.py evaluation: first-word for CLOSED, substring for OPEN."""
    pred = result.get('prediction', '').lower().strip().rstrip('.')
    gt = result.get('answer', '').lower().strip().rstrip('.')
    answer_type = result.get('answer_type', 'OPEN')
    if pred == gt:
        return True
    if answer_type == 'CLOSED' or gt in ('yes', 'no'):
        return (pred.split()[0] == gt) if pred else False
    return gt in pred or pred in gt


def compute_metrics(results: list[dict]) -> dict:
    """Compute accuracy metrics matching the paper's evaluation format.

    Splits by answer_type field (CLOSED / OPEN) which maps to:
      CLOSED — yes/no binary questions
      OPEN   — free-form / entity questions
    """
    total = len(results)
    if total == 0:
        return {'total': 0, 'correct': 0, 'accuracy': 0.0}

    # Re-compute correct using consistent normalization
    for r in results:
        r['correct'] = _is_correct(r)

    correct = sum(1 for r in results if r['correct'])

    # Closed vs open split — answer_type field from dataset JSON
    closed = [r for r in results if r.get('answer_type', '').upper() == 'CLOSED']
    open_q = [r for r in results if r.get('answer_type', '').upper() == 'OPEN']

    metrics: dict = {
        'total': total,
        'correct': correct,
        'accuracy': round(correct / total, 4),
    }

    if closed:
        c_correct = sum(1 for r in closed if r['correct'])
        metrics['closed_total'] = len(closed)
        metrics['closed_correct'] = c_correct
        metrics['closed_accuracy'] = round(c_correct / len(closed), 4)

    if open_q:
        o_correct = sum(1 for r in open_q if r['correct'])
        metrics['open_total'] = len(open_q)
        metrics['open_correct'] = o_correct
        metrics['open_accuracy'] = round(o_correct / len(open_q), 4)

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

    def acc_str(acc, n):
        return f"{acc*100:.2f}%  (n={n})"

    lines = [
        f"  File   : {name}",
        f"  Config : {tag}",
        f"  ── Overall ─────────────────────────────",
        f"  Total  : {metrics['total']}  Correct: {metrics['correct']}",
        f"  Acc    : {metrics['accuracy']*100:.2f}%",
    ]

    if 'closed_accuracy' in metrics:
        lines.append(
            f"  Closed : {acc_str(metrics['closed_accuracy'], metrics['closed_total'])}"
            f"  correct={metrics['closed_correct']}"
        )
    if 'open_accuracy' in metrics:
        lines.append(
            f"  Open   : {acc_str(metrics['open_accuracy'], metrics['open_total'])}"
            f"  correct={metrics['open_correct']}"
        )
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
        print('\n' + '═' * 70)
        print('Comparison  (논문 Table 형식 — Closed / Open / Overall)')
        print('─' * 70)
        header = (
            f"{'Run':<28}  {'Dataset':<9}  "
            f"{'Closed':>7}  {'Open':>7}  {'Overall':>8}  "
            f"{'γ':>4}  {'LoRA':>4}  {'Attn':>4}"
        )
        print(header)
        print('─' * 70)
        for path, meta, m in all_metrics:
            name = Path(path).stem[:26]
            dataset = meta.get('dataset', '?')[:9]
            cfg = str(meta.get('cfg', '?'))
            lora = 'Y' if meta.get('use_lora') else 'N'
            patch = 'Y' if meta.get('use_attn_patch') else 'N'
            closed_s = f"{m['closed_accuracy']*100:.1f}%" if 'closed_accuracy' in m else '  —  '
            open_s   = f"{m['open_accuracy']*100:.1f}%"   if 'open_accuracy'   in m else '  —  '
            overall  = f"{m['accuracy']*100:.1f}%"
            print(f"{name:<28}  {dataset:<9}  {closed_s:>7}  {open_s:>7}  {overall:>8}  {cfg:>4}  {lora:>4}  {patch:>4}")

    print('─' * 50)


if __name__ == '__main__':
    main()
