"""
Expert-CFG selective 5% pipeline — Phase 2 rerun.

Phase 1: Run demo.py baseline (--no_rag or plain RAG) on ALL samples → gets PE per sample.
Phase 2: THIS SCRIPT — select top k% by PE, rerun with Expert-CFG settings, merge.

Usage:
    python selective_rerun.py \
        --baseline results/vqa-rad_1_base.json \
        --top_k_pct 5.0 \
        --model_name_or_path ./Phi-3.5-vision-instruct \
        --input_json data/RadVQA/test.json \
        --img_root data/RadVQA/images \
        --output_dir lora_weights/logs_phi35_pubmed_instruct \
        --use_qlora --bf16 \
        --cfg 1.5 --attn 3.0 \
        --use_attn_patch --use_embed_perturb \
        --save_path results/vqa-rad_5_5pct_expert_cfg.json
"""

import argparse
import datetime
import json
import os

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor, BitsAndBytesConfig

from src.datasets.radvqa import RADVQADataset
from src.datasets.slakevqa import SLAKEVQADataset
from src.datasets.pathvqa import PathVQADataset
from src.highlighter_modules.guidance import ProbCFGLogitsProcessor
from src.highlighter_modules.attention_phi3 import (
    patch_phi3_attention,
    patch_phi3_embedding,
    set_highlight_mask,
    clear_highlight_mask,
)
from src.highlighter_modules.uncertainty import compute_predictive_entropy


DATASET_MAP = {
    'vqa-rad': RADVQADataset,
    'slake': SLAKEVQADataset,
    'pathvqa': PathVQADataset,
}


import re as _re


def _tokenize(text: str) -> list:
    return _re.findall(r'[a-z0-9]+', str(text).lower())


def _score_answer(prediction: str, gt_answer: str, answer_type: str = 'OPEN') -> float:
    gt = str(gt_answer).strip().lower()
    pred_tokens = _tokenize(prediction)
    if gt in ('yes', 'no') or (answer_type or '').upper() == 'CLOSED':
        pred_label = next((t for t in pred_tokens if t in ('yes', 'no')), None)
        return float(pred_label == gt)
    gt_tokens = _tokenize(gt)
    if not gt_tokens:
        return 0.0
    return sum(t in pred_tokens for t in gt_tokens) / len(gt_tokens)


def _is_correct(prediction: str, gt_answer: str, answer_type: str = 'OPEN') -> bool:
    return _score_answer(prediction, gt_answer, answer_type) == 1.0


def select_top_k(baseline_results: list, top_k_pct: float) -> set:
    """Return dataset indices of the top k% samples by predictive entropy (most uncertain)."""
    ranked = sorted(
        range(len(baseline_results)),
        key=lambda i: baseline_results[i].get('predictive_entropy', 0.0),
        reverse=True,
    )
    k = max(1, round(len(baseline_results) * top_k_pct / 100.0))
    return set(ranked[:k])


@torch.no_grad()
def rerun_selected(model, processor, eval_dataset, selected_indices, args):
    device = f'cuda:{int(os.environ.get("LOCAL_RANK", 0))}' if torch.cuda.is_available() else 'cpu'
    model.eval()
    results = []

    for i in tqdm(sorted(selected_indices), desc=f'Selective rerun (top {args.top_k_pct}%)'):
        example = eval_dataset[i]
        image = example['image']
        question = example['question']
        gt_answer = example['answer']
        caption = example['top_k_captions'][0]
        highlights = example['highlights']

        prompt_message = {
            'role': 'user',
            'content': f'{caption} <|image_1|>\n{question}',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            prompt, [image], return_tensors='pt', qs_highlighted_parts=highlights
        ).to(device)

        binary_hl_mask = inputs['highlight_attention_mask'].float()
        del inputs['highlight_attention_mask']

        cfg_batched_input = inputs['input_ids'].expand(2, -1)
        pixel_values = inputs['pixel_values'].expand(2, *inputs['pixel_values'].shape[1:])
        image_sizes = inputs['image_sizes'].expand(2, *inputs['image_sizes'].shape[1:])
        attention_mask = inputs['attention_mask'].expand(2, -1)

        if args.use_attn_patch:
            set_highlight_mask(binary_hl_mask)

        try:
            generated_outputs = model.generate(
                input_ids=cfg_batched_input,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=64,
                num_beams=1,
                logits_processor=[ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)],
                output_scores=True,
                return_dict_in_generate=True,
            )
        finally:
            if args.use_attn_patch:
                clear_highlight_mask()

        prompt_len = inputs['input_ids'].size(1)
        prediction = processor.batch_decode(
            generated_outputs.sequences[:1, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip().strip('.')

        token_probs, token_preds = [], []
        for step_idx, step_scores in enumerate(generated_outputs.scores):
            probs = torch.softmax(step_scores[0].float(), dim=-1)
            token_id = generated_outputs.sequences[0, prompt_len + step_idx]
            token_probs.append(probs[token_id].item())
            token_preds.append(processor.decode(token_id))

        pe = compute_predictive_entropy(generated_outputs.scores)
        answer_type = example.get('answer_type', 'OPEN')
        score = _score_answer(prediction, gt_answer, answer_type)
        correct = (score == 1.0)
        print(f'[{i}] Q: {question}  GT: {gt_answer}  Pred: {prediction}  {"✓" if correct else "✗"}')

        results.append({
            'dataset_index': i,
            'img_id': example.get('image_id', i),
            'question': question,
            'answer': gt_answer,
            'answer_type': answer_type,
            'prediction': prediction,
            'score': round(score, 4),
            'correct': correct,
            'predictive_entropy': round(pe, 6),
            'token_probs': token_probs,
            'token_preds': token_preds,
        })

    return results


def main():
    parser = argparse.ArgumentParser(description='Expert-CFG selective 5% rerun (Phase 2)')
    parser.add_argument('--baseline', type=str, required=True,
                        help='Phase 1 baseline results JSON (must have predictive_entropy)')
    parser.add_argument('--top_k_pct', type=float, default=5.0,
                        help='Top k%% of samples by PE to rerun (default 5.0)')
    parser.add_argument('--model_name_or_path', type=str, default='./Phi-3.5-vision-instruct')
    parser.add_argument('--input_json', type=str, required=True)
    parser.add_argument('--img_root', type=str, required=True)
    parser.add_argument('--dataset', type=str, choices=['vqa-rad', 'slake', 'pathvqa'], default='vqa-rad')
    parser.add_argument('--output_dir', type=str, help='LoRA weights directory')
    parser.add_argument('--use_qlora', action='store_true')
    parser.add_argument('--bf16', action='store_true')
    parser.add_argument('--num_crops', type=int, default=4)
    parser.add_argument('--cfg', type=float, default=1.5)
    parser.add_argument('--attn', type=float, default=3.0)
    parser.add_argument('--perturb_weight', type=float, default=0.01)
    parser.add_argument('--use_attn_patch', action='store_true')
    parser.add_argument('--use_embed_perturb', action='store_true')
    parser.add_argument('--save_path', type=str, required=True)
    args = parser.parse_args()

    # Load baseline
    with open(args.baseline) as f:
        baseline_data = json.load(f)
    baseline_results = baseline_data['results'] if 'results' in baseline_data else baseline_data

    selected = select_top_k(baseline_results, args.top_k_pct)
    n_total = len(baseline_results)
    n_selected = len(selected)
    print(f'[selective] top {args.top_k_pct}% = {n_selected}/{n_total} samples will be rerun')

    # Load model
    use_bf16 = args.bf16
    bnb_cfg = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        )
        if args.use_qlora
        else None
    )

    print(f'Loading model: {args.model_name_or_path}')
    processor = AutoProcessor.from_pretrained(
        args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        trust_remote_code=True,
        _attn_implementation='eager',
        quantization_config=bnb_cfg,
    )
    if args.use_qlora and args.output_dir:
        model.load_adapter(args.output_dir)
        print(f'Loaded LoRA: {args.output_dir}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    if args.use_attn_patch:
        patch_phi3_attention(model, beta=args.attn)
    if args.use_embed_perturb:
        patch_phi3_embedding(model, alpha=args.perturb_weight)

    # Load dataset
    eval_dataset = DATASET_MAP[args.dataset](annotation_file=args.input_json, vis_root=args.img_root)

    # Run Phase 2 on selected samples
    new_results = rerun_selected(model, processor, eval_dataset, selected, args)
    new_by_idx = {r['dataset_index']: r for r in new_results}

    # Merge: baseline for non-selected, new results for selected
    merged = []
    for i, base_r in enumerate(baseline_results):
        merged.append(new_by_idx[i] if i in new_by_idx else base_r)

    # Summary — same metric as teammate: mean score per split
    closed = [r for r in merged if r.get('answer_type', '').upper() == 'CLOSED']
    open_q = [r for r in merged if r.get('answer_type', '').upper() == 'OPEN']
    c_acc = sum(r.get('score', float(r['correct'])) for r in closed) / len(closed) if closed else 0.0
    o_acc = sum(r.get('score', float(r['correct'])) for r in open_q) / len(open_q) if open_q else 0.0
    ov_acc = sum(r.get('score', float(r['correct'])) for r in merged) / len(merged) if merged else 0.0

    output = {
        'meta': {
            'dataset': args.dataset,
            'model': args.model_name_or_path,
            'use_lora': bool(args.output_dir),
            'cfg': args.cfg,
            'attn_beta': args.attn,
            'perturb_weight': args.perturb_weight,
            'use_attn_patch': args.use_attn_patch,
            'use_embed_perturb': args.use_embed_perturb,
            'selective_top_k_pct': args.top_k_pct,
            'selective_n': n_selected,
            'baseline_path': args.baseline,
            'timestamp': datetime.datetime.now().isoformat(),
        },
        'summary': {
            'total': len(merged),
            'overall_score': round(ov_acc, 4),
            'closed_total': len(closed),
            'closed_score': round(c_acc, 4),
            'open_total': len(open_q),
            'open_score': round(o_acc, 4),
        },
        'results': merged,
    }

    with open(args.save_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f'\nClosed: {c_acc*100:.2f}%  Open: {o_acc*100:.2f}%  Overall: {ov_acc*100:.2f}%')
    print(f'Saved → {args.save_path}')
    print(f'Saved → {args.save_path}')


if __name__ == '__main__':
    main()
