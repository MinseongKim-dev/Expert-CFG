import argparse
import json
import os
import torch
from accelerate import Accelerator
from accelerate import DataLoaderConfiguration
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

import datetime

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


# suggested deepspeed config
DS_CONFIG_DICT = {
    'zero_optimization': {
        'stage': 2,
        'allgather_partitions': True,
        'allgather_bucket_size': 5e8,
        'overlap_comm': True,
        'reduce_scatter': True,
        'reduce_bucket_size': 5e8,
        'contiguous_gradients': True,
        'round_robin_gradients': True,
    },
    'fp16': {
        'enabled': 'auto',
        'loss_scale': 0,
        'loss_scale_window': 1000,
        'initial_scale_power': 16,
        'hysteresis': 2,
        'min_loss_scale': 1,
    },
    'bf16': {'enabled': 'auto'},
    'train_micro_batch_size_per_gpu': 'auto',
    'train_batch_size': 'auto',
    'gradient_accumulation_steps': 'auto',
    'gradient_clipping': 'auto',
}


DATASET_MAP = {
    'vqa-rad': RADVQADataset,
    'slake': SLAKEVQADataset,
    'pathvqa': PathVQADataset,
}

def create_dataset(args):
    dataset_cls = DATASET_MAP.get(args.dataset)
    if dataset_cls is None:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Choose from: {list(DATASET_MAP)}")
    return dataset_cls(annotation_file=args.input_json, vis_root=args.img_root)



class NoGradHook:
    def __init__(self):
        self.prev_enabled = True

    def maybe_enable_grad_hook(self, *_):
        torch.set_grad_enabled(self.prev_enabled)

    def disable_grad_hook(self, *_):
        self.prev_enabled = torch.is_grad_enabled()
        torch.set_grad_enabled(False)


def freeze_vision_model(model):
    vision_no_grad_hook = NoGradHook()
    vision_module = model.model.vision_embed_tokens
    vision_module.register_forward_pre_hook(vision_no_grad_hook.disable_grad_hook)
    vision_module.register_forward_hook(vision_no_grad_hook.maybe_enable_grad_hook)
    for p in vision_module.parameters():
        p.requires_grad_(False)


def create_model(model_name_or_path, use_flash_attention=False, use_qlora=False, use_bf16=False):
    use_bfloat16 = use_flash_attention or use_bf16
    bnb_config = (
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
        )
        if use_qlora
        else None
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        # Phi-3-V is originally trained in bf16 + flash attn
        # For fp16 mixed precision training, load in f32 to avoid hf accelerate error
        torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
        trust_remote_code=True,
        _attn_implementation='flash_attention_2' if use_flash_attention else 'eager',
        quantization_config=bnb_config,
    )

    return model


def _normalize_answer(ans: str) -> str:
    return ans.lower().strip().rstrip('.')


def _is_correct(prediction: str, gt_answer: str, answer_type: str = 'OPEN') -> bool:
    pred = _normalize_answer(prediction)
    gt = _normalize_answer(gt_answer)
    if pred == gt:
        return True
    # Closed (yes/no): match on first word of prediction
    if answer_type == 'CLOSED' or gt in ('yes', 'no'):
        return pred.split()[0].rstrip(',.!?;:') == gt if pred else False
    # Open: exact match only (논문 표준 VQA-RAD 평가)
    return False


@torch.no_grad()
def evaluate(model, processor, eval_dataset, args, save_path=None, disable_tqdm=False):
    rank = int(os.environ.get('RANK', 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))

    model.eval()
    results = []

    for i in tqdm(range(len(eval_dataset)), disable=(rank != 0) or disable_tqdm):
        # Phi-3-V currently only supports batch_size == 1
        example = eval_dataset[i]
        image = example['image']
        question = example['question']
        gt_answer = example['answer']
        caption = example["top_k_captions"][0]
        highlights = example['highlights']

        prompt_message = {
            'role': 'user',
            'content': f'{caption} <|image_1|>\n{question}' if not args.no_rag else f'<|image_1|>\n{question}',
        }
        prompt = processor.tokenizer.apply_chat_template(
            [prompt_message], tokenize=False, add_generation_prompt=True
        )

        inputs = processor(
            prompt, [image], return_tensors='pt', qs_highlighted_parts=highlights
        ).to(f'cuda:{local_rank}')

        # Binary highlight mask for attention patch (Eq. 9-10)
        binary_hl_mask = inputs['highlight_attention_mask'].float()  # [1, seq_len], 0 or 1
        del inputs['highlight_attention_mask']

        # CFG batch: [conditional, unconditional] (Eq. 8)
        # Use expand (no copy) since both branches share identical inputs
        cfg_batched_input = inputs['input_ids'].expand(2, -1)
        pixel_values = inputs['pixel_values'].expand(2, *inputs['pixel_values'].shape[1:])
        image_sizes = inputs['image_sizes'].expand(2, *inputs['image_sizes'].shape[1:])
        attention_mask = inputs['attention_mask'].expand(2, -1)

        # Register highlight mask for the attention patch (cleared after generate)
        if args.use_attn_patch:
            set_highlight_mask(binary_hl_mask)

        try:
            generated_outputs = model.generate(
                input_ids=cfg_batched_input,
                pixel_values=pixel_values,
                attention_mask=attention_mask,
                image_sizes=image_sizes,
                eos_token_id=processor.tokenizer.eos_token_id,
                max_new_tokens=32,
                num_beams=args.num_beams,
                logits_processor=[ProbCFGLogitsProcessor(guidance_scale=args.cfg, use_log=True)],
                output_scores=True,
                return_dict_in_generate=True,
            )
        finally:
            if args.use_attn_patch:
                clear_highlight_mask()

        prompt_len = inputs['input_ids'].size(1)

        # Decode conditional branch output (index 0)
        prediction = processor.batch_decode(
            generated_outputs.sequences[:1, prompt_len:],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0].strip().strip('.')

        # Per-token probabilities from conditional branch
        token_probs = []
        token_preds = []
        for step_idx, step_scores in enumerate(generated_outputs.scores):
            probs = torch.softmax(step_scores[0].float(), dim=-1)
            token_id = generated_outputs.sequences[0, prompt_len + step_idx]
            token_probs.append(probs[token_id].item())
            token_preds.append(processor.decode(token_id))

        # Predictive Entropy (Eq. 4) — length-normalised uncertainty measure
        pe = compute_predictive_entropy(generated_outputs.scores)

        correct = _is_correct(prediction, gt_answer, example.get('answer_type', 'OPEN'))

        print(f'Q: {question}  GT: {gt_answer}  Pred: {prediction}  PE: {pe:.3f}  {"✓" if correct else "✗"}')

        results.append({
            'img_id': example.get('image_id', i),
            'question': question,
            'answer': gt_answer,
            'answer_type': example.get('answer_type', 'OPEN'),  # CLOSED | OPEN
            'prediction': prediction,
            'correct': correct,
            'predictive_entropy': round(pe, 6),
            'token_probs': token_probs,
            'token_preds': token_preds,
        })

    if rank == 0 and save_path:
        # Wrap results with run metadata for cross-team comparison
        correct_count = sum(r['correct'] for r in results)
        output = {
            'meta': {
                'dataset': args.dataset,
                'model': args.model_name_or_path,
                'use_lora': args.use_lora,
                'cfg': args.cfg,
                'attn_beta': args.attn,
                'perturb_weight': args.perturb_weight,
                'use_attn_patch': args.use_attn_patch,
                'use_embed_perturb': args.use_embed_perturb,
                'no_rag': args.no_rag,
                'num_beams': args.num_beams,
                'timestamp': datetime.datetime.now().isoformat(),
            },
            'summary': {
                'total': len(results),
                'correct': correct_count,
                'accuracy': round(correct_count / len(results), 4) if results else 0.0,
            },
            'results': results,
        }
        with open(save_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f'\nAccuracy: {correct_count}/{len(results)} = {output["summary"]["accuracy"]:.4f}')
        print(f'Saved → {save_path}')



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_name_or_path',
        type=str,
        # default='./Phi-3-vision-128k-instruct',
        default='./Phi-3.5-vision-instruct',
        help='Model name or path to load from',
    )
    parser.add_argument('--use_flash_attention', action='store_true', help='Use Flash Attention')
    parser.add_argument('--bf16', action='store_true', help='Use BF16')
    parser.add_argument('--use_lora', action='store_true', help='Use LoRA')
    parser.add_argument('--use_qlora', action='store_true', help='Use QLora')
    parser.add_argument('--output_dir', type=str, help='Output directory')
    parser.add_argument('--save_path', type=str, help='Save json path')
    parser.add_argument('--input_json', type=str, help='Question and Answer json path')
    parser.add_argument('--img_root', type=str, help='Image Folder')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--num_crops', type=int, default=16, help='Number of maximum image crops')
    parser.add_argument('--no-tqdm', dest='tqdm', action='store_false', help='Disable tqdm')
    parser.add_argument('--lora_rank', type=int, default=64, help='LoRA rank')
    parser.add_argument(
        '--lora_alpha_ratio', type=float, default=2, help='LoRA alpha to rank ratio'
    )
    parser.add_argument('--lora_dropout', type=float, default=0.0, help='LoRA dropout')
    parser.add_argument('--freeze_vision_model', action='store_true', help='Freeze vision model')

    parser.add_argument(
        '--dataset',
        type=str,
        choices=['vqa-rad', 'slake', 'pathvqa'],
        default='vqa-rad',
        help='Dataset to evaluate: vqa-rad | slake | pathvqa (default: vqa-rad)',
    )
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--cfg", type=float, default=1.5,
                        help="CFG guidance scale γ (paper Eq. 8, default 1.5)")
    parser.add_argument("--attn", type=float, default=3.0,
                        help="Attention beta β for highlight activation (paper Eq. 9-10, default 3.0)")
    parser.add_argument("--perturb_weight", type=float, default=0.01,
                        help="Embedding perturbation α for unconditional branch (paper Eq. 7, default 0.01)")
    parser.add_argument("--use_attn_patch", action="store_true",
                        help="Apply highlight attention activation via Phi3Attention patch (Eq. 9-10)")
    parser.add_argument("--use_embed_perturb", action="store_true",
                        help="Apply embedding perturbation to unconditional branch (paper Eq. 7, alpha=perturb_weight)")
    parser.add_argument("--no_rag", action="store_true",
                        help="Disable RAG: omit retrieved caption from prompt (pure baseline)")

    args = parser.parse_args()
    args.attention_weight = args.attn

    assert args.num_crops <= 16, 'num_crops must be less than or equal to 16'
    if args.use_qlora:
        args.use_lora = True

    dataloader_config = DataLoaderConfiguration(
        dispatch_batches=None,
        split_batches=False,
        even_batches=True,
        use_seedable_sampler=True
    )
    accelerator = Accelerator(dataloader_config)

    with accelerator.local_main_process_first():
        processor = AutoProcessor.from_pretrained(
            args.model_name_or_path, trust_remote_code=True, num_crops=args.num_crops
        )
        model = create_model(
            args.model_name_or_path,
            use_flash_attention=args.use_flash_attention,
            use_qlora=args.use_qlora,
            use_bf16=args.bf16,
        )

    eval_dataset = create_dataset(args)

    num_gpus = accelerator.num_processes
    print(f'training on {num_gpus} GPUs')
    assert args.batch_size % num_gpus == 0, 'Batch size must be divisible by the number of GPUs'


    # eval after fine-tuning
    if args.use_lora:
        # first try to clear GPU memory
        del model
        __import__('gc').collect()
        torch.cuda.empty_cache()

        # reload the model for inference
        # this part also serves as an example of how to load a trained model
        use_bfloat16 = args.use_flash_attention or args.bf16
        bnb_cfg = (
            BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type='nf4',
                bnb_4bit_compute_dtype=torch.bfloat16 if use_bfloat16 else torch.float16,
            )
            if args.use_qlora
            else None
        )
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16 if use_bfloat16 else torch.float32,
            trust_remote_code=True,
            _attn_implementation='flash_attention_2' if args.use_flash_attention else 'eager',
            quantization_config=bnb_cfg,
        )
        if args.output_dir:
            model.load_adapter(args.output_dir)

    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    model = model.to(f'cuda:{local_rank}')

    # Apply attention activation patch (Eq. 9-10) — requires eager attention, not flash
    if args.use_attn_patch:
        if args.use_flash_attention:
            raise ValueError("--use_attn_patch is incompatible with --use_flash_attention. "
                             "Remove --use_flash_attention to enable the attention patch.")
        patch_phi3_attention(model, beta=args.attn)

    # Apply embedding perturbation hook (Eq. 7)
    if args.use_embed_perturb:
        patch_phi3_embedding(model, alpha=args.perturb_weight)

    evaluate(
        model,
        processor,
        eval_dataset,
        args,
        save_path=args.save_path,
        disable_tqdm=not args.tqdm,
    )



if __name__ == '__main__':
    main()
