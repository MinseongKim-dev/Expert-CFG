"""
PathVQA test.json 생성 스크립트 (top_k_captions, highlights 포함)

BioMedCLIP으로 PathVQA 이미지 피처를 추출하고,
VQA-RAD와 동일한 PEIR 데이터베이스로 top_k 캡션 + highlights를 계산합니다.

필요한 파일 (VQA-RAD용과 동일):
  - data/RadVQA/text_features.h5        (PEIR 텍스트 피처)
  - data/filter_cap.json                (PEIR GPT3.5 캡션)
  - data/PEIR_openai_match_keyword.json (highlights용)

Usage (Colab/Kaggle):
  python scripts/build_pathvqa_test_json.py \
      --text_features /kaggle/input/expert-cfg-data/text_features.h5 \
      --filter_cap /kaggle/input/expert-cfg-data/filter_cap.json \
      --match_keyword /kaggle/input/expert-cfg-data/PEIR_openai_match_keyword.json \
      --output data/PathVQA/test.json \
      --top_k 5
"""

import argparse
import json
import os
import pickle
import numpy as np
import h5py
import torch
from pathlib import Path
from tqdm import tqdm
from PIL import Image


# ── BioMedCLIP 로드 ──────────────────────────────────────────────────────────

def load_biomedclip():
    try:
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )
        model.eval()
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model = model.to(device)
        print(f'BioMedCLIP 로드 완료 (device={device})')
        return model, preprocess, device
    except Exception as e:
        raise RuntimeError(f'BioMedCLIP 로드 실패: {e}\n  pip install open_clip_torch')


@torch.no_grad()
def extract_image_features(model, preprocess, images, device,
                           batch_size=64, checkpoint_path=None):
    """PIL Image 리스트 → numpy feature matrix (N, D).

    checkpoint_path: .npy 경로. 배치마다 덮어쓰며 저장해두고,
    재실행 시 이미 완료된 배치는 건너뜁니다.
    """
    total = len(images)
    feat_dim = None

    # 기존 체크포인트 로드
    done_batches = 0
    all_feats = []
    if checkpoint_path and Path(checkpoint_path).exists():
        saved = np.load(checkpoint_path)
        done_batches = len(saved) // batch_size  # 완전히 끝난 배치 수
        loaded = done_batches * batch_size
        all_feats = [saved[:loaded]]
        feat_dim = saved.shape[1]
        print(f'체크포인트 복원: {loaded}/{total}개 완료, {done_batches}배치 스킵')

    start_batch = done_batches
    for bi in tqdm(range(start_batch, (total + batch_size - 1) // batch_size),
                   desc='BioMedCLIP 피처 추출',
                   initial=start_batch,
                   total=(total + batch_size - 1) // batch_size):
        i = bi * batch_size
        batch = images[i:i + batch_size]
        tensors = torch.stack([preprocess(img) for img in batch]).to(device)
        feats = model.encode_image(tensors)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_feats.append(feats.cpu().float().numpy())

        if checkpoint_path:
            np.save(checkpoint_path, np.concatenate(all_feats, axis=0))

    return np.concatenate(all_feats, axis=0)


# ── PEIR 데이터 로드 ─────────────────────────────────────────────────────────

def load_text_features(h5_path):
    features, keys = {}, []
    with h5py.File(h5_path, 'r') as f:
        for key in f.keys():
            features[key] = f[key][()]
            keys.append(key)
    print(f'텍스트 피처 로드: {len(features)}개')
    return features, keys


def cosine_similarity_batch(query, matrix):
    query = query / (np.linalg.norm(query) + 1e-8)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return (matrix / norms) @ query


# ── 메인 ─────────────────────────────────────────────────────────────────────

def build_pathvqa_test_json(text_features_path, filter_cap_path,
                             match_keyword_path, output_path, top_k=5):

    # 1. PathVQA 로드
    from datasets import load_dataset
    print('PathVQA 로드 중...')
    ds = load_dataset('flaviagiammarino/path-vqa', split='test')
    print(f'PathVQA test set: {len(ds)}개')

    # 2. BioMedCLIP으로 이미지 피처 추출 (배치마다 체크포인트 저장)
    out_dir = Path(output_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    feat_ckpt = out_dir / 'pathvqa_img_feats_ckpt.npy'   # 배치 단위 체크포인트
    feat_save = out_dir / 'pathvqa_image_features.pkl'    # 최종 완성본

    if feat_save.exists():
        print(f'이미지 피처 캐시 발견, 로드: {feat_save}')
        with open(feat_save, 'rb') as f:
            img_feats = pickle.load(f)['features']
        print(f'이미지 피처 shape: {img_feats.shape}')
    else:
        model, preprocess, device = load_biomedclip()
        images = [row['image'].convert('RGB') for row in tqdm(ds, desc='이미지 로드')]
        img_feats = extract_image_features(
            model, preprocess, images, device,
            checkpoint_path=str(feat_ckpt),
        )
        print(f'이미지 피처 shape: {img_feats.shape}')
        with open(feat_save, 'wb') as f:
            pickle.dump({'features': img_feats, 'indices': list(range(len(ds)))}, f)
        print(f'이미지 피처 저장: {feat_save}')
        if feat_ckpt.exists():
            feat_ckpt.unlink()  # 완료 후 임시 체크포인트 삭제

    # 3. PEIR 텍스트 피처 로드
    text_features, text_keys = load_text_features(text_features_path)
    text_feat_matrix = np.stack(list(text_features.values()), axis=0).astype(np.float32)

    # 4. filter_cap / match_keyword 로드
    with open(filter_cap_path) as f:
        raw = json.load(f)
    if 'annotations' in raw:
        filter_cap = {str(a['image_id']): a['caption'] for a in raw['annotations']}
    else:
        filter_cap = {str(k): v for k, v in raw.items()}
    print(f'filter_cap: {len(filter_cap)}개')

    with open(match_keyword_path) as f:
        match_keyword = json.load(f)
    print(f'match_keyword: {len(match_keyword)}개')

    # match_id 역인덱스: text_features.h5 키 = match_id → img_ids/keywords
    mid_to_img_ids: dict = {}
    mid_to_keywords: dict = {}
    for info in match_keyword.values():
        mid = info.get('match_id')
        if mid is None:
            continue
        mid_key = str(mid)
        if mid_key not in mid_to_img_ids:
            mid_to_img_ids[mid_key] = info.get('img_ids', [])
            mid_to_keywords[mid_key] = info.get('match_keyword', [])
        else:
            mid_to_img_ids[mid_key].extend(info.get('img_ids', []))
            mid_to_keywords[mid_key].extend(info.get('match_keyword', []))
    print(f'역인덱스 구성: {len(mid_to_img_ids)}개 match_id')

    # 5. 각 샘플에 top_k 캡션 + highlights 계산
    annotations = []
    for i, row in enumerate(tqdm(ds, desc='RAG 계산')):
        img_feat = img_feats[i]
        sims = cosine_similarity_batch(img_feat, text_feat_matrix)
        top_k_idx = np.argsort(sims)[::-1][:top_k]

        top_captions, all_highlights = [], []
        for idx in top_k_idx:
            mid_key = str(text_keys[idx])  # match_id

            img_ids = mid_to_img_ids.get(mid_key, [])
            for img_id in img_ids:
                cap = filter_cap.get(str(img_id), '')
                if cap:
                    top_captions.append(cap)
                    break

            all_highlights.extend(mid_to_keywords.get(mid_key, []))

        if not top_captions:
            top_captions = ['']
        all_highlights = list(dict.fromkeys(all_highlights))

        # PathVQA answer_type: yes/no면 CLOSED, 나머지 OPEN
        answer = str(row['answer']).strip().lower()
        answer_type = 'CLOSED' if answer in ('yes', 'no') else 'OPEN'

        annotations.append({
            'img_id': f'test_{i:05d}',
            'question': row['question'],
            'answer': row['answer'],
            'answer_type': answer_type,
            'top_k_captions': top_captions,
            'highlights': all_highlights,
        })

    # 6. 저장
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f'\n저장 완료: {output_path}  ({len(annotations)}개)')

    # 통계
    closed = sum(1 for a in annotations if a['answer_type'] == 'CLOSED')
    opened = len(annotations) - closed
    empty_hl = sum(1 for a in annotations if not a['highlights'])
    print(f'  CLOSED={closed}  OPEN={opened}')
    print(f'  highlights 비어있는 샘플: {empty_hl}')
    print(f'  샘플 예시: {annotations[0]}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_features', required=True)
    parser.add_argument('--filter_cap', required=True)
    parser.add_argument('--match_keyword', required=True)
    parser.add_argument('--output', default='data/PathVQA/test.json')
    parser.add_argument('--top_k', type=int, default=5)
    args = parser.parse_args()

    build_pathvqa_test_json(
        args.text_features,
        args.filter_cap,
        args.match_keyword,
        args.output,
        args.top_k,
    )
