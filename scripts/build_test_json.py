"""
VQA-RAD test.json 생성 스크립트 (top_k_captions, highlights 포함)

필요한 파일:
  - data/RadVQA/features/chest.pkl  (BiomedCLIP 이미지 피처)
  - data/RadVQA/text_features.h5    (PEIR 텍스트 피처)
  - data/filter_cap.json            (PEIR GPT3.5 캡션)
  - data/PEIR_openai_match_keyword.json  (highlights용)

Usage (Kaggle):
  python scripts/build_test_json.py \
      --features_dir /kaggle/input/expert-cfg-data/RadVQA/features \
      --text_features /kaggle/input/expert-cfg-data/text_features.h5 \
      --filter_cap /kaggle/input/expert-cfg-data/filter_cap.json \
      --match_keyword /kaggle/input/expert-cfg-data/PEIR_openai_match_keyword.json \
      --output data/RadVQA/test.json \
      --top_k 5
"""

import argparse
import json
import pickle
import os
import numpy as np
import h5py
from tqdm import tqdm


def load_image_features(features_dir):
    """features/ 폴더의 pkl 파일들을 로드해 {img_id: feature} dict 반환"""
    all_features = {}
    for fname in os.listdir(features_dir):
        if not fname.endswith(".pkl"):
            continue
        fpath = os.path.join(features_dir, fname)
        with open(fpath, "rb") as f:
            data = pickle.load(f)
        # dict 또는 list[(img_id, feat)] 형태 처리
        if isinstance(data, dict):
            all_features.update(data)
        elif isinstance(data, list):
            for item in data:
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    all_features[item[0]] = item[1]
    print(f"이미지 피처 로드: {len(all_features)}개")
    return all_features


def load_text_features(h5_path):
    """text_features.h5 로드 → {caption_id_or_key: feature}"""
    features = {}
    keys = []
    with h5py.File(h5_path, "r") as f:
        print(f"h5 키 목록: {list(f.keys())[:10]}")
        # 구조 파악
        for key in f.keys():
            val = f[key][()]
            features[key] = val
            keys.append(key)
    print(f"텍스트 피처 로드: {len(features)}개")
    return features, keys


def cosine_similarity_batch(query, matrix):
    """query (D,) vs matrix (N, D) → similarities (N,)"""
    query = query / (np.linalg.norm(query) + 1e-8)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    normed = matrix / norms
    return normed @ query


def build_test_json(features_dir, text_features_path, filter_cap_path,
                    match_keyword_path, output_path, top_k=5):

    # 1. 이미지 피처 로드
    img_features = load_image_features(features_dir)

    # 2. PEIR 텍스트 피처 로드
    text_features, text_keys = load_text_features(text_features_path)

    # 3. filter_cap 로드 → {image_id: caption}
    with open(filter_cap_path) as f:
        filter_cap_raw = json.load(f)

    if "annotations" in filter_cap_raw:
        filter_cap = {str(a["image_id"]): a["caption"]
                      for a in filter_cap_raw["annotations"]}
    else:
        filter_cap = {str(k): v for k, v in filter_cap_raw.items()}

    print(f"filter_cap 캡션: {len(filter_cap)}개")

    # 4. match_keyword 로드 → {caption_text: {keywords, match_keyword}}
    with open(match_keyword_path) as f:
        match_keyword = json.load(f)
    print(f"match_keyword: {len(match_keyword)}개")

    # 5. 텍스트 피처를 matrix로 변환
    text_feat_matrix = np.stack(list(text_features.values()), axis=0).astype(np.float32)
    text_feat_keys = list(text_features.keys())

    # 6. VQA-RAD 데이터 로드 (HuggingFace)
    try:
        from datasets import load_dataset
        ds = load_dataset("flaviagiammarino/vqa-rad", split="test")
        print(f"VQA-RAD test set: {len(ds)}개")
    except Exception as e:
        print(f"HuggingFace 로드 실패: {e}")
        print("VQA-RAD 데이터셋을 먼저 설치하세요: pip install datasets")
        return

    # 7. 각 질문에 대해 top_k 캡션 + highlights 계산
    annotations = []
    skipped = 0

    for item in tqdm(ds, desc="Processing"):
        img_id = item.get("image_name") or item.get("img_id") or item.get("image_id")
        img_id = str(img_id)

        if img_id not in img_features:
            skipped += 1
            # 피처 없으면 빈 캡션으로 대체
            annotations.append({
                "img_id": img_id,
                "question": item["question"],
                "answer": item["answer"],
                "answer_type": item.get("answer_type", "OPEN"),
                "top_k_captions": [""],
                "highlights": [],
            })
            continue

        # 이미지 피처로 PEIR 텍스트 피처 검색
        img_feat = np.array(img_features[img_id]).astype(np.float32)
        if img_feat.ndim > 1:
            img_feat = img_feat.flatten()

        sims = cosine_similarity_batch(img_feat, text_feat_matrix)
        top_k_idx = np.argsort(sims)[::-1][:top_k]

        top_captions = []
        all_highlights = []

        for idx in top_k_idx:
            peir_key = text_feat_keys[idx]
            cap = filter_cap.get(str(peir_key), "")
            if cap:
                top_captions.append(cap)

            # highlights: match_keyword에서 키워드 찾기
            for caption_text, info in match_keyword.items():
                if str(peir_key) in [str(x) for x in info.get("img_ids", [])]:
                    kws = info.get("match_keyword", [])
                    all_highlights.extend(kws)
                    break

        if not top_captions:
            top_captions = [""]

        # 중복 제거
        all_highlights = list(dict.fromkeys(all_highlights))

        annotations.append({
            "img_id": img_id,
            "question": item["question"],
            "answer": item["answer"],
            "answer_type": item.get("answer_type", "OPEN"),
            "top_k_captions": top_captions,
            "highlights": all_highlights,
        })

    print(f"\n완료: {len(annotations)}개 (피처 없어 스킵: {skipped}개)")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(annotations, f, indent=2, ensure_ascii=False)
    print(f"저장: {output_path}")

    # 샘플 확인
    sample = annotations[0]
    print(f"\n샘플:")
    print(f"  img_id: {sample['img_id']}")
    print(f"  question: {sample['question']}")
    print(f"  top_k_captions[0]: {sample['top_k_captions'][0][:80]}...")
    print(f"  highlights: {sample['highlights'][:5]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features_dir", default="data/RadVQA/features")
    parser.add_argument("--text_features", default="data/RadVQA/text_features.h5")
    parser.add_argument("--filter_cap", default="data/filter_cap.json")
    parser.add_argument("--match_keyword", default="data/PEIR_openai_match_keyword.json")
    parser.add_argument("--output", default="data/RadVQA/test.json")
    parser.add_argument("--top_k", type=int, default=5)
    args = parser.parse_args()

    build_test_json(
        args.features_dir,
        args.text_features,
        args.filter_cap,
        args.match_keyword,
        args.output,
        args.top_k,
    )
