# Contributing Guide

## 데이터셋 담당 분담

| 데이터셋 | 브랜치 이름 |
|---------|------------|
| VQA-RAD | `feat/vqa-rad` |
| SLAKE | `feat/slake` |
| PathVQA | `feat/pathvqa` |

각자 담당 데이터셋 브랜치에서만 작업하고 `main`으로 PR을 올립니다.

## 브랜치 시작 방법

```bash
# main 최신 상태에서 브랜치 생성
git fetch origin main
git checkout -b feat/<dataset> origin/main
```

## 실행 방법

```bash
# VQA-RAD 예시
python demo.py \
  --dataset vqa-rad \
  --input_json data/RadVQA/test.json \
  --img_root data/RadVQA/images \
  --model_name_or_path ./Phi-3.5-vision-instruct \
  --save_path results/vqa-rad_results.json

# SLAKE 예시
python demo.py \
  --dataset slake \
  --input_json data/Slake/test.json \
  --img_root data/Slake/imgs \
  ...

# PathVQA 예시
python demo.py \
  --dataset pathvqa \
  --input_json data/PathVQA/test.json \
  --img_root data/PathVQA/images \
  ...
```

## 각 데이터셋 파일 위치

- `src/datasets/radvqa.py` — VQA-RAD 로더
- `src/datasets/slakevqa.py` — SLAKE 로더
- `src/datasets/pathvqa.py` — PathVQA 로더

공용 파일(`demo.py`, `src/highlighter_modules/`)은 꼭 필요한 경우에만 수정하고, 수정 시 다른 팀원에게 미리 알립니다.

## PR 규칙

- PR 제목: `[VQA-RAD] 설명` / `[SLAKE] 설명` / `[PathVQA] 설명`
- 결과 json 파일은 커밋하지 않습니다 (`.gitignore` 참고)
- 리뷰어 최소 1명 지정

## 알려진 버그 및 현재 상태

코드베이스의 주요 버그와 논문-코드 불일치는 아래를 참고하세요.

### 수정 완료 (이 베이스라인에 반영)
- `highlight_attention_mask` dtype 버그: `torch.long`으로 인해 `perturb_weight=0.01`이 0으로 잘리는 문제 → `.float()` 추가로 수정
- `--dataset` 인자 추가: `RADVQADataset` 하드코딩 → `vqa-rad|slake|pathvqa` 선택 가능
- `peft` 패키지 `requirements.txt` 누락 수정
- Attention activation 구현 (`src/highlighter_modules/attention_phi3.py`): 논문 Eq. 9-10, `--use_attn_patch` 플래그로 활성화
- Predictive Entropy 구현 (`src/highlighter_modules/uncertainty.py`): 논문 Eq. 4, 결과 json에 `predictive_entropy` 필드로 저장
- 결과 출력 형식 통일: `{meta, summary, results}` 구조로 팀 간 비교 가능
- `evaluate.py` 추가: 결과 json → accuracy 출력, 여러 파일 비교 지원

### 미구현 / 향후 작업
- Embedding perturbation (Eq. 7): 무조건 분기의 하이라이트 토큰 임베딩에 α(=0.01) 곱하기 — Phi-3.5의 visual embedding 파이프라인과 연계 필요
- RAG 미구현: `top_k_captions`와 `highlights`는 JSON에 미리 계산된 값을 사용 (BioMedCLIP + FAISS 파이프라인 미구현)
