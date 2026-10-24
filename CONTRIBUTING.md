# 팀 협업 가이드

> **Expert-CFG** 재현 프로젝트 — 팀원 온보딩 문서

---

## 1. 역할 분담

| 이름 | 담당 데이터셋 | 브랜치 |
|------|-------------|--------|
| (팀원 A) | **VQA-RAD** | `feat/vqa-rad` |
| (팀원 B) | **SLAKE** | `feat/slake` |
| (팀원 C) | **PathVQA** | `feat/pathvqa` |

각자 담당 브랜치에서 작업 → `main`으로 PR 올립니다.

---

## 2. 환경 세팅

### 2-1. 패키지 설치

```bash
pip install -r requirements.txt
```

### 2-2. 모델 가중치 다운로드 (HuggingFace)

**로컬 또는 Kaggle 환경에서 실행하세요** (이 레포의 클라우드 컨테이너에서는 huggingface.co 접근 불가).

```bash
pip install huggingface_hub
bash scripts/download_weights.sh
```

스크립트에 repo ID가 이미 채워져 있습니다:
- 베이스 모델: `microsoft/Phi-3.5-vision-instruct`
- LoRA 가중치: `ecoxial2007/Phi-3.5V-Med`

**다운로드 후 디렉토리 구조:**
```
Expert-CFG/
├── Phi-3.5-vision-instruct/      # 베이스 모델 (~8 GB)
│   ├── config.json
│   ├── model-00001-of-00002.safetensors
│   └── ...
└── lora_weights/
    └── logs_phi35_pubmed_instruct/   # Medical LoRA 가중치
        ├── adapter_config.json
        └── adapter_model.safetensors
```

### 2-3. 데이터셋 준비 (Baiduyun)

논문 저자가 제공하는 pre-extracted 파일이 필요합니다.  
각 데이터셋에는 두 가지 파일이 필요합니다:

| 파일 종류 | 내용 | 출처 |
|----------|------|------|
| Annotation JSON (features 포함) | `top_k_captions`, `highlights` 필드 포함 | Baiduyun (저자 제공) |
| 이미지 | 원본 의료 이미지 | Baiduyun 또는 공식 데이터셋 |

#### Baiduyun 링크 (README 참고)

| 데이터셋 | Baiduyun 링크 |
|---------|--------------|
| PathVQA | README에서 확인 |
| SLAKE | README에서 확인 |
| VQA-RAD (RadVQA) | README에서 확인 |
| BioMedCLIP features (공통) | README에서 확인 |

> **Baiduyun 접근이 어려울 경우**: 저자에게 직접 이메일 문의하거나,  
> 팀장이 다운로드 후 공유 드라이브(Google Drive 등)에 올려서 공유하세요.

#### 다운로드 후 배치 위치

```
Expert-CFG/data/
├── RadVQA/
│   ├── test.json          # annotation + top_k_captions + highlights
│   └── images/            # 이미지 파일들
├── Slake/
│   ├── test.json
│   └── imgs/
└── PathVQA/
    ├── test.json
    ├── test/              # 이미지 (split별 폴더)
    └── val/
```

> **JSON 필드 확인**: annotation JSON 항목은 반드시 `top_k_captions`와 `highlights` 필드를 포함해야 합니다.  
> 없으면 Expert-CFG가 작동하지 않고 기본 VQA만 실행됩니다.

---

## 3. 실행 방법

### 3-1. 기본 실행 (논문 재현 메인 설정)

```bash
torchrun --nproc_per_node=1 demo.py \
    --bf16 \
    --use_lora \
    --dataset <vqa-rad|slake|pathvqa> \
    --input_json data/<DatasetDir>/test.json \
    --img_root data/<DatasetDir>/images \
    --save_path results/<dataset>_results.json \
    --output_dir ./lora_weights/logs_phi35_pubmed_instruct \
    --cfg 1.5
```

### 3-2. Ablation 4가지 설정

논문 Table에 맞춰 4가지를 모두 실행해야 합니다:

```bash
DATASET=vqa-rad
INPUT=data/RadVQA/test.json
IMG=data/RadVQA/images
LORA=./lora_weights/logs_phi35_pubmed_instruct

# ① base + no-CFG (γ=1.0, LoRA 없음)
torchrun --nproc_per_node=1 demo.py --bf16 \
    --dataset $DATASET --input_json $INPUT --img_root $IMG \
    --cfg 1.0 \
    --save_path results/${DATASET}_1_base_noCFG.json

# ② base + CFG (γ=1.5, LoRA 없음)
torchrun --nproc_per_node=1 demo.py --bf16 \
    --dataset $DATASET --input_json $INPUT --img_root $IMG \
    --cfg 1.5 \
    --save_path results/${DATASET}_2_base_CFG.json

# ③ LoRA + no-CFG (γ=1.0)
torchrun --nproc_per_node=1 demo.py --bf16 --use_lora \
    --dataset $DATASET --input_json $INPUT --img_root $IMG \
    --output_dir $LORA --cfg 1.0 \
    --save_path results/${DATASET}_3_lora_noCFG.json

# ④ LoRA + CFG = Expert-CFG 풀 설정 (γ=1.5)
torchrun --nproc_per_node=1 demo.py --bf16 --use_lora \
    --dataset $DATASET --input_json $INPUT --img_root $IMG \
    --output_dir $LORA --cfg 1.5 \
    --save_path results/${DATASET}_4_ExpertCFG.json
```

> `--use_attn_patch` 플래그를 추가하면 논문 Eq. 9-10 (attention highlight activation)도 활성화됩니다.  
> 단, flash attention과 함께 사용 불가 — `--use_flash_attention` 빼고 실행하세요.

### 3-3. 결과 평가 (논문 Table 형식)

```bash
# 단일 파일
python evaluate.py results/vqa-rad_4_ExpertCFG.json

# 4가지 ablation 비교표
python evaluate.py \
    results/vqa-rad_1_base_noCFG.json \
    results/vqa-rad_2_base_CFG.json \
    results/vqa-rad_3_lora_noCFG.json \
    results/vqa-rad_4_ExpertCFG.json
```

출력 예시:
```
Run                           Dataset    Closed     Open  Overall     γ  LoRA  Attn
vqa-rad_1_base_noCFG          vqa-rad    72.3%    45.1%    61.2%   1.0     N     N
vqa-rad_4_ExpertCFG           vqa-rad    82.1%    55.6%    71.3%   1.5     Y     N
```

---

## 4. 브랜치 작업 방법

```bash
# 브랜치 생성 (최신 main 기준)
git fetch origin main
git checkout -b feat/vqa-rad origin/main   # 본인 담당으로 변경

# 작업 후 PR
git add src/datasets/radvqa.py   # 본인 데이터셋 파일만
git commit -m "[VQA-RAD] 설명"
git push -u origin feat/vqa-rad
```

**수정 범위:**
- ✅ 본인 담당 `src/datasets/<dataset>.py`
- ✅ 본인 브랜치의 실험 결과 메모
- ⚠️ `demo.py`, `evaluate.py`, `src/highlighter_modules/` 수정 시 → 다른 팀원에게 먼저 알리기

---

## 5. Kaggle 환경에서 실행 (권장)

로컬 GPU가 부족할 경우 Kaggle T4 (16GB) 무료 사용:

1. Kaggle Notebook → **GitHub 연결** 또는 `git clone`
2. 모델/데이터를 Kaggle Dataset으로 업로드 후 마운트
3. 위의 `torchrun` 명령어 그대로 실행

---

## 6. 결과 공유 방법

결과 JSON 파일(`results/*.json`)은 `.gitignore`에 포함되어 있어 커밋되지 않습니다.  
팀원 간 결과 공유는 다음 중 하나로:

- Google Drive / Slack 공유
- `evaluate.py` 출력 텍스트를 PR description에 붙여넣기

---

## 7. 현재 구현 상태

| 기능 | 상태 | 관련 파일 |
|------|------|----------|
| CFG 추론 (Eq. 8, γ=1.5) | ✅ 구현 | `src/highlighter_modules/guidance.py` |
| Attention highlight (Eq. 9-10) | ✅ 구현 (`--use_attn_patch`) | `src/highlighter_modules/attention_phi3.py` |
| Predictive Entropy (Eq. 4) | ✅ 구현 | `src/highlighter_modules/uncertainty.py` |
| LoRA 로드 | ✅ 구현 (`--use_lora`) | `demo.py` |
| Closed/Open accuracy 분리 평가 | ✅ 구현 | `evaluate.py` |
| Embedding perturbation (Eq. 7) | ⬜ 미구현 | — |
| RAG (BioMedCLIP 검색) | ⬜ 미구현 (pre-extracted 사용) | — |
