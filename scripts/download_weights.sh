#!/usr/bin/env bash
# Download Phi-3.5-vision-instruct base model and Expert-CFG LoRA weights from HuggingFace.
#
# Usage:
#   bash scripts/download_weights.sh
#
# Requirements:
#   pip install huggingface_hub
#   huggingface-cli login   (필요한 경우)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Expert-CFG 가중치 다운로드 ==="
echo "저장 위치: $REPO_ROOT"
echo ""

# ──────────────────────────────────────────────────────────────
# 1. Phi-3.5-vision-instruct 베이스 모델
# ──────────────────────────────────────────────────────────────
MODEL_DIR="$REPO_ROOT/Phi-3.5-vision-instruct"
MODEL_HF="microsoft/Phi-3.5-vision-instruct"

echo "[1/2] 베이스 모델 다운로드: $MODEL_HF"
echo "      저장 위치: $MODEL_DIR"
echo ""

# 기존 config 파일 유지하면서 모델 가중치만 추가로 받기
huggingface-cli download \
    "$MODEL_HF" \
    --local-dir "$MODEL_DIR" \
    --local-dir-use-symlinks False \
    --exclude "*.md" "original/" 2>&1 | tail -5

echo ""
echo "[1/2] 완료"
echo ""

# ──────────────────────────────────────────────────────────────
# 2. Medical LoRA 가중치 (Phi-3.5V-Med)
#    README의 HuggingFace 링크에서 실제 repo ID를 확인 후 아래를 수정하세요.
# ──────────────────────────────────────────────────────────────
LORA_DIR="$REPO_ROOT/lora_weights/logs_phi35_pubmed_instruct"
LORA_HF="REPLACE_WITH_LORA_HUGGINGFACE_REPO_ID"   # ← README에서 확인 필요

echo "[2/2] LoRA 가중치 다운로드: $LORA_HF"
echo "      저장 위치: $LORA_DIR"
echo ""

if [ "$LORA_HF" = "REPLACE_WITH_LORA_HUGGINGFACE_REPO_ID" ]; then
    echo "⚠️  LoRA 가중치의 HuggingFace repo ID를 이 스크립트에 입력하세요."
    echo "   README > Pre-trained weights > Phi-3.5V-Med 링크 확인"
    echo "   예: LORA_HF=\"authorname/Phi-3.5V-Med\""
else
    mkdir -p "$LORA_DIR"
    huggingface-cli download \
        "$LORA_HF" \
        --local-dir "$LORA_DIR" \
        --local-dir-use-symlinks False 2>&1 | tail -5
    echo "[2/2] 완료"
fi

echo ""
echo "=== 다운로드 완료 ==="
echo ""
echo "다음 단계: Baiduyun에서 데이터셋 annotation JSON + 이미지 다운로드"
echo "  → CONTRIBUTING.md의 '데이터 준비' 섹션 참고"
