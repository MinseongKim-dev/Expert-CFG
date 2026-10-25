# PathVQA test.json 생성 — Colab 단독 실행

## Cell 1. 레포 클론 & 의존성 설치
```python
!git clone https://github.com/MinseongKim-dev/Expert-CFG.git
%cd Expert-CFG
!pip install -q open_clip_torch datasets h5py
```

## Cell 2. Google Drive 마운트 (PEIR 데이터)
```python
from google.colab import drive
drive.mount('/content/drive')

# PEIR 파일 경로 — Drive에 올린 위치로 수정
PEIR_DIR = '/content/drive/MyDrive/ExpertCFG_replication'

TEXT_FEATURES  = f'{PEIR_DIR}/text_features.h5'
FILTER_CAP     = f'{PEIR_DIR}/filter_cap.json'
MATCH_KEYWORD  = f'{PEIR_DIR}/PEIR_openai_match_keyword.json'

import os
for p in [TEXT_FEATURES, FILTER_CAP, MATCH_KEYWORD]:
    status = '✅' if os.path.exists(p) else '❌ 없음'
    print(f'{status}  {p}')
```

## Cell 3. PathVQA test.json 생성
```python
!python scripts/build_pathvqa_test_json.py \
    --text_features "{TEXT_FEATURES}" \
    --filter_cap "{FILTER_CAP}" \
    --match_keyword "{MATCH_KEYWORD}" \
    --output data/PathVQA/test.json \
    --top_k 5
```

> PathVQA test set ~6,000개 기준 T4에서 약 10분 소요

## Cell 4. 결과 확인
```python
import json
with open('data/PathVQA/test.json') as f:
    ann = json.load(f)

print(f'총 샘플: {len(ann)}')
closed = sum(1 for a in ann if a['answer_type'] == 'CLOSED')
print(f'CLOSED: {closed}  OPEN: {len(ann)-closed}')
print(f'highlights 비어있는 샘플: {sum(1 for a in ann if not a["highlights"])}')
print(f'\n샘플 예시:')
print(f'  Q: {ann[0]["question"]}')
print(f'  A: {ann[0]["answer"]} ({ann[0]["answer_type"]})')
print(f'  caption: {ann[0]["top_k_captions"][0][:80]}')
print(f'  highlights: {ann[0]["highlights"][:5]}')
```

## Cell 5. Drive에 백업
```python
import shutil
shutil.copy2('data/PathVQA/test.json', f'{PEIR_DIR}/pathvqa_test.json')
shutil.copy2('data/PathVQA/pathvqa_image_features.pkl', f'{PEIR_DIR}/pathvqa_image_features.pkl')
print('백업 완료!')
```
