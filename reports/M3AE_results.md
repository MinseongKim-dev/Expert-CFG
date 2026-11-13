# M3AE 재현 결과 (VQA-RAD / SLAKE)

작성 기준일: 2026-11-11. 이 문서의 수치는 Colab 실행 출력에서 옮긴 값이다. 확인하지 못한 항목은 "미확인"으로 적었다.

## 1. 요약 (논문 Table 1의 M3AE 수치와 비교)

| 데이터셋 | 행 | open | closed | overall | 논문 overall 대비 | 규칙 |
|---|---|---|---|---|---|---|
| VQA-RAD | **paper 프로토콜, 마지막 epoch (보고값)** | 66.48 | 82.72 | **76.27** | **−0.74** | 규칙 완화 |
| VQA-RAD | paper 프로토콜, epoch 30 (val 최고 체크포인트, 참고) | 65.92 | 80.51 | 74.72 | −2.29 | 규칙 완화 |
| VQA-RAD | 논문 | 67.23 | 83.46 | 77.01 | — | — |
| SLAKE | 원 레포 코드 그대로 (paper 방식과 동일, 아래 §4 주의) | 79.84 | 87.02 | 82.66 | −0.48 | 규칙 완화 |
| SLAKE | 논문 | 80.31 | 87.82 | 83.14 | — | — |

- VQA-RAD 논문 수치: open 67.23 / closed 83.46 / overall 77.01. SLAKE 논문 수치: open 80.31 / closed 87.82 / overall 83.14 (노트북 결과 셀의 `paper` 딕셔너리 기준).
- 모든 행은 1회 실행 결과이고 시드는 0이다. 시드 간 변동은 측정하지 않았다.

## 2. VQA-RAD 실험 설정 (paper 프로토콜)

- 데이터: 공통 분할(이미지 단위, 시드 42). train 2,749 / val 315 / test 451문항 (open 179 / closed 272).
- 학습: **train+val 3,064문항**으로 학습 (val이 학습에 포함됨). test 문항은 학습에 쓰지 않았다.
- label 사전: 원 레포 방식, 즉 train+val+test 정답 전체. candidate coverage는 test 전체 100% (451/451), open 100% (179/179), closed 100% (272/272).
  - 결과 셀 출력의 "후보(train 정답) 510개" 문구는 고정된 문구이고, 실제 후보는 train+val+test 정답이다.
- 명령 설정: `task_finetune_vqa_vqa_rad`, `per_gpu_batchsize=16`, `clip16 text_roberta`, `image_size=384`, `tokenizer=roberta-base`, 사전학습 `downloaded/pretrained/m3ae.ckpt`, 실험 이름 `vqa_rad_finetune_common_paper`, 시드 0.
- epoch: 50 (로그의 마지막 체크포인트 시각 차이로 추정. `config.py`의 `max_epoch = 50` 줄이 VQA-RAD 설정인지는 미확인).
- 속도: epoch당 약 1분 23초 (로그 기준).
- GPU 종류, 총 학습 시간: 미확인.

## 3. 체크포인트 선택 규칙

- val(315문항)이 학습 데이터에 들어 있어서 val 점수가 포화된다 (로그의 `val/the_metric` 0.98). 그래서 val 최고 체크포인트(epoch 30)는 사실상 임의의 epoch이다.
- 이 행의 보고값은 **"마지막 epoch 체크포인트(`last.ckpt`)로 test를 한 번 평가"**하는 규칙을 먼저 정한 뒤 얻은 값이다. epoch 30 값은 참고용으로만 남긴다.
- 두 체크포인트의 overall 차이는 1.55%p이다. 체크포인트 선택만으로 이 정도 변동이 생긴다.

## 4. 주의 사항 (보고 시 반드시 함께 적을 것)

1. 이 결과는 **교수님 규칙을 지킨 결과가 아니다.** val을 학습에 썼고, label 사전에 test 정답이 들어 있다. 논문 수치와의 비교용으로만 쓰고, 규칙을 지킨 결과와 별도 행으로 표기한다.
2. SLAKE 행(82.66)은 이전에 원 레포 코드를 그대로 실행한 결과를 옮긴 것이다. 원 레포 코드는 label 사전에 모든 split의 정답을 넣으므로 paper 방식과 같다. 다만 그 실행의 학습 설정이 현재 노트북과 같은지는 확인하지 못했다. 현재 노트북의 `paper` 설정으로 다시 실행해 확인하는 것이 안전하다.
3. 이전에 기록된 VQA-RAD 73.61 (overall)은 label 사전 패치 전에 이전 분할로 얻은 값이어서 보고하지 않는다.
4. 규칙을 지킨(clean) VQA-RAD 결과는 label 사전 패치 후 재학습을 하지 않아 **아직 없다.** 필요하면 `RAD_PROTOCOL = 'clean'`으로 한 번 더 학습해야 한다 (train만 학습, label 사전은 train 정답만, 후보에 없는 test 정답은 0점 처리).

## 5. 재현 방법 (VQA-RAD)

1. 데이터 셀에서 `RAD_PROTOCOL = 'paper'`로 설정하고 실행 (출력에 "학습 3064문항(train+val)"이 나와야 함).
2. 4-0 패치 셀 실행 (paper에서는 원본 유지) → 4절 전처리로 arrow 재생성.
3. VQA-RAD 학습 셀 실행 (SLAKE 학습 셀은 실행하지 않음).
4. 학습이 끝나면 val 최고 체크포인트로 test가 자동 실행된다. 보고값은 §5-2 셀(`last.ckpt`로 test 한 번 평가)의 출력이다.
5. 결과 셀에서 coverage와 논문 대비 격차를 확인한다.

## 6. 미확인 항목

- 사용한 GPU와 총 학습 시간
- `max_epoch`가 50인지 (VQA-RAD 설정 기준)
- 틀린 문항 분석 (노트북 맨 끝 `# 5-3.` 셀로 볼 수 있으나 출력은 아직 받지 못함)
- 시드 간 변동
