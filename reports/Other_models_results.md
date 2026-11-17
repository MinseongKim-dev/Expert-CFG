# MUMC · M3AE · LLaVA-Med 재현 시도 정리 (VQA-RAD / SLAKE)

작성 기준일: 2026-11-15. 수치는 Colab 실행 출력에서 옮겼거나 그 출력으로 계산한 값이다. "계산값"은 보고된 수치에서 이 문서 작성 중 산술로 구한 것이고, 확인하지 못한 항목은 "미확인"으로 적었다.
Phi-3.5V 쪽은 `Phi35V_results.md`(베이스라인)와 `RAG_CFG_results.md`(RAG · CFG)에 있다.

## 0. 한눈에 보기 (논문 Table 1 overall 대비, 2%p가 팀 목표)

| 모델 | 데이터셋 | 행 | overall | 논문 | 격차 | 규칙 | 판정 |
|---|---|---|---|---|---|---|---|
| M3AE | VQA-RAD | paper 프로토콜, 마지막 epoch | 76.27 | 77.01 | −0.74 | 규칙 완화 | 이내 (규칙 준수 행은 **없음**) |
| M3AE | SLAKE | 원 레포 코드 그대로 | 82.66 | 83.14 | −0.48 | 규칙 완화 | 이내 (규칙 준수 행은 **없음**) |
| MUMC | VQA-RAD | clean, val 선택(ep28) | 73.39 (MUMC 자체 채점) | 79.20 | **−5.81** | 규칙 준수 | **미달** |
| MUMC | SLAKE | val 선택(ep22, val 동점 22/37/38/39 중 이른 epoch) | 84.45 (자체 채점) | 84.90 | −0.45 | 규칙 준수 | 이내 |
| MUMC | SLAKE | A: 원래 프로토콜(train+val 학습), 마지막 ep39 | 85.67 (자체 채점) | 84.90 | +0.77 | 규칙 완화 | 이내 |
| MUMC | VQA-RAD | A: paper 프로토콜(train+val 학습, 후보에 test 정답), 마지막 ep39 | 73.39 (자체 채점; open 61.45 / closed 81.25) | 79.20 | **−5.81** (open −10.05) | 규칙 완화 | **미달** (후보·프로토콜로 설명되지 않음) |
| LLaVA-Med | VQA-RAD | 공식 delta, greedy, closed 접미어 | 69.69 (S1+S2 70.02) | 75.19 | **−5.50** (−5.17) | 규칙 준수 (학습 없음) | **미달** |
| LLaVA-Med | SLAKE | QLoRA 근사 (공식 delta 사용 불가) | 79.40 | 83.97 | **−4.57** | 규칙 준수 | **미달** |

- **학습을 다시 해야 규칙 준수 행이 생기는 것은 M3AE 하나뿐이다.** MUMC · LLaVA-Med는 규칙 준수 행이 이미 있다.
- 모두 1회 실행이다. 시드 간 변동은 측정하지 않았다.
- 논문 수치(Open / Closed / Overall)는 팀 가이드(`MODEL_RUN_GUIDE.md`)가 인용한 값이다: MUMC VQA-RAD 71.50 / 84.20 / 79.20, SLAKE overall 84.90만 있음. M3AE VQA-RAD 67.23 / 83.46 / 77.01, SLAKE 80.31 / 87.82 / 83.14. LLaVA-Med 7B VQA-RAD 61.52 / 84.19 / 75.19, SLAKE 83.08 / 85.34 / 83.97.

## 1. 공통 규칙과 채점

- **프로토콜.** train으로만 학습, val로 모델·체크포인트 선택, test는 마지막에 한 번. val을 학습에 쓰거나 후보 목록에 test 정답을 넣거나 test로 설정을 고른 행은 규칙 완화로 따로 표기한다.
- **채점이 모델마다 다르다.**
  - Phi 방식: closed는 정답 단어가 예측 단어에 있으면 정답, open은 토큰 recall.
  - 통일 방식(`vqa_scoring.summarize_unified`): closed 정확도(첫 yes/no 단어 기준) + open exact match, overall은 문항 수 가중.
  - MUMC 자체 채점: 정답 문자열 완전 일치(`vqaEval.py`).
- **S1+S2 보조 채점**(open만: 표기 차이, `plane`·`weighted`·`mri` 생략 허용)은 Phi의 VQA-RAD test 오답에서 만든 규칙이라 규칙 완화다. VQA-RAD에서 모델마다 overall을 약 +0.3~+1.2 올리고 SLAKE는 바꾸지 않는다(LLaVA VQA-RAD는 +0.33). 항상 원본 채점과 병기한다.
- **best epoch를 test로 고르는 구조.** MUMC `vqaEvaluate.py`는 모든 epoch 결과를 overall로 정렬하고, M3AE는 `*_best_epoch`를 기록한다. 원 논문 수치가 test로 고른 epoch의 값일 가능성이 있어(원 논문의 선택 방식은 미확인), 마지막 epoch와 val 선택 epoch를 같이 기록했다.

## 2. M3AE (분류형, ViLT·METER 기반)

상세는 `M3AE_results.md`. 요약:

| 데이터셋 | 행 | open | closed | overall | 논문 대비 | 규칙 |
|---|---|---|---|---|---|---|
| VQA-RAD | paper 프로토콜, 마지막 epoch (보고값) | 66.48 | 82.72 | 76.27 | −0.74 | 완화 |
| VQA-RAD | paper 프로토콜, epoch 30 (val 최고, 참고) | 65.92 | 80.51 | 74.72 | −2.29 | 완화 |
| SLAKE | 원 레포 코드 그대로 | 79.84 | 87.02 | 82.66 | −0.48 | 완화 |

- **paper 프로토콜 = 원 레포 방식**: train+val 3,064문항으로 학습(val이 학습에 들어가 val 점수가 포화, 0.98), label 사전에 train+val+test 정답 전체. test 문항 자체는 학습에 쓰지 않았다. 실험: `task_finetune_vqa_vqa_rad`, batch 16, 이미지 384, 시드 0, 약 50 epoch(미확인), epoch당 약 1분 23초.
- 두 체크포인트(마지막 vs epoch 30)의 overall 차이가 1.55%p라 체크포인트 선택만으로 그 정도 변동이 생긴다.
- **규칙 준수(clean) 행은 없다.** label 사전 패치 후 재학습하지 않았다. `RAD_PROTOCOL='clean'`(train만 학습, label 사전은 train 정답만, 후보에 없는 test 정답은 0점)으로 한 번 더 학습해야 한다. SLAKE 행은 이전 실행을 옮긴 것이라 현재 노트북 설정과 같은지도 미확인이다.
- 이전에 기록된 VQA-RAD 73.61은 label 사전 패치 전, 이전 분할의 값이라 보고하지 않는다.
- 예측 파일이 저장되어 있는지, GPU와 총 학습 시간, `max_epoch`는 미확인.

## 3. MUMC (분류형, ALBEF 기반)

노트북 `MUMC_expert_cfg_reproduction.ipynb`. 사전학습 가중치(`med_pretrain_29.pth`)만 공개되어 데이터셋별 파인튜닝을 했다.

**노트북에서 한 일**
- 원 코드 패치: `--output_dir`이 `/mnt/sda/lpf/...`로 덮어써지는 문제와 `ruamel_yaml` import 수정, `--evaluate`가 평가 없이 종료하는 점 확인.
- **val 선택 패치**(`train_vqa.py`): 환경변수 `MUMC_VAL_FILE`이 있으면 매 epoch val을 평가해 `*_val_result_<epoch>.json`으로 저장하고 val 정답 수(`pre_answer` 완전 일치)가 최고인 epoch의 가중치를 보관했다가 학습 후 그 epoch의 test를 한 번만 평가한다. 마지막 epoch의 test도 저장한다(`EVAL_EVERY=N`이면 진단용으로 N epoch마다 추가).
- **프로토콜 스위치** `RAD_PROTOCOL`: `clean`(train만, 후보=train 정답), `trainval`(train+val 학습), `paper`(train+val 학습, 후보=train+val+test 정답, 규칙 완화).
- SLAKE는 train만 쓰도록 설정 패치(원 레포는 train+val로 학습하고 `answer_trainval_list`를 씀).
- 분할: 공통 분할(train 2,749 / val 315 / test 451). Drive 출력 폴더: `rad_common`(clean), `rad_common_paper`, `rad_std`(이전 분할), `slake`(이전 실행, val 없음), `slake_val_sel`(val 선택).

**VQA-RAD 결과 (clean, MUMC 자체 채점 = 통일 방식의 closed 정확도 / open EM)**

| 행 | closed | open | overall | 논문 대비 |
|---|---|---|---|---|
| val 선택 epoch **28** (val 161/315) | 83.09 | 58.66 | **73.39** | **−5.81** (open −12.84, closed −1.11) |
| 마지막 epoch 39 (val 157/315) | 82.72 | 59.78 | 73.61 | −5.59 |

- val 최고 epoch는 `pre_answer` 기준으로 28(161)이고 29도 같은 점수지만 동점이면 앞선 epoch를 택하는 규칙이라 28이다. 다른 정규화(단순 소문자 비교)로 세면 29가 최고로 나와 28과 29는 사실상 구분되지 않는다(1문항 차이).
- open **recall**(참조 채점)로는 ep28 63.17, ep39 64.00이다. ep28의 closed 83.09와 open recall 63.17을 문항 수로 가중하면 overall **약 75.2 (계산값, 논문 대비 약 −4.0)** 이다. 이전 정리에서 "MUMC VQA-RAD 약 −2.8"이라고 적은 값은 통일 채점(+S1+S2)의 값으로 보였지만 직접 재확인하지 못했다. **정확한 통일 채점 재계산이 아직 없다.**
- **`rad_common_paper`(규칙 완화)를 채점했다:** train+val 학습, 후보 train+val+test 정답(open 상한 100%)인데도 마지막 ep39가 open 61.45 / closed 81.25 / overall 73.39(−5.81)다. clean(ep28 open EM 58.66)보다 open이 +2.8만 올랐다. 저장된 test epoch(4, 9, …, 39) 중 최고도 ep39다. 따라서 open 격차(−10.05)는 후보 목록이나 학습 데이터로 설명되지 않는다. 같은 예측의 phi(recall 계열) 채점은 B의 clean ep28이 75.33(−3.87), 통일 채점(closed accuracy + open EM)은 73.39다. MUMC 공식 저장소는 사전학습 가중치만 공개하고 VQA-RAD · SLAKE 미세조정 하이퍼파라미터(epoch, lr, 배치)와 보고 epoch 선택 방식을 적지 않았다. 후보 상한(test open 정답이 후보에 있는 비율): train만 74.3%, train+val 78.2%, paper 방식 100%.
- **MUMC 원 논문(arXiv 2307.05314) 대조 (PDF 본문 직접 확인):** Table 1의 MUMC는 VQA-RAD open 71.5 / closed 84.2 / overall 79.2, SLAKE overall 84.9로 `MODEL_RUN_GUIDE.md` 인용값과 같다. 지표는 accuracy이고 "생성한 답과 후보 목록 답의 유사도를 계산해 최고 점수를 최종 답으로 선택"한다고 적혀 있다. 미세조정 설정은 **30 epoch, 배치 8, AdamW, lr 2e-5에서 1e-8까지 감쇠, 이미지 256→384(위치 인코딩 보간)**이다. 레포의 `configs/VQA.yaml`은 batch_size_train 8, init_lr 2e-5, min_lr 1e-8, image_res 384, k_test 128로 논문과 같고 `max_epoch`만 40(우리 실행도 40)이다. 논문은 weight decay를 미세조정에 대해 따로 적지 않았다(사전학습 0.002, 레포 기본 0.05). VQA-RAD는 3,064문항 중 451이 test이고 val 사용은 적혀 있지 않으며, 보고 epoch 선택 방식도 적혀 있지 않다. 따라서 하이퍼파라미터 차이는 격차(−5.81)의 설명이 되지 않고, 남는 후보는 사전학습 체크포인트(공개된 `med_pretrain_29.pth`가 논문에서 쓴 것과 같은지 미확인), 시드, VQA-RAD 학습 분할(논문 분할이 3,064 중 어떻게 나뉘는지 미확인), 30 대 40 epoch다. `rad_common_paper_wrong_ep39.json`은 이름대로 잘못 만든 결과 파일로 보고 제외했다.

**SLAKE 결과 (MUMC 자체 채점, `slake_val_sel/result/all_acc.json`)**

| 행 | open | closed | overall | 논문 overall 84.90 대비 |
|---|---|---|---|---|
| 파일 epoch 22 (val 최고 후보) | 80.00 | 91.35 | 84.45 | −0.45 |
| 마지막 epoch 39 | 81.40 | 91.35 | 85.30 | +0.40 |

- **val 최고 epoch 22는 val 결과 파일로 재계산해 확정했다**(`slake_val_sel/result/*_val_result_<epoch>.json`, val 1,053문항). 다만 epoch 22, 37, 38, 39가 모두 895로 동점이고 "동점이면 이른 epoch"라는 사전 규칙이 22를 골랐다. val은 17 epoch 이후 891~895로 평평해서 22와 39를 구분하지 못한다. 두 epoch의 test 차이는 0.85%p(약 9문항)이다.
- test 점수는 epoch 5 이후 계속 오르는데(80.4 → 85.3) val 선택은 22를 골랐다. val과 test의 추세가 다르다. 논문에는 overall만 있어 open / closed 비교는 못 했다.
- **A(논문 재현 판정, 이전 실행 `output/slake`: train+val 학습, 후보 train+val, MUMC 자체 채점):** 마지막 ep39 open 81.40 / closed 92.31 / overall **85.67 (+0.77)**, 저장된 test epoch(0, 4, …, 39) 중 최고 ep19 86.05 (+1.15, test로 고른 값). 규칙 완화 행이다.
- 통일 채점(closed accuracy, open EM · F1): val 선택 ep22 closed 91.35 / open EM 82.02 / F1 86.95 / overall 85.67, 마지막 ep39 91.35 / 83.41 / 88.36 / 86.52. 자체 EM(84.45)보다 +1.22 높은 것은 정규화(관사 제거 등) 차이로 보이나 항목별로 분해하지 않았다.

## 4. LLaVA-Med 7B (생성형)

합친 노트북 `LLaVA_Med_reproduction.ipynb`(VQA-RAD · SLAKE · 부록). 정리 전 전체 이력은 `LLaVA_Med_expert_cfg_reproduction.ipynb`(70셀)에 남아 있다.

**환경.** v1.0.0 태그, Python 3.10 venv, `torch==2.0.0`, `transformers`는 git 커밋 `cae78c46`(Phi 환경과 충돌해 별도 venv). 추론은 `/content/venv_llava`, QLoRA 학습은 별도 `/content/venv_qlora`.

### 4-1. VQA-RAD: 공식 delta 병합 (학습 없음)

- `katielink/llava-med-7b-vqarad-delta`를 LLaMA-7B에 병합. 레포 추론 코드는 기본이 샘플링(T=0.7)이라 **greedy로 패치**했다.
- closed 질문에는 레포 샘플 프롬프트 형식의 접미어 ` Please choose from the following two options: [yes, no]`를 붙였다. 접미어 없이 물으면 yes/no로 답하지 않아 closed가 크게 떨어진다. test로 고른 설정은 없다.

| 채점 | open | closed | overall | 논문(61.52 / 84.19 / 75.19) 대비 |
|---|---|---|---|---|
| 원본 | 66.08 | 72.06 | 69.69 | −5.50 |
| S1+S2 | 66.91 | 72.06 | 70.02 | −5.17 |

- 신뢰구간(원본): open 59.57~72.49(논문 61.52 안), closed 66.54~77.21(논문 84.19 **밖**), overall 65.43~73.79(논문 75.19 **밖**).
- **격차는 closed에서 나온다**(−12.13). open은 오히려 논문보다 +4.56 높다.
- 부록 A(규칙 완화 행, test 기반 탐색): 예전 아카이브 노트북(`LLaVA_Med_expert_cfg_reproduction.ipynb`)의 C-19 셀 출력에서 결과를 찾았다. yes/no 형태 248문항을 두 가지 선택지 순서로 추론해 보정한 결과, 기준(접미어 `[yes, no]`, greedy) closed 73.90%(yes/no 정답 190개)가 두 순서 평균 + prior 규칙으로 76.10%(196개), oracle 상한 76.84%(198개)였다. 가장 좋은 상한도 논문 closed 84.19에 닿지 않으므로 편향 보정이나 프롬프트 변형으로는 격차를 메우지 못한다. 이 출력의 기준 closed(73.90)는 현재 예측 파일의 phi closed(72.06)와 달라서 다른 실행으로 보이나 확인하지 못했다.
- **closed 오답 분석(현재 예측 파일, VQA-RAD):** 정답이 yes/no인 251문항의 정확도 76.5%이고 오답 59건 중 50건이 "정답 no인데 yes로 답함"이다(예측 yes 63.3% 대 정답 yes 47.0%, yes 편향). 정답이 yes/no가 아닌 21문항("A or B" 질문)에는 접미어 `[yes, no]`가 붙어 `no`·`yes`로 답했다. LLaVA-Med 자체 평가 코드 이식본(closed는 예측에 yes/no가 있어야 정답)으로는 이 21문항이 모두 0점이라 closed 70.59, overall 68.65(−6.54), phi로는 closed 72.06, overall 69.69(−5.50)이다. SLAKE는 yes 편향이 없고(예측 yes 48.7% 대 정답 49.3%) 오답이 no→yes 37건, yes→no 39건으로 대칭이다.

### 4-2. SLAKE: QLoRA 근사 (공식 delta에 가중치가 없음)

- `katielink/llava-med-7b-slake-delta`는 README만 있고 가중치가 없다. 대신 `microsoft/llava-med-7b-delta`(의료 instruct 모델)를 LLaMA-7B에 병합하고, **QLoRA로 SLAKE 영어 train을 학습**했다. 논문의 전체 파인튜닝과 같은 조건이 아니라서 결과표에 "QLoRA 근사"로 표기한다.
- 설정: 4비트 NF4(double quant) + LoRA r=64, alpha 16, dropout 0.05, LLaMA 선형층 7종. 2단계에서는 `mm_projector`도 함께 학습(`final_proj`). 1단계로 학습률 3개를 val로 비교하고, 고른 학습률 하나를 `FINAL_EPOCHS`로 길게 학습해 val로 epoch를 골랐다. test는 선택 후 한 번만 읽었다.
- peft가 `modules_to_save` 가중치를 키 이름에 그 문구 없이 저장하는 문제로 병합 스크립트가 "mm_projector 없음"이라고 잘못 출력했다. 병합 모델과 어댑터의 `mm_projector` 최대 차이가 0.0임을 확인해 프로젝터가 반영됨을 검증했고, 필터를 수정했다.

| 행 | closed | open (계산값) | overall | 논문(83.08 / 85.34 / 83.97) 대비 |
|---|---|---|---|---|
| QLoRA 근사, val 선택 | 75.72 | 약 81.8 | **79.40** | **−4.57** (closed 약 −9.6, open 약 −1.3) |

- **yes/no 편향 보정(C-20, 부록 B)은 효과가 없었다.** yes/no 로그확률 임계값을 val로 정해 closed를 다시 판정했지만 재판정 대상 정답 수가 279 → 277 / 355(closed 75.72 → 75.24, overall 79.40 → 79.21)로 개선이 없었다. 선택지가 yes/no가 아닌 closed 문항은 36/61 정답이었다. 격차는 yes/no 편향이 아니라 정답 판단 자체의 정확도 문제로 보이며, QLoRA 근사의 한계로 추정한다(검증하지 않음).
- **79.40 행은 6 epoch 학습이다(Drive 기록으로 확인).** `qlora_slake/final_proj/lr0.0002/run_info.json`이 epochs 6.0, max_steps 1842, n_train 4919, n_val 1053이고, best와 last 체크포인트가 모두 `checkpoint-1842`(global_step 1842, epoch 5.99)이다. val overall은 epoch 1~6이 66.20 → 71.76 → 74.88 → 77.32 → 80.13 → 80.30으로 계속 올라 최고가 마지막 epoch였다. 노트북 설정 셀의 `FINAL_EPOCHS = 9`는 실제 실행과 맞지 않아 6으로 고쳤다. 공식 delta는 9 epoch 학습이므로 학습 길이가 같지 않다. 1단계 학습률 탐색(2 epoch, val 400문항 부분집합)은 2e-4가 74.59, 1e-4가 74.07, 2e-5가 66.65였다.

## 5. 정리: 시도한 것과 남은 것

| 모델 | 시도한 것 | 결과 | 남은 것 |
|---|---|---|---|
| M3AE | paper 프로토콜 학습, 마지막 vs val 최고 체크포인트 | 논문 ±0.74 이내(규칙 완화) | **clean 행 재학습**, 예측 파일 저장 여부 확인 |
| MUMC | val 선택 패치, `clean`/`trainval`/`paper` 프로토콜, SLAKE train만 | VQA-RAD −5.81 미달, SLAKE −0.45 이내 | `rad_common_paper` 채점, 통일 채점 재계산, SLAKE val 최고 epoch 확정 |
| LLaVA-Med VQA-RAD | 공식 delta, greedy, closed 접미어 | −5.50 (closed가 원인) | 부록 A 결과 정리, (선택) val 기반 closed 보정 |
| LLaVA-Med SLAKE | QLoRA 근사(+projector), val로 학습률·epoch 선택, yes/no 보정 | −4.57 미달, 보정 무효 | `FINAL_EPOCHS`·best epoch 확인. 재학습은 개선 보장이 없고 비용이 커서 권하지 않음 |

**공통으로 남은 것**: 시드 간 변동 미측정, 통일 채점으로 네 모델을 한 표에 정리, RAG · Expert-CFG를 다른 모델에 적용하는 것은 시작하지 않았다.

## 6. 미확인 항목

- M3AE: 예측 파일 저장 여부, GPU · 학습 시간, `max_epoch`.
- MUMC: 통일 채점 값(VQA-RAD 약 −2.8은 미재확인, 계산값 약 −4.0과 다름), `rad_common_paper` 점수, SLAKE ep22가 val 최고인지.
- LLaVA-Med: VQA-RAD 부록 A 결과, SLAKE best epoch · `FINAL_EPOCHS`, QLoRA 근사의 격차가 근사 방식 때문이라는 가설.
