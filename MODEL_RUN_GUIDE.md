# 담당 모델 3개 실행 가이드 (VQA-RAD / SLAKE 베이스라인)

대상: 이번 주 과제를 맡은 팀원. 각 저장소의 README와 설정 파일을 직접 읽어 정리했다.
**실제로 돌려 본 것은 하나도 없다.** "확인됨"은 문서·코드에서 읽은 내용, "미확인"은 문서에 없거나 실행해 봐야 아는 내용이다.

## 0. 목표 수치 (논문 Table 1, 각 모델 원 논문의 보고값을 인용한 것)

| 모델 | VQA-RAD (Open / Closed / Overall) | SLAKE (Open / Closed / Overall) |
|---|---|---|
| MUMC | 71.50 / 84.20 / 79.20 | – / – / 84.90 |
| M3AE | 67.23 / 83.46 / 77.01 | 80.31 / 87.82 / 83.14 |
| LLaVA-Med 7B | 61.52 / 84.19 / 75.19 | 83.08 / 85.34 / 83.97 |

- overall 은 open·closed 의 가중평균이다. VQA-RAD 는 open 179 / closed 272, SLAKE(영어) 는 open 645 / closed 416 으로 계산하면 위 표가 맞는다.
- 논문(Expert-CFG) §4.3 지표: closed = 정확도, open = 재현율(정답 토큰이 예측에 나타나는 비율).

## 1. 먼저 알아둘 점 (결과 해석에 영향)

1. **best epoch 를 테스트 성능으로 고르는 코드 구조.**
   - MUMC `vqaEvaluate.py` 는 모든 epoch 의 결과를 평가한 뒤 overall 로 정렬한다(`all_result_list.sort(key=lambda x: x['Overall'])`).
   - M3AE 는 `vqa/test/score_best_epoch`, `close_score_best_epoch`, `open_score_best_epoch` 처럼 "best" 값을 기록한다.
   - 즉 논문에 보고된 수치가 테스트 점수로 고른 epoch 의 값일 가능성이 있다(원 논문들의 선택 방식은 미확인). 재현할 때는 **마지막 epoch 점수와 best epoch 점수를 모두 기록**할 것.
2. **지표가 모델마다 다르다.** 공식 평가 코드의 open 점수가 논문 지표(재현율)와 같다는 보장이 없다. 비교용 표는 예측 파일을 모아 **우리 평가 함수(closed 정확도 / open 재현율)로 다시 채점**하는 것을 권한다.
3. **분할이 다를 수 있다.** SLAKE 는 영어만 쓰는지(M3AE 는 `q_lang == "en"` 만 사용), MUMC 의 SLAKE 는 train+val 로 학습하는지 등을 표에 적을 것.

## 2. MUMC (분류형)

- 저장소: https://github.com/pengfeiliHEU/MUMC (MICCAI 2023, ALBEF 기반)
- 환경: `pip install -r requirements.txt` (Python·CUDA 버전은 README 에 없음 — 미확인)
- 체크포인트: **사전학습 가중치만** 공개(Google Drive: https://drive.google.com/file/d/1ZxwjfDeBYTMpw4mN_R9-gAcR9UOJiAFb/view?usp=sharing → `pretrain/med_pretrain_29.pth`). 파인튜닝된 체크포인트는 README 에 없음 → **데이터셋별 파인튜닝이 필요**하다.
- 데이터 배치
  - VQA-RAD: `data_RAD/images/`, `trainset.json`, `testset.json`, `answer_list.json`
  - SLAKE: `data_Slake/imgs/`(`xmlab*/source.jpg.jpg`, `question.json`), `slake_train.json`, `slake_val.json`, `slake_test.json`, `answer_list.json`
  - 주의: `configs/VQA.yaml` 에는 RAD 답 목록이 `answer_all_list.json`, SLAKE 는 `answer_trainval_list.json` 으로 적혀 있어 README 와 파일 이름이 다르다. 실행 전에 맞출 것.
- 명령
  ```
  python3 train_vqa.py --dataset_use rad   --checkpoint ./pretrain/med_pretrain_29.pth --output_dir ./output/rad
  python3 train_vqa.py --dataset_use slake --checkpoint ./pretrain/med_pretrain_29.pth --output_dir ./output/slake
  ```
  (SLAKE 명령은 README 가 `rad / pathvqa / slake` 중 선택한다고 한 데서 유추한 것.)
- 설정(`configs/VQA.yaml`, 확인됨): 이미지 384, batch train 8 / test 4, `k_test` 128, lr 2e-5 → 1e-8, weight decay 0.05, **max_epoch 40**, alpha 0.4, distillation on.
  - SLAKE 설정은 `slake_train.json` + `slake_val.json` 으로 학습하고 `slake_test.json` 으로 테스트한다.
- 평가: `vqaEvaluate.py`(driver) → `vqaTools/vqaEval.py`. 정확 일치 여부와 closed 판정 방식은 `vqaEval.py` 를 직접 읽어야 한다(미확인).
- 시간: 미측정. 첫 epoch 시간을 재서 40 epoch 로 곱해 판단할 것.

## 3. M3AE (분류형)

- 저장소: https://github.com/zhjohnchan/M3AE (arXiv 2209.07098, ViLT·METER 기반)
- 환경: `pip install -r requirements.txt`
- 체크포인트: README 가 Google Drive 폴더(https://drive.google.com/drive/folders/1lS2fm8RqFxoKQ8cgYD9hMFoOuHuKZNoY)에 "pre-trained and fine-tuned models" 가 있다고 한다. 어떤 파일이 VQA-RAD / SLAKE 용 파인튜닝본인지는 폴더를 열어 확인해야 한다(미확인). 파인튜닝본이 있으면 `bash run_scripts/test_m3ae.sh` 로 평가만 할 수 있다(README 는 이것이 환경 점검에 좋다고 함).
- 데이터 배치 → 변환(`python prepro/prepro_finetuning_data.py`)
  - VQA-RAD: `data/finetune_data/vqa_rad/{trainset,valset,testset}.json` + `images/`. **`valset.json` 이 필요하다.** VQA-RAD 에는 공식 val 이 없으므로 이 파일을 어떻게 만드는지 확인해야 한다(미확인). `answer_type` 필드를 그대로 읽는다.
  - SLAKE(코드 이름은 `slack`): `data/finetune_data/slack/{train,validate,test}.json` + `imgs/`, 영어(`q_lang == "en"`)만 사용.
  - 출력: `data/finetune_arrows/vqa_vqa_rad_*.arrow`, `vqa_slack_*.arrow`
- 명령: `bash run_scripts/finetune_m3ae.sh`. 이 스크립트는 VQA-RAD, SLAKE(`task_finetune_vqa_slack`), MedVQA-2019, MELINDA, ROCO 검색까지 **여러 작업을 한 번에** 돌린다. 필요한 줄만 남기도록 복사해서 쓸 것.
  - 공통: `num_gpus=1`, `per_gpu_batchsize=16`, `load_path=downloaded/pretrained/m3ae.ckpt`, `clip16` + `text_roberta`, `image_size=384`(스크립트가 덮어씀. `config.py` 의 값은 576).
- 설정(`m3ae/config.py`, 확인됨)

  | | VQA-RAD | SLAKE |
  |---|---|---|
  | max_epoch | 50 | 15 |
  | batch_size | 64 | 32 |
  | learning_rate | 5e-6 | 5e-6 |
  | vqa_label_size | 498 | 222 |

- 평가: 점수 계산은 `gadgets/my_metrics.py` 의 `VQARADScore` 에 있다(이 파일은 읽지 못함, 미확인). 로그 키는 `vqa/test/score_epoch`, `close_score_best_epoch`, `open_score_best_epoch` 등.
- 시간: 미측정.

## 4. LLaVA-Med 7B (생성형)

- 저장소: https://github.com/microsoft/LLaVA-Med
- **버전을 구분해야 한다.**
  - **v1.0.0**(태그): LLaMA-7B + **delta 가중치** 방식. 다운스트림 파인튜닝본이 있다. → 논문 Table 1 의 7B 수치와 가장 가까운 쪽(그 수치가 어느 버전인지는 미확인).
  - **v1.5**(main): `microsoft/llava-med-v1.5-mistral-7b`(HF), delta 없음. 문서에는 VQA-RAD/SLAKE 파인튜닝 체크포인트가 없다.
- v1.0.0 파인튜닝된 delta(각 11.06GB, 이름에 `9epoch`): `data_RAD-9epoch_delta.zip`, `Slake1.0-9epoch_delta.zip`, `pvqa-9epoch_delta.zip`, 호스팅: `hanoverprod.z21.web.core.windows.net/med_llava/models/` (링크가 아직 살아 있는지는 미확인).
- 필요한 것: **LLaMA-7B 원본 가중치(HF 형식)**. LLaMA 라이선스를 지켜야 한다.
- 병합·평가 명령(v1.0.0)
  ```
  python3 -m llava.model.apply_delta --base /path/to/llama-7b --target ./llava-med-7b-vqarad --delta /path/to/data_RAD-9epoch_delta
  python3 -m llava.model.apply_delta --base /path/to/llama-7b --target ./llava-med-7b-slake  --delta /path/to/Slake1.0-9epoch_delta
  # 추론 → 채점
  llava/eval/model_vqa_med.py ... --question-file <test.json> --image-folder <images> --answers-file <test-answer-file.jsonl>
  llava/eval/run_eval.py --gt <test.json> --pred <test-answer-file.jsonl>
  ```
  테스트 JSON 은 대화 형식으로 변환된 `train.json / test.json / images` 가 필요하다. 이 데이터를 어디서 받는지는 가져온 문서에 없다(미확인, README 의 데이터 절 확인).
- 환경(v1.0.0, 확인됨): Python 3.10, `torch==2.0.0+cu117`, `torchvision==0.15.1+cu117`, `openai==0.27.8`, `transformers` 는 git 커밋 `cae78c46`, `flash-attn`(`--no-build-isolation`). **Phi 노트북 환경과 충돌하므로 별도 환경이 필요하다.**
- 평가 지표(`run_eval.py`, 확인됨)
  - 정답 유형은 정답 필드의 `answer_type`(OPEN/CLOSED)을 그대로 쓴다.
  - closed: 예측에 "yes"/"no" 가 포함되고 정답이 그 안에 들어 있으면 1(부분 문자열 일치, 정규화 후).
  - open: 재현율, F1, 정밀도, BLEU, exact match, **"hit"(정답이 예측에 나타나는 비율)** 을 모두 출력. 논문이 인용한 open 점수가 어느 것인지는 미확인(§4.3 은 재현율이라고 적었다).
  - 출력 줄 `num_open` / `num_close` 는 라벨이 서로 바뀌어 찍힌다(코드 버그, 값은 반대로 읽을 것).
- 직접 파인튜닝할 경우(참고): 전체 모델 튜닝, 8×A100 40GB 에서 약 1시간(README), `--num_train_epochs 3`, `--learning_rate 2e-5`, `--model_max_length 2048`, `--bf16 True`, `--fsdp "full_shard auto_wrap"`. 배포된 delta 이름의 `9epoch` 와 에폭 수가 다르다.

## 5. 우리 비교표를 위한 공통 규칙

- 모든 모델의 **예측을 파일로 저장**하고, 같은 채점 함수(closed 정확도 / open 재현율, 원본 `answer_type`)로 다시 계산한다.
- 기록할 것: 체크포인트 출처, 학습 분할(train 만인지 train+val 인지), 테스트 문항 수, 사용한 epoch(마지막인지 best 인지), 라이브러리 버전, GPU.
- SLAKE: 영어 부분만(테스트 약 1,061문항, open 645 / closed 416)인지 확인. MUMC 논문 수치는 overall 만 있다.

## 6. 문서로 확인하지 못한 항목 (실행 전 확인)

- 각 모델의 실제 학습·평가 시간.
- MUMC: Python/CUDA 버전, SLAKE 명령, `vqaTools/vqaEval.py` 의 채점 방식, 파인튜닝 체크포인트 유무(README 에는 없음).
- M3AE: Drive 폴더 안의 파인튜닝 체크포인트 목록, `valset.json` 생성 방법, `VQARADScore` 의 채점 방식.
- LLaVA-Med: delta 다운로드 링크의 유효성, 테스트 JSON 의 출처·형식, 논문 Table 1 의 7B 수치가 v1.0.0 의 파인튜닝본인지.

## 출처

- https://github.com/pengfeiliHEU/MUMC (README, `configs/VQA.yaml`, `vqaEvaluate.py`)
- https://github.com/zhjohnchan/M3AE (README, `run_scripts/finetune_m3ae.sh`, `m3ae/config.py`, `prepro/prepro_finetuning_data.py`, `m3ae/modules/m3ae_utils.py`)
- https://github.com/microsoft/LLaVA-Med (main README, v1.0.0 README, `llava/eval/run_eval.py`)
- Expert-CFG 논문 (arXiv 2507.09209) Table 1, §4.3

## 정정·추가 확인 사항 (저장소 코드를 직접 읽고 확인, 실행은 아직 안 함)
- **SLAKE 데이터/이미지**: HF `BoKelvin/SLAKE` 에 원본 `train.json / validation.json / test.json / imgs.zip` 이 들어 있다 (`datasets.load_dataset` 표 형식에는 이미지 컬럼이 없음). 영어 test 는 1,061문항 (OPEN 645 / CLOSED 416), 이미지는 `imgs/xmlabN/source.jpg`.
- **MUMC**: `train_vqa.py` 는 `--output_dir` 을 무시하고 `/mnt/sda/lpf/...` 로 덮어쓰며 `ruamel_yaml` 를 import 한다(패치 필요). `--evaluate` 는 평가 없이 종료하고, 평가는 매 epoch 학습 중 test 셋에서 수행되어 `result/*_vqa_result_<epoch>.json` 으로 저장된다(val 셋 없음 → best epoch 는 test 선택 값). 데이터 키는 `image_name, qid, question, answer, answer_type, question_type`.
- **M3AE**: prepro 는 상대 경로 `data/finetune_data/...` → `data/finetune_arrows/` 고정이고 `__main__` 이 다른 데이터셋까지 실행하므로 `prepro_vqa_vqa_rad()`, `prepro_vqa_slack()` 만 호출해야 한다. 학습은 `data_root=data/finetune_arrows/ image_size=384` (SLAKE 는 `clip_resizedcrop` 추가). 환경은 torch 1.9 / PL 1.3.2 / numpy 1.19 → Python 3.8 venv.
- **LLaVA-Med**: `model_vqa_med.py` 질문 파일은 JSON 배열(`id`, `image`, `conversations`), 출력은 jsonl(`question_id`, `text`). `apply_delta` 인자는 `--base-model-path --target-model-path --delta-path`. SLAKE/VQA-RAD delta 는 HF `katielink/llava-med-7b-{slake,vqarad}-delta` 에 존재(research license).
- 위 내용을 반영한 노트북: `MUMC_/M3AE_/LLaVA_Med_expert_cfg_reproduction.ipynb`.
