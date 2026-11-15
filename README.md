# Phi-3.5V-Med: VQA-RAD · SLAKE 베이스라인 + RAG · Expert-CFG 재현

논문 *Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models* (arXiv:2507.09209)의 **Phi-3.5V-Med** 행을 VQA-RAD와 SLAKE에서 재현한다. 이 브랜치에는 그 실험에 필요한 코드만 들어 있다. (PathVQA, Phi-3V, 다른 베이스라인 모델은 다루지 않는다.)

- 베이스라인: Phi-3.5V-Med LoRA를 train으로 이어서 학습하고, val로 체크포인트를 고른 뒤 test를 한 번만 평가한다 (`01`).
- RAG · Expert-CFG: 같은 어댑터에 검색된 캡션을 붙이고, 불확실성이 높은 5% 문항에만 하이라이트 + CFG를 적용한다 (`02`).

## 논문 Table 1과 실행 이름

논문 수치(Phi3.5V-Med, Open / Closed / Overall)는 arXiv HTML 본문에서 추출한 값이다. 베이스라인 82.04 / 85.20은 프로젝트 재현 목표와 일치해 신뢰하지만, 나머지 행은 원문과 대조하기 바란다.

| 논문 행 | 실행 이름 (`02`) | VQA-RAD | SLAKE |
|---|---|---|---|
| Phi3.5V-Med | `baseline` (`01`의 test 결과 파일) | 69.27 / 90.44 / **82.04** | 82.95 / 88.70 / **85.20** |
| w/ RAG (top-1, 전 문항) | `RAG-100` | 70.39 / 88.24 / **81.15** | 82.79 / 89.90 / **85.58** |
| w/ 5% Expert-RAG | `ExpertRAG-5` | 72.06 / 90.44 / **83.37** | 83.56 / 89.90 / **86.05** |
| w/ 5% Expert-CFG | `ExpertCFG-5` | 74.86 / 91.18 / **84.70** | 85.27 / 91.59 / **87.75** |

논문에 없는 우리 쪽 추가 행: `ExpertRAG-5-oracle`, `ExpertCFG-5-oracle` (open 하이라이트=정답), `ExpertRAG-5-inject`, `ExpertCFG-5-inject` (open 캡션에 정답을 덧붙임). 모두 **규칙 완화(oracle)** 행이다. 아래 "규칙과 채점" 참고.

## 파일 구성

```
01_Phi35V_baseline_VQARAD_SLAKE.ipynb   학습 · 체크포인트 선택(val) · 최종 test 평가 · 결과 정리 (DATASET 스위치로 두 데이터셋)
02_Expert_CFG_RAG_comparison.ipynb      RAG · Expert-RAG · Expert-CFG 평가와 논문 비교표 (DATASET 스위치로 두 데이터셋)
vqa_scoring.py                          채점: 원본(Phi 방식) / S1+S2 보조 채점 / 부트스트랩 신뢰구간
tools/build_rag_testjson.py             RAG 검색 파일(rag_*.json) 생성 (VQA-RAD · SLAKE에 같은 파이프라인)
src/highlighter_modules/guidance.py     CFG logits 처리 (논문 저자 코드)
src/highlighter_modules/utils.py        하이라이트 문구 → 토큰 마스크 (프로세서가 사용, 논문 저자 코드)
src/rag/biomedclip_rag.py               이미지→이미지 FAISS 검색 (이 브랜치의 노트북은 쓰지 않음, 참고용)
Phi-3.5-vision-instruct/                processing_phi3_v_cfg.py, processor_config.json (하이라이트 attention 마스크를 내는 프로세서)
data_common/vqarad_img/                 VQA-RAD 공통 분할 train 2,749 / val 315 / test 451 (이미지 단위, 시드 42)
data_common/make_common_split.py        위 분할을 만드는 스크립트
reports/Phi35V_results.md               베이스라인 결과 · 체크포인트 선택 · 주의사항
```

Phi-3.5 베이스 가중치, Med LoRA(`ecoxial2007/Phi-3.5V-Med`), BiomedCLIP, SLAKE(`BoKelvin/SLAKE`)는 노트북이 Hugging Face에서 자동으로 받는다. 코드와 데이터 분할만 이 브랜치에서 받으면 된다.

## 실행 (Colab, GPU)

두 노트북 모두 첫 셀이 이 브랜치를 얕게 clone한다 (`git clone --depth 1 -b handoff/phi35v-vqarad-slake-rag-cfg https://github.com/MinseongKim-dev/Expert-CFG.git`).

### 0. Drive 준비

| Drive 경로 | 내용 | 출처 |
|---|---|---|
| `MyDrive/vqa_rad/VQA_RAD Dataset Public.json`, `VQA_RAD Image Folder/` | VQA-RAD 원본 | [OSF](https://osf.io/89kps/) |
| `MyDrive/vqarad_common/{trainset,valset,testset}.json` | 공통 분할 (train 2,749 / val 315 / test 451, 이미지 단위) | 이 브랜치의 `data_common/vqarad_img/*.json`을 올린다. **Drive 폴더 이름은 `vqarad_common`이지만 내용은 `vqarad_img`다.** 이전 버전(질문 단위, 2,758 / 306 / 451)은 이 브랜치에 넣지 않았다 |
| `MyDrive/ExpertCFG_replication/PEIR_openai_imgid2caption.json`, `PEIR_openai_match_keyword.json` | PEIR 캡션 · 키워드 | 논문 저자 배포 파일 (README의 PEIR 링크) |
| `MyDrive/expert_cfg/` | 학습 · 평가 결과 (노트북이 만든다) | |

### 1. 베이스라인 — `01`

1. 설정 셀(5절)에서 `DATASET`과 `PHI_EPOCHS`를 정한다. 보고한 실행값은 **VQA-RAD 10 epoch, SLAKE 5 epoch**다 (노트북 기본값은 VQA-RAD 기준).
2. 첫 셀은 논문 환경에 가까운 패키지(torch 2.3.0, transformers 4.43.0, peft 0.12.0 등)를 설치한다. 설치 후 **런타임을 다시 시작**하고 그 셀을 한 번 더 실행해 버전을 확인한다.
3. 위에서 아래로 실행하면 `MyDrive/expert_cfg/logs_phi35_{dataset}_{tag}/checkpoint-*`와 체크포인트별 val 결과가 생긴다. val 최고 체크포인트를 `eval_checkpoints(..., run_test=True)`로 **test 한 번** 평가한다.
4. 마지막 셀(9절)이 최종 점수, 신뢰구간, 불확실성 AUC를 출력한다. `DS`, `RUN_TAG`는 5절의 값과 같게 둔다.

보고한 체크포인트: VQA-RAD `checkpoint-257` (epoch 6), SLAKE `checkpoint-380` (5 epoch). 결과 파일 이름은 `vqarad_continue_e10_env_common_ckpt257_test.json`, `slake_continue_e5_env_common_ckpt380_test.json`이다.

### 2. RAG 검색 파일 만들기 — `tools/build_rag_testjson.py`

`02`에서 쓰는 검색 결과(`rag_vqarad.json`, `rag_slake.json`)를 만든다. **`02`와 다른 런타임(새 Colab 세션)에서** 돌리는 것을 권한다. `open_clip_torch` 설치가 `02`의 transformers 5.x 패치 환경과 충돌할 수 있다.

```python
!pip -q install open_clip_torch pyarrow huggingface_hub
import subprocess
R = '/content/drive/MyDrive/ExpertCFG_replication'
subprocess.run(['git', 'clone', '--depth', '1', '-b', 'handoff/phi35v-vqarad-slake-rag-cfg', 'https://github.com/MinseongKim-dev/Expert-CFG.git', '/content/Expert-CFG'])
cmd = ['python', '-u', '/content/Expert-CFG/tools/build_rag_testjson.py', '--peir_dir', R,
       '--vqarad_common', '/content/drive/MyDrive/vqarad_common/testset.json',
       '--vqarad_train', '/content/drive/MyDrive/vqarad_common/trainset.json',
       '--vqarad_img', '/content/drive/MyDrive/vqa_rad/VQA_RAD Image Folder', '--out_dir', R,
       '--kb', 'peir']          # 'peir,roco,trainqa' 로 지식베이스를 넓힐 수 있다 (아래 참고)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in p.stdout: print(line, end='')
```

- PEIR 캡션 33,572개 인코딩은 처음 한 번만 걸린다 (L4에서 대략 수 분~십수 분, 결과는 `peir_text_feats_ours.npy`로 캐시).
- 출력 마지막 줄들의 **장기 일치율**(검색된 top-1 캡션이 이미지의 장기를 언급하는 비율, 무작위 대비)로 검색이 쓸 만한지 확인한다. 실행 때 VQA-RAD 96.5% (무작위 42.8%), SLAKE 71.1% (무작위 34.3%)였다. SLAKE 값은 장기 단어 목록이 불완전해 하한이다.

### 3. RAG · CFG 평가 — `02`

1. 설정 셀의 `DATASET`을 `'vqarad'`로 한 번, `'slake'`로 한 번, 처음부터 실행한다 (어댑터가 달라 모델을 다시 로드해야 한다).
2. `KB`는 기본 `'peir'`다. `RUN_RAG_ALL = False`면 전 문항 w/ RAG를 건너뛰고 5% 행만 돈다 (SLAKE는 전 문항 RAG가 1,061번 생성이라 오래 걸린다).
3. 결과는 `MyDrive/expert_cfg/ragcfg_{dataset}[_{kb}]/`에 문항 단위로 저장되어 끊겨도 이어서 실행된다. 맨 아래 "종합 비교" 셀이 두 데이터셋을 논문과 나란히 보여 준다 (GPU 불필요).

속도 참고 (L4): VQA-RAD 전 문항 w/ RAG 약 30분 (451문항, 문항당 약 4초). 이 노트북은 CFG 때문에 같은 입력을 2벌 넣는다.

## 결과 요약

### 베이스라인 (규칙 준수: train만 학습, val로 선택, test 1회)

| 데이터셋 | 채점 | open | closed | overall | 논문 overall 대비 |
|---|---|---|---|---|---|
| VQA-RAD (ckpt-257) | **S1+S2 (공식)** | 67.99 | 88.97 | **80.64** | −1.40 |
| | 원본 | 65.33 | 88.97 | 79.59 | −2.45 |
| SLAKE (ckpt-380) | 원본 = S1+S2 | 85.04 | 86.54 | **85.63** | +0.43 |

모두 1회 실행이다 (시드 간 변동 미측정). 자세한 내용은 `reports/Phi35V_results.md`.

### RAG · Expert-CFG (overall, S1+S2 / 베이스라인 대비 Δ)

검색 지식베이스(KB)가 다른 실행이 섞여 있으므로 열마다 KB를 적는다. **같은 KB끼리만 비교한다.**

| 행 | VQA-RAD, KB=`peir` | VQA-RAD, KB=`peir,roco,trainqa` | SLAKE, KB=`peir,roco,trainqa` | 논문 Δ (VQA-RAD / SLAKE) |
|---|---|---|---|---|
| 베이스라인 | 80.64 | 80.64 | 85.63 | |
| w/ RAG | 76.18 (**−4.46**, CI가 0 제외) | 75.81 (**−4.83**, CI가 0 제외) | 85.70 (+0.07) | −0.89 / +0.38 |
| 5% Expert-RAG | 81.23 (+0.59, CI가 0 포함) | 80.97 (+0.33)* | 85.46 (−0.17)* | +1.33 / +0.85 |
| 5% Expert-CFG | 80.79 (+0.15, CI가 0 포함) | 80.97 (+0.33)* | 85.45 (−0.18)* | +2.66 / +2.55 |
| 5% Expert-RAG (정답 주입, oracle) | 82.86 (+2.22) | 82.82 (+2.18) | 86.58 (+0.96) | |
| 5% Expert-CFG (정답 주입, oracle) | 82.56 (+1.92) | 83.15 (+2.51) | 86.65 (+1.02) | |

\* **이 행들은 Expert 효과를 측정한 값이 아니다.** 이 KB에서는 top-1 캡션의 대부분이 ROCO라서 PEIR 키워드 하이라이트가 쓰이지 않고, 대신 쓰는 단순 규칙(질문과 캡션이 공유하는 단어)은 거의 비어 있다. VQA-RAD(KB=`peir,roco,trainqa`)에서 선택 23문항을 측정했을 때 top-1 출처는 ROCO 22 / train QA 1 / PEIR 0, 마스크된 토큰은 평균 1.8개(0개인 문항 9/23)였고, 일반 Expert 행 네 개의 점수가 소수점 둘째 자리까지 같았다. 그래서 이 행들은 "캡션만 붙인 생성"과 같다. SLAKE는 마스크 토큰 수를 따로 측정하지 않았지만 Expert-RAG(85.46)와 Expert-CFG(85.45)가 거의 같아 같은 상황으로 보인다.

- **하이라이트가 실제로 작동한 일반 5% 행은 VQA-RAD(KB=`peir`) 하나뿐이다** (마스크 평균 46.4토큰). 그 행의 개선은 통계적으로 확인되지 않았다(CI가 0 포함).
- **SLAKE의 KB=`peir` 결과는 아직 없다.** SLAKE는 새 KB로만 돌렸다(`summary.json`에 KB가 기록된 값). 두 데이터셋을 같은 조건으로 비교하려면 SLAKE를 `KB='peir'`로 다시 돌려야 한다 (5% 행만이면 약 21분, L4 기준).
- 규칙 준수로 보고할 수 있는 행은 베이스라인, w/ RAG, 일반 5% 행이다.
- **정답 주입 행에서만 논문 규모의 개선이 나온다.** 논문은 open 문항에서 정답을 전문가 어노테이션으로 쓰고 전문가가 참조 텍스트를 수정한다고 설명한다. 그 해석과 일치하지만, 주입 형식(`Expert note: {정답}.`)은 우리가 정한 것이라 논문과 같다고 확인하지는 못했다. 정답 주입 행은 KB와 무관하게 비슷한 값이 나왔다.
- KB를 넓혀도 w/ RAG는 개선되지 않았다(VQA-RAD −4.46 → −4.83, 오차 범위). KB 구성이 원인이라는 가설은 이 실험으로는 뒷받침되지 않는다.

## 구현 대응

| 논문 요소 | 위치 |
|---|---|
| BioMedCLIP 이미지 쿼리 × 지식베이스 캡션 검색 (코사인 top-k) | `tools/build_rag_testjson.py`: `Encoder`, `KB`, `rows_with_rag` |
| 하이라이트 키워드 | 같은 파일의 `peir_highlight_index` (PEIR 키워드), `qmatch_highlights` (ROCO · train QA 항목용 규칙) |
| 예측 엔트로피로 상위 5% 선택 | `02` 데이터 셀의 `ent`, `sel`. 엔트로피는 `01` 평가가 저장한 문항별 평균 토큰 엔트로피 |
| 하이라이트 토큰의 attention 가중 | `02`의 `build_expert_cfg_inputs`, `_bias_pre_hook`, `install_attention_bias_hooks` |
| CFG 생성 `guided = γ(cond − uncond) + uncond` (log-prob) | `02`의 `generate_expert_cfg` (`src/highlighter_modules/guidance.py`의 `ProbCFGLogitsProcessor`와 같은 식) |
| 하이라이트 문구 → 토큰 마스크 | `src/highlighter_modules/utils.py`: `txt_highlight_mask` (프로세서가 호출) |
| 채점 · 신뢰구간 | `vqa_scoring.py`: `score_phi`, `score_s1s2`, `summarize`, `bootstrap_ci`, `print_ci` |

FAISS: 논문은 FAISS로 검색한다고 했다. `build_rag_testjson.py`는 정규화한 내적을 전부 계산해 정렬한다. 이는 `IndexFlatIP`와 같은 정확 검색이라 결과가 같고, 지식베이스가 약 13만 개 이하라 충분히 빠르다. 근사 검색을 쓰지 않았다.

## 논문 저자 데모와 의도적으로 다른 점

- **attention 가중을 4D additive mask에 `log(weight)`로 더한다.** 이 모델 코드는 2D attention mask에서 1이 아닌 값을 전부 "가림"으로 처리해서, 데모처럼 (attn=3, perturb=0.01)을 그대로 넣으면 가중이 아니라 해당 행 전체가 가려진다. 노트북의 `check_mask_effect`가 이를 점검한다.
- **`generate()` 대신 forward만 쓰는 greedy 루프**를 쓴다 (transformers 5.x + 구버전 remote code에서 `generate` 내부의 position_ids / cache 처리가 어긋난다).
- **환경이 두 가지다.** `01`은 논문 환경에 가깝게 고정(torch 2.3.0 / transformers 4.43.0)하고, `02`는 Colab 기본 환경(transformers 5.x)에서 호환 패치(`_attn_implementation`, `DynamicCache`)를 적용한다. 어댑터는 `01`의 고정 환경에서 학습한 것을 `02`가 불러 쓴다.
- Expert-RAG(캡션 + 하이라이트 attention, CFG 없음)의 조건부 가지 배율(`cond_boost`)은 논문 코드에서 확인된 정의가 아니라 우리 구현이다.

## 논문과 다를 수 있는 점 (결과 해석 시 반드시 고려)

- **검색 지식베이스:** 논문은 ROCO 70,306 + PEIR 33,572 + 세 데이터셋 train QA(캡션으로 변환)를 쓴다. 기본 설정은 PEIR만이다. `--kb roco,trainqa`는 공개 ROCO(train 65,423)와 VQA-RAD · SLAKE train QA를 "질문 정답" 텍스트로 이어 붙인 변형이며, PathVQA는 없고 train QA → 캡션 변환 방식은 논문이 밝히지 않아 우리가 정했다. 검색은 이미지 쿼리 × 텍스트 KB이고, 논문의 쿼리 구성은 본문에서 확인하지 못했다.
- **하이라이트:** 논문은 GPT-4로 키워드를 뽑고 전문가가 수정하며, open 문항은 정답을 그대로 쓴다. 우리 기본 행은 정답을 쓰지 않는다 (PEIR `match_keyword`).
- **w/ RAG:** top-1 캡션을 전 문항에 붙인다 (논문 Table 1과 같은 정의).
- **γ:** 1.3 (논문 Table 7에서 γ ∈ {1, 1.3, 1.5} 중 최고, 본문 기본은 1.5).
- **VQA-RAD 분할:** 공통 분할(train 2,749 / val 315 / test 451)이다. val 점수는 test보다 훨씬 낮아(test 이미지가 train과 많이 겹침) 체크포인트 간 상대 비교에만 쓴다.
- 5% 게이팅은 VQA-RAD 23문항, SLAKE 53문항만 바꾼다. 개선폭은 부트스트랩 CI와 함께 읽는다.

## 규칙과 채점

- **프로토콜:** train으로만 학습, val로 모델 · 체크포인트 선택, test는 최종에 한 번. val을 학습에 쓰거나 test로 설정을 고른 행은 별도 행으로 표기한다.
- **공식 채점은 S1+S2** (open만: 대소문자 · 하이픈 · 복수형 · 같은 용어의 약어 허용, 일반어 plane / weighted / mri 생략 허용. closed는 그대로). **이 규칙은 Phi의 VQA-RAD test 오답을 보고 만든 것이라 규칙 준수 결과가 아니다.** 그래서 항상 원본 채점(Phi 방식: closed는 정답 단어가 예측에 포함되면 정답, open은 단어 recall)을 같이 표기하고, 보고서에 이 사실을 적는다. SLAKE에서는 두 채점이 같다.
- **oracle 행(정답 하이라이트, 정답 주입)은 규칙 완화**다. 정답 정보가 들어가므로 규칙 준수 결과로 보고하지 않는다.

## 확인되지 않은 것

- 시드 간 변동: 모두 1회 실행이다.
- SLAKE를 `KB='peir'`로 돌린 결과 (현재 SLAKE RAG · CFG 열은 `peir,roco,trainqa` 한 가지뿐).
- 논문 Table 1의 RAG · Expert 행 수치 (HTML 추출값, 원문 대조 필요).
- w/ RAG가 VQA-RAD에서만 크게 나빠지는 이유 (어댑터가 캡션 없는 프롬프트로 학습된 영향일 수 있다는 가설은 검증하지 않았다).
- 자동 테스트는 없다. 동작 점검은 노트북 안의 `check_mask_effect`와 Colab 실행으로 했다.

## 출처

논문과 원 코드: Liang et al., *Uncertainty-Driven Expert Control: Enhancing the Reliability of Medical Vision-Language Models*, arXiv:2507.09209. `src/highlighter_modules/`와 `Phi-3.5-vision-instruct/processing_phi3_v_cfg.py`는 논문 저자가 공개한 코드를 바탕으로 하며, 이 프로젝트에서 수정한 부분이 있다 (수정 이력은 `main`, `VQA_RAD_reproduction` 브랜치 참고).
