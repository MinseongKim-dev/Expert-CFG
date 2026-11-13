"""공통 VQA-RAD 분할 생성 (이미지 단위): train(frame 포함 3,064 기본) 에서 val 을 떼어내고 test 는 그대로 쓴다.
 - val 은 **이미지 ID 단위**로 뽑는다. 한 이미지에 달린 질문은 모두 같은 쪽(train 또는 val)에 들어간다.
   (질문 단위로 나누면 같은 X-ray 가 train 과 val 에 모두 들어가 모델이 이미지를 외운 효과로 val 점수가 부풀 수 있다)
 - 시드 42 고정(이미지 목록을 정렬한 뒤 셔플), val 목표 약 10% 문항. 같은 입력이면 항상 같은 분할이 나온다.
 - test 는 공개 JSON 의 451문항 그대로(원본 분할). 이미지 단위 분리는 train/val 에만 적용한다.
사용: python make_common_split.py <train+val 합친 json(3,064)> <testset.json> <출력 폴더>
"""
import json, os, random, re, sys
from collections import Counter, defaultdict

train_p, test_p, out_dir = sys.argv[1:4]
SEED, VAL_FRAC = 42, 0.10
EXPECT_TRAIN = int(os.environ.get("EXPECT_TRAIN", "3064"))   # frame 포함 3,064 (noframed 는 1797)
norm = lambda a: re.sub(r'\s+', ' ', re.sub(r'[^a-z0-9\s]', ' ', str(a).lower())).strip()

train = json.load(open(train_p, encoding='utf8'))
test = json.load(open(test_p, encoding='utf8'))
for r in train + test:
    r['answer'] = str(r['answer'])
    r['answer_type'] = r['answer_type'].strip().upper()   # 원본에 'CLOSED ' 처럼 공백이 붙은 값이 있다

groups = defaultdict(list)
for r in train:
    groups[r['image_name']].append(r)      # 이미지 ID 단위 묶음
keys = sorted(groups)
random.Random(SEED).shuffle(keys)

val, target = [], round(len(train) * VAL_FRAC)
chosen = set()
for k in keys:
    if len(val) >= target:
        break
    val += groups[k]; chosen.add(k)
val_q = {r['qid'] for r in val}
tr = [r for r in train if r['qid'] not in val_q]

# ── 검증 ──
qs = lambda d: {r['qid'] for r in d}
assert len(tr) + len(val) == len(train) == EXPECT_TRAIN, (len(tr), len(val), len(train))
assert not (qs(tr) & qs(val)) and not (qs(tr) & qs(test)) and not (qs(val) & qs(test))
assert not ({r['image_name'] for r in tr} & {r['image_name'] for r in val}), '같은 이미지가 train 과 val 에 걸쳐 있습니다'
assert (sum(r['answer_type'] == 'CLOSED' for r in test), sum(r['answer_type'] == 'OPEN' for r in test)) == (272, 179)
pair = lambda r: (r['image_name'], norm(r['question']))
os.makedirs(out_dir, exist_ok=True)
for name, d in (('trainset.json', tr), ('valset.json', val), ('testset.json', test)):
    json.dump(d, open(os.path.join(out_dir, name), 'w', encoding='utf8'), ensure_ascii=False)
    c = Counter(r['answer_type'] for r in d)
    print(f'{name:12s} n={len(d):4d}  CLOSED {c["CLOSED"]:4d} / OPEN {c["OPEN"]:4d}  이미지 {len({r["image_name"] for r in d})}개')
print('train∩test (이미지,질문) 동일쌍:', len({pair(r) for r in tr} & {pair(r) for r in test}))
print('val∩test   (이미지,질문) 동일쌍:', len({pair(r) for r in val} & {pair(r) for r in test}))
print('val 이미지 중 train 에도 있는 이미지(0 이어야 함):', len({r['image_name'] for r in val} & {r['image_name'] for r in tr}), '/', len({r['image_name'] for r in val}))
print('test 이미지 중 train 에도 있는 이미지:', len({r['image_name'] for r in test} & {r['image_name'] for r in tr}), '/', len({r['image_name'] for r in test}))
print('val 정답이 train 정답에 있는 비율(open):',
      '%.1f%%' % (100 * sum(norm(r['answer']) in {norm(x['answer']) for x in tr if x['answer_type'] == 'OPEN'} for r in val if r['answer_type'] == 'OPEN') / max(1, sum(r['answer_type'] == 'OPEN' for r in val))))
print('test 정답이 train 정답에 있는 비율(open):',
      '%.1f%%' % (100 * sum(norm(r['answer']) in {norm(x['answer']) for x in tr if x['answer_type'] == 'OPEN'} for r in test if r['answer_type'] == 'OPEN') / 179))
print('검증 OK  (시드', SEED, ', val 목표', target, ')')
