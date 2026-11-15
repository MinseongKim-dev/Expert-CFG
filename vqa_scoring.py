"""VQA 채점 정리: Phi-3.5 노트북 방식('phi')과 MUMC 레포 방식('native')을 한 파일로 비교한다.

입력 레코드: {'answer': 정답, 'prediction': 예측, 'answer_type': 'OPEN' | 'CLOSED'}
실행하면 맨 아래 자체 검사(python vqa_scoring.py)가 돈다.
"""
import re
from collections import Counter


# ───────────────────────── Phi-3.5 노트북 방식 (VQA_RAD_expert_cfg_reproduction.ipynb 와 동일) ─────────────────────────
def normalize(text):
    """소문자 → <|end|> 제거 → 영문/숫자/공백이 아닌 문자는 공백 → 단어 리스트."""
    text = str(text).lower().replace('<|end|>', ' ')
    return re.sub(r'[^a-z0-9\s]', ' ', text).split()


def closed_accuracy(pred, gt):
    """정답 문자열이 예측의 '단어 리스트 안에' 있으면 1, 아니면 0.
    주의: 한 단어 정답(yes/no 등)에만 의도대로 동작한다. 'right lung' 같은 여러 단어 정답이나
    'posterior-anterior' 처럼 구두점이 든 정답은 리스트 원소와 같을 수 없어 항상 0이다."""
    return float(str(gt).strip().lower() in normalize(pred))


def open_recall(pred, gt):
    """정답 단어 중 예측에 들어 있는 비율(0~1). 예측이 길어도 감점 없음."""
    gt_toks, pred_toks = Counter(normalize(gt)), Counter(normalize(pred))
    if not gt_toks:
        return 0.0
    return sum((gt_toks & pred_toks).values()) / sum(gt_toks.values())


def score_phi(rec):
    f = closed_accuracy if rec['answer_type'] == 'CLOSED' else open_recall
    return f(rec['prediction'], rec['answer'])


# ───────────────────────── MUMC 레포 방식 (vqaTools/vqaEval.py + dataset/utils.pre_answer) ─────────────────────────
def pre_answer(answer):
    """MUMC dataset/utils.py 의 pre_answer 와 동일: 소문자, 일부 구두점 제거, x-ray 표기 통일."""
    answer = str(answer)
    answer = re.sub(r"([,.'!?\"()*#:;~])", '', answer.lower()).replace(' \t', ' ')
    answer = answer.replace('x ray', 'xray').replace('x-ray', 'xray')
    return answer.replace(' - ', '-')


def score_native(rec):
    """정규화한 정답과 예측이 완전히 같을 때만 1 (open/closed 구분 없음)."""
    return float(pre_answer(rec['answer']) == pre_answer(rec['prediction']))


# ───────────────────────── 집계 ─────────────────────────
def summarize(records, scorer=score_phi):
    """open / closed / overall (모두 0~100). overall 은 문항 단위 평균."""
    sc = [(r['answer_type'], scorer(r)) for r in records]
    op = [s for t, s in sc if t == 'OPEN']
    cl = [s for t, s in sc if t == 'CLOSED']
    mean = lambda xs: 100 * sum(xs) / len(xs) if xs else float('nan')
    return {'open': mean(op), 'closed': mean(cl), 'overall': mean(op + cl), 'n_open': len(op), 'n_closed': len(cl)}


def compare(records):
    """같은 예측을 두 방식으로 채점해 나란히 보여준다."""
    out = {'phi': summarize(records, score_phi), 'native': summarize(records, score_native)}
    print(f"{'scorer':8s} {'open':>7s} {'closed':>7s} {'overall':>8s}")
    for k, m in out.items():
        print(f"{k:8s} {m['open']:7.2f} {m['closed']:7.2f} {m['overall']:8.2f}")
    return out


# ───────────────────────── 공통 평가 프로토콜 (모든 모델에 동일하게 적용) ─────────────────────────
# 지침: closed 는 같은 정규화 + accuracy. open 은 normalized exact match + token-level F1.
#       recall 은 따로 보고할 수 있지만 accuracy 와 직접 비교하지 않는다.
#   - 위의 'phi'(closed 포함 판정, open recall)와 'native'(레포) 는 비교용으로만 남겨 둔다. 보고는 아래 unified 를 쓴다.
_ARTICLES = re.compile(r'\b(a|an|the)\b')


def norm_text(text):
    """공통 정규화: 소문자, 특수 토큰 제거, x-ray 표기 통일, 구두점(하이픈 포함)→공백, 관사(a/an/the) 제거, 공백 정리."""
    s = str(text).lower()
    for tok in ('<|end|>', '###', 'assistant:'):
        s = s.replace(tok, ' ')
    s = s.replace('x-ray', 'xray').replace('x ray', 'xray')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = _ARTICLES.sub(' ', s)
    return ' '.join(s.split())


def closed_correct(pred, gt):
    """closed accuracy (0/1). 정규화한 예측이 정답과 같으면 맞음.
    정답이 yes/no 이면 생성 모델의 문장형 답('Yes, there is a mass')을 위해 **예측의 첫 단어**가 정답이면 맞음으로 본다
    (정답 단어가 문장 어딘가에 있기만 하면 맞던 기존 phi 방식과 달리, 첫 단어만 본다). 모든 모델에 같은 규칙."""
    p, g = norm_text(pred), norm_text(gt)
    if p == g:
        return 1.0
    if g in ('yes', 'no'):
        toks = p.split()
        return float(bool(toks) and toks[0] == g)
    return 0.0


def open_em(pred, gt):
    """open normalized exact match (0/1)."""
    return float(norm_text(pred) == norm_text(gt))


def open_f1(pred, gt):
    """open token-level F1 (SQuAD 방식, 단어 중복은 개수까지 반영)."""
    p, g = norm_text(pred).split(), norm_text(gt).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def open_recall_u(pred, gt):
    """open token recall — 별도 참고 지표. accuracy/EM 과 직접 비교하지 말 것."""
    g = Counter(norm_text(gt).split())
    if not g:
        return 0.0
    return sum((g & Counter(norm_text(pred).split())).values()) / sum(g.values())


def summarize_unified(records):
    """공통 프로토콜 집계. overall 은 accuracy 계열(closed accuracy + open EM)만 문항 수 가중평균하며 F1/recall 은 섞지 않는다."""
    cl = [closed_correct(r['prediction'], r['answer']) for r in records if r['answer_type'] == 'CLOSED']
    op = [r for r in records if r['answer_type'] == 'OPEN']
    em = [open_em(r['prediction'], r['answer']) for r in op]
    f1 = [open_f1(r['prediction'], r['answer']) for r in op]
    rc = [open_recall_u(r['prediction'], r['answer']) for r in op]
    mean = lambda xs: 100 * sum(xs) / len(xs) if xs else float('nan')
    return {'closed_acc': mean(cl), 'open_em': mean(em), 'open_f1': mean(f1), 'overall_acc': mean(cl + em),
            'open_recall(ref)': mean(rc), 'n_closed': len(cl), 'n_open': len(op)}


def report(models):
    """models: {모델이름: records}. 한 표로 출력하고 결과 딕셔너리를 돌려준다."""
    out = {k: summarize_unified(v) for k, v in models.items()}
    print(f"{'model':14s} {'closed acc':>10s} {'open EM':>8s} {'open F1':>8s} {'overall acc':>11s} | {'open recall(ref)':>16s}")
    for k, m in out.items():
        print(f"{k:14s} {m['closed_acc']:10.2f} {m['open_em']:8.2f} {m['open_f1']:8.2f} {m['overall_acc']:11.2f} | {m['open_recall(ref)']:16.2f}")
    return out


# ── 예측 파일 → 공통 레코드 변환 ──
def records_from(gts, preds):
    """gts: {qid: {'answer', 'answer_type'}}, preds: {qid: 예측 문자열} → [{'answer','prediction','answer_type'}]"""
    assert set(gts) == set(preds), f'예측 {len(preds)}개 / 정답 {len(gts)}개 — qid 가 맞지 않습니다'
    return [{'qid': q, 'answer': str(gts[q]['answer']), 'prediction': str(preds[q]),
             'answer_type': str(gts[q]['answer_type']).upper()} for q in gts]


def load_mumc(test_json, result_json):
    """MUMC: testset.json(qid/answer/answer_type) + result/*_vqa_result_<epoch>.json([{qid, answer}])"""
    import json
    gts = {int(r['qid']): r for r in json.load(open(test_json))}
    preds = {int(r['qid']): r['answer'] for r in json.load(open(result_json))}
    return records_from(gts, preds)


def load_llava(ref_json, pred_jsonl):
    """LLaVA-Med: *_ref.json({id: {answer, answer_type}}) + 예측 jsonl(question_id, text)"""
    import json
    gts = {int(k): v for k, v in json.load(open(ref_json)).items()}
    preds = {}
    for line in open(pred_jsonl):
        p = json.loads(line)
        preds[int(p['question_id'])] = p['text']
    return records_from(gts, preds)


def load_records_json(path):
    """Phi 노트북처럼 이미 [{'answer','answer_type','prediction',...}] 형태로 저장된 파일"""
    import json
    return [{'answer': str(r['answer']), 'prediction': str(r['prediction']), 'answer_type': str(r['answer_type']).upper()}
            for r in json.load(open(path))]


# ───────────────────────── 참고 채점 S1+S2 (규칙 완화 · 공식 결과가 아님) ─────────────────────────
# 지위: 공식 채점은 원본('phi')이다. S1+S2 는 Phi 의 VQA-RAD test 오답에서 도출한 규칙이라 교수님 규칙(test 는 마지막에 한 번) 기준으로
#       규칙 완화이며, "표기 차이가 논문과의 격차에서 차지하는 몫"을 보는 참고 열로만 쓴다. 쓸 때는 모든 모델에 같은 함수를 적용한다.
# 구성: S1 = 표기 정규화(대소문자·하이픈·단수복수·약어 3개), S2 = 일반어 3개 생략. score_s1 / score_s1s2 로 따로 잴 수 있다.
# 기반은 'phi' 방식(open = 토큰 recall, closed = 정답 단어가 예측에 포함)이고, open 에서만 표기 차이를 허용한다. closed 는 건드리지 않는다.
#  S1 (표기 차이): 대소문자, 하이픈·공백(x-ray = xray), 단수·복수(sinuses = sinus), 같은 용어의 약어(posterior anterior = pa, x ray = xray, diffusion weighted = dwi)
#  S2 (일반어 생략): 단독으로는 정보를 더하지 않는 plane · weighted · mri 를 뺀 형태끼리 비교 (Axial plane = axial, T2-weighted = T2)
#  불허: 정보를 더하거나 빼는 것, 서로 다른 용어를 같은 뜻으로 보는 것 (plain film ↔ Xray, brain ↔ CNS, Right lung ↔ Right, CT with contrast ↔ CT, T2-MRI ↔ MRI)
#  점수 = max(원본 recall, S1 recall, S2 recall) → 원본보다 낮아지지 않는다.
# 주의: 이 규칙은 Phi-3.5 의 VQA-RAD test 오답 70개를 보고 만들었다. 규칙 준수(test 를 마지막에 한 번) 결과가 아니므로 항상 원본 채점 행과 함께 적는다.
S1S2_ABBR = [('x ray', 'xray'), ('posterior anterior', 'pa'), ('diffusion weighted', 'dwi')]
S1S2_REDUNDANT = {'plane', 'weighted', 'mri'}


def _s1_singular(w):
    if len(w) > 3 and w.endswith('uses'):
        return w[:-2]                                  # sinuses -> sinus
    if len(w) > 3 and w.endswith('s') and not w.endswith(('ss', 'us', 'is')):
        return w[:-1]
    return w


def _s1_tokens(s):
    t = re.sub(r'[^a-z0-9\s]', ' ', str(s).lower().replace("'", '').replace('-', ' '))
    t = ' '.join(t.split())
    for k, v in S1S2_ABBR:
        t = t.replace(k, v)
    out = []
    for w in map(_s1_singular, t.split()):
        if not out or out[-1] != w:
            out.append(w)                              # 약어 치환으로 생긴 연속 중복(dwi dwi)만 합친다
    return out


def _s1_recall(pred, gt, drop):
    f = (lambda L: [w for w in L if w not in S1S2_REDUNDANT] or L) if drop else (lambda L: L)
    g, p = Counter(f(_s1_tokens(gt))), Counter(f(_s1_tokens(pred)))
    return sum((g & p).values()) / sum(g.values()) if g else 0.0


S1S2_SPEC = {
    '지위': '참고(규칙 완화). 공식 채점은 원본(phi)',
    '도출 출처': 'Phi-3.5V 의 VQA-RAD test 오답 70개 (test 를 보고 만든 규칙)',
    'S1': {'정규화': '소문자, 하이픈·아포스트로피 제거 후 공백 정리, 단수·복수 통일(sinuses→sinus 등)',
           '약어 치환': dict(S1S2_ABBR)},
    'S2': {'생략 단어': sorted(S1S2_REDUNDANT)},
    '불허': 'plain film↔Xray, brain↔CNS, Right lung↔Right, CT with contrast↔CT, T2-MRI↔MRI 처럼 정보를 더하거나 빼거나 다른 용어를 같게 보는 것',
    '점수': 'open = max(원본 recall, S1 recall, S1+S2 recall) → 원본보다 낮아지지 않음. closed 는 원본 그대로',
    '관찰': 'SLAKE 에서는 점수가 바뀌지 않음(규칙이 VQA-RAD test 오답에 맞춰짐). 표준 정규화(llava·std)만으로는 거의 안 오름',
}


def score_s1(rec):
    """S1 만 (일반어 생략 없음): open = max(원본 recall, S1 recall). closed 는 원본."""
    base = score_phi(rec)
    if rec['answer_type'] != 'OPEN':
        return base
    return max(base, _s1_recall(rec['prediction'], rec['answer'], False))


def score_s1s2(rec):
    """S1+S2 보조 점수. closed 는 'phi' 점수 그대로, open 은 max(원본 recall, S1, S1+S2)."""
    base = score_phi(rec)
    if rec['answer_type'] != 'OPEN':
        return base
    return max(base, _s1_recall(rec['prediction'], rec['answer'], False), _s1_recall(rec['prediction'], rec['answer'], True))


# ───────────────────────── (1) 논문이 따랐다는 LLaVA-Med v1.0 평가 코드의 이식: 'llava' ─────────────────────────
# 논문 4.3절: "LLaVA-Med 를 따라 closed 는 accuracy, open 은 recall". 출처: microsoft/LLaVA-Med v1.0.0 의
#   llava/eval/run_eval.py (closed 규칙), eval_metrics/glossary.py::normalize_word, eval_metrics/evaluate_metrics.py::calculate_f1score (recall).
# 우리 'phi' 채점과의 차이: ① 정규화가 다르다(관사 a/an/the 제거, 숫자 단어→숫자, 소수점·천 단위 쉼표 유지, 하이픈 등은 공백/삭제)
#   ② closed: 예측에 'yes' 나 'no' 가 부분문자열로 있어야 하고, 정답이 정규화된 예측 문자열의 '부분문자열'이면 맞음 (단어 목록 원소가 아님)
#   ③ open recall: 맞힌 단어 수를 '예측 쪽 중복 횟수'로 센다 (정답 단어가 예측에 여러 번 나오면 부풀 수 있음)
# 이식하지 않은 것: contractions(dont→don't 등 약 120개 사전). 의료 답변에는 거의 나오지 않아 영향이 없다고 보지만 직접 확인하지 않았다.
_LM_PUNCT = [';', '/', '[', ']', '"', '{', '}', '(', ')', '=', '+', '\\', '_', '-', '>', '<', '@', '`', ',', '?', '!']
_LM_PERIOD = re.compile(r'(?!<=\d)(\.)(?!\d)')
_LM_COMMA = re.compile(r'(\d)(\,)(\d)')
_LM_NUM = {'none': '0', 'zero': '0', 'one': '1', 'two': '2', 'three': '3', 'four': '4', 'five': '5',
           'six': '6', 'seven': '7', 'eight': '8', 'nine': '9', 'ten': '10'}
_LM_ARTICLES = {'a', 'an', 'the'}


def llava_normalize(text):
    token = str(text).lower()
    _t = token
    for p in _LM_PUNCT:
        if (p + ' ' in token or ' ' + p in token) or (re.search(_LM_COMMA, token) is not None):
            _t = _t.replace(p, '')
        else:
            _t = _t.replace(p, ' ')
    token = _LM_PERIOD.sub('', _t)
    words = [_LM_NUM.get(w, w) for w in token.split()]
    return ' '.join(w for w in words if w not in _LM_ARTICLES).replace(',', '')


def _llava_recall_from_tokens(cand, ref):
    c, r = Counter(cand), Counter(ref)
    if not c or not r:
        return 0.0
    tp = sum(c[w] for w in c if w in r)                # 원본과 같이 예측 쪽 횟수로 센다
    fn = sum(r[w] for w in r if w not in c)
    return tp / (tp + fn) if tp else 0.0


def llava_closed(pred, gt):
    p, g = llava_normalize(pred), llava_normalize(gt)
    return float(('yes' in p or 'no' in p) and g in p)


def llava_open_recall(pred, gt):
    return _llava_recall_from_tokens(llava_normalize(pred).split(), llava_normalize(gt).split())


def score_llava(rec):
    """LLaVA-Med v1.0 평가 방식(논문이 따랐다고 적은 방식). 모든 모델에 같은 함수를 쓴다."""
    return llava_closed(rec['prediction'], rec['answer']) if rec['answer_type'] != 'OPEN' else llava_open_recall(rec['prediction'], rec['answer'])


# ───────────────────────── (2) 사전 정의한 표준 정규화 'std' (test 를 보지 않고 정한 일반 규칙) ─────────────────────────
# 'llava' 에 단수·복수 통일(_s1_singular)만 더한다. 약어 사전(x ray, pa, dwi)과 일반어 생략(plane/weighted/mri)은 넣지 않는다 —
# 그 둘은 Phi 의 VQA-RAD test 오답에서 나온 것이라 사전 정의가 아니다. closed 는 'llava' 규칙 그대로.
def std_tokens(text):
    return [_s1_singular(w) for w in llava_normalize(text).split()]


def score_std(rec):
    if rec['answer_type'] != 'OPEN':
        return llava_closed(rec['prediction'], rec['answer'])
    return _llava_recall_from_tokens(std_tokens(rec['prediction']), std_tokens(rec['answer']))


# ───────────────────────── (3) val 에서만 도출하는 일반어 생략 규칙 ─────────────────────────
# 절차: ① val 예측으로 mine_generic_words 를 돌려 후보를 본다 → ② 사람이 검토해 파일(JSON 리스트)로 동결한다
#       → ③ 동결한 뒤에만 make_val_rule_scorer 로 test 를 한 번 채점한다. test 오답은 규칙 도출에 쓰지 않는다.
def mine_generic_words(records, min_count=2):
    """val 의 open 오답 중 '정답에서 단 한 단어만 빠져서' 틀린 경우를 모아, 그 단어를 빈도순으로 돌려준다.
    반환: [(단어, 그 단어 하나 때문에 틀린 문항 수, 정답에 그 단어가 들어 있는 open 문항 수)]. 자동으로 규칙이 되지 않으며 사람이 검토해야 한다."""
    miss, appear = Counter(), Counter()
    for r in records:
        if r['answer_type'] != 'OPEN':
            continue
        g, p = std_tokens(r['answer']), std_tokens(r['prediction'])
        for w in set(g):
            appear[w] += 1
        gone = [w for w in set(g) if w not in set(p)]
        if len(gone) == 1 and len(set(g)) > 1 and p:
            miss[gone[0]] += 1
    return sorted(((w, n, appear[w]) for w, n in miss.items() if n >= min_count), key=lambda x: -x[1])


def make_val_rule_scorer(drop_words):
    """동결한 일반어 목록을 양쪽에서 뺀 recall 과 'std' 점수 중 큰 값 (open 만). 원본보다 낮아지지 않는다."""
    drop = set(drop_words)
    def scorer(rec):
        base = score_std(rec)
        if rec['answer_type'] != 'OPEN':
            return base
        f = lambda L: [w for w in L if w not in drop] or L
        return max(base, _llava_recall_from_tokens(f(std_tokens(rec['prediction'])), f(std_tokens(rec['answer']))))
    return scorer


def rule_overlap(drop_words):
    """val 에서 도출한 규칙이 test 오답에서 도출한 기존 S1+S2 의 일반어 목록과 얼마나 겹치는지."""
    a, b = set(drop_words), set(S1S2_REDUNDANT)
    return {'val 도출': sorted(a), 'S1+S2': sorted(b), '겹침': sorted(a & b), 'S1+S2 에만': sorted(b - a)}


# ───────────────────────── (4) 학습 데이터의 정답 분포에서 S1+S2 를 일반화: 'data' ─────────────────────────
# 손으로 만든 S1+S2 의 세 요소(일반어 생략 · 약어 · 표기/복수)를 train 정답 문자열의 통계에서 자동으로 학습한다.
# 입력은 train(필요하면 val)의 open 정답뿐이다 — 모델 예측도 test 도 쓰지 않으므로 test 를 본 규칙이 아니다.
#   일반어: "A" 와 "A w" 가 둘 다 정답으로 존재하고(삽입쌍 ≥ min_support), w 를 다른 단어로 바꾼 정답은 없는 단어 w
#          (axial / axial plane 은 plane 이 일반어, right lung / left lung 은 right 가 정보어라 제외)
#   약어: 정답에 나오는 짧은 토큰 x 가 같은 정답 집합의 다단어 구의 머리글자와 같으면 구→x
#   붙여쓰기: 인접한 두 토큰의 합이 정답 어휘에 있으면(x ray / xray) 합친다
#   복수: w 와 w+s / w+es 가 모두 정답 어휘에 있으면 복수형→단수형
def learn_rules(open_answers, min_support=2, max_ngram=4):
    from collections import defaultdict
    ans = [tuple(llava_normalize(a).split()) for a in open_answers]
    A = {t for t in ans if t}
    vocab = Counter(w for t in A for w in t)
    ins, pat = Counter(), defaultdict(set)
    for t in A:
        for i in range(len(t)):
            if len(t) >= 2 and t[:i] + t[i + 1:] in A:
                ins[t[i]] += 1
            if len(t) >= 2:
                pat[(t[:i], t[i + 1:])].add(t[i])
    sub = defaultdict(set)
    for ws in pat.values():
        if len(ws) >= 2:
            for w in ws:
                sub[w] |= ws - {w}
    generic = {w: n for w, n in ins.items() if n >= min_support and not sub[w]}
    ngrams = {t[i:i + n] for t in A for n in range(2, max_ngram + 1) for i in range(len(t) - n + 1)}
    abbr = {}
    stop = {'and', 'of', 'the', 'to', 'is', 'in', 'as', 'at', 'a', 'an', 'or', 'with', 'on', 'for', 'by', 'from', 'it', 'no', 'not'}
    singles = {t[0] for t in A if len(t) == 1}              # 약어는 그 자체로 정답이 되는 짧은 토큰이어야 한다
    for g in ngrams:
        ini = ''.join(w[0] for w in g)
        if 2 <= len(ini) <= 5 and ini in singles and not (set(g) & stop):
            abbr[' '.join(g)] = ini
    joined = {}
    for t in A:
        for i in range(len(t) - 1):
            if t[i] + t[i + 1] in vocab:
                joined[t[i] + ' ' + t[i + 1]] = t[i] + t[i + 1]
    plural = {}
    for w in vocab:
        for suf in ('s', 'es'):
            if w + suf in vocab:
                plural[w + suf] = w
    return {'generic_words': generic, 'abbreviations': abbr, 'joined': joined, 'plurals': plural,
            'n_answers': len(A), 'min_support': min_support}


def _apply_rules(text, rules, drop=False):
    t = ' ' + llava_normalize(text) + ' '
    for k, v in {**rules['joined'], **rules['abbreviations']}.items():
        t = t.replace(f' {k} ', f' {v} ')
    toks = [rules['plurals'].get(w, w) for w in t.split()]
    if drop:
        kept = [w for w in toks if w not in rules['generic_words']]
        toks = kept or toks
    return toks


def make_data_scorer(rules):
    """학습한(동결된) 규칙으로 만든 채점. open = max(원본 recall, 규칙 적용 recall, 규칙+일반어 생략 recall). closed 는 원본."""
    def scorer(rec):
        base = score_phi(rec)
        if rec['answer_type'] != 'OPEN':
            return base
        best = base
        for drop in (False, True):
            best = max(best, _llava_recall_from_tokens(_apply_rules(rec['prediction'], rules, drop), _apply_rules(rec['answer'], rules, drop)))
        return best
    return scorer


def bootstrap_ci(records, scorer=score_phi, n_boot=10000, seed=0, alpha=0.05):
    """문항을 복원추출해 open / closed / overall 의 (평균, 하한, 상한) 을 돌려준다. 표본 크기만으로 생기는 변동 범위를 본다
    (학습 시드 변동은 포함하지 않는다). 반환: {'open': (m, lo, hi), 'closed': ..., 'overall': ...} (0~100)."""
    import numpy as np
    rng = np.random.default_rng(seed)
    sc = np.array([scorer(r) for r in records], dtype=float)
    isop = np.array([r['answer_type'] == 'OPEN' for r in records])
    out = {}
    for name, mask in (('open', isop), ('closed', ~isop), ('overall', np.ones_like(isop))):
        x = sc[mask]
        idx = rng.integers(0, len(x), size=(n_boot, len(x)))
        means = 100 * x[idx].mean(axis=1)
        out[name] = (100 * x.mean(), float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2))))
    return out


def print_ci(records, paper, label='', scorer=score_phi, n_boot=10000):
    """부트스트랩 95% 신뢰구간과 논문 수치 비교를 한 표로 출력. paper = {'open','closed','overall'}"""
    ci = bootstrap_ci(records, scorer, n_boot)
    print(f'{label}  (n={len(records)}, 문항 복원추출 {n_boot}회, 95% 구간)')
    print(f"{'':9s}{'점수':>8s}{'하한':>8s}{'상한':>8s}{'논문':>8s}  논문이 구간 안?")
    for k in ('open', 'closed', 'overall'):
        m, lo, hi = ci[k]
        print(f"{k:9s}{m:8.2f}{lo:8.2f}{hi:8.2f}{paper[k]:8.2f}  {'예' if lo <= paper[k] <= hi else '아니오'}")
    return ci


# ───────────────────────── 자체 검사 ─────────────────────────
if __name__ == '__main__':
    R = lambda a, p, t: {'answer': a, 'prediction': p, 'answer_type': t}
    # closed: 포함이면 맞음 (phi) / 완전일치만 맞음 (native)
    assert closed_accuracy('Yes, there is a mass', 'yes') == 1.0
    assert score_native(R('yes', 'Yes, there is a mass', 'CLOSED')) == 0.0
    # closed 약점: 둘 다 포함하면 맞음 처리, 여러 단어 정답은 항상 0
    assert closed_accuracy('no yes', 'no') == 1.0
    assert closed_accuracy('right lung', 'right lung') == 0.0
    assert score_native(R('right lung', 'right lung', 'CLOSED')) == 1.0
    # open: 부분 점수 (phi) / 완전일치 (native)
    assert open_recall('right', 'right lung') == 0.5
    assert score_native(R('right lung', 'right', 'OPEN')) == 0.0
    assert open_recall('chest xray', 'xray') == 1.0          # 예측이 길어도 감점 없음
    # 정규화 차이: native 는 x-ray 표기를 통일하고 phi 는 단어로 쪼갠다
    assert pre_answer('X-ray.') == 'xray' and normalize('X-ray.') == ['x', 'ray']
    demo = [R('yes', 'Yes, mass', 'CLOSED'), R('no', 'yes', 'CLOSED'),
            R('right lung', 'right', 'OPEN'), R('xray', 'xray', 'OPEN')]
    compare(demo)

    # ── 공통 프로토콜 검사 ──
    assert norm_text('The X-ray.') == 'xray' and norm_text('Posterior-anterior') == 'posterior anterior'
    assert closed_correct('Yes, there is a mass', 'yes') == 1.0
    assert closed_correct('There is no mass, yes', 'yes') == 0.0       # 첫 단어만 본다 (phi 방식이면 1)
    assert closed_correct('no yes', 'yes') == 0.0                      # 둘 다 포함해도 첫 단어 기준 (phi 방식이면 1)
    assert closed_correct('Right lung.', 'right lung') == 1.0           # 여러 단어 정답도 정규화 후 완전일치면 맞음
    assert closed_correct('left', 'right') == 0.0
    assert open_em('The right lung', 'right lung') == 1.0 and open_em('right', 'right lung') == 0.0
    assert abs(open_f1('right', 'right lung') - 2 / 3) < 1e-9          # P=1, R=.5 → F1=.667
    assert open_f1('', '') == 1.0 and open_f1('x', '') == 0.0
    assert open_f1('chest xray', 'xray') == 2 / 3                      # 예측이 길면 F1 이 깎인다(recall 은 1)
    assert open_recall_u('chest xray', 'xray') == 1.0
    m = summarize_unified(demo)
    assert m['n_closed'] == 2 and m['n_open'] == 2
    # overall = (closed 1개 맞음 + open EM 1개 맞음) / 4 = 50
    assert abs(m['overall_acc'] - 50.0) < 1e-9, m
    # 로더: 임시 파일로 왕복 검사
    import json, tempfile, os
    d = tempfile.mkdtemp()
    json.dump([{'qid': 1, 'answer': 'yes', 'answer_type': 'CLOSED'}, {'qid': 2, 'answer': 'liver', 'answer_type': 'OPEN'}], open(os.path.join(d, 't.json'), 'w'))
    json.dump([{'qid': 1, 'answer': 'yes'}, {'qid': 2, 'answer': 'lung'}], open(os.path.join(d, 'r.json'), 'w'))
    rec = load_mumc(os.path.join(d, 't.json'), os.path.join(d, 'r.json'))
    assert summarize_unified(rec)['overall_acc'] == 50.0
    json.dump({'0': {'answer': 'no', 'answer_type': 'CLOSED'}}, open(os.path.join(d, 'ref.json'), 'w'))
    open(os.path.join(d, 'p.jsonl'), 'w').write(json.dumps({'question_id': 0, 'text': 'No, it is normal.'}) + '\n')
    assert summarize_unified(load_llava(os.path.join(d, 'ref.json'), os.path.join(d, 'p.jsonl')))['closed_acc'] == 100.0
    # ── S1+S2 검사: 허용하는 것 / 허용하지 않는 것 ──
    O = lambda a, p: {'answer': a, 'prediction': p, 'answer_type': 'OPEN'}
    for gt, pr in [('Posterior-Anterior', 'PA'), ('T2-weighted', 'T2'), ('Axial plane', 'axial'), ('Maxillary sinuses', 'Maxillary sinus'),
                   ('DWI Diffusion weighted', 'DWI'), ('Xray', 'X-ray'), ('T2 weighted MRI', 'T2 weighted')]:
        assert score_s1s2(O(gt, pr)) == 1.0, (gt, pr)
    for gt, pr in [('plain film', 'Xray'), ('brain', 'CNS'), ('Right lung', 'Right'), ('CT with contrast', 'CT'), ('T2-MRI', 'MRI'),
                   ('Left', 'Right'), ('2.5cm x 1.7cm x 1.6cm', '2.5')]:
        assert score_s1s2(O(gt, pr)) == score_phi(O(gt, pr)) < 1.0, (gt, pr)
    assert score_s1s2(O('3.4', '34')) == 0.0                                  # 소수점은 숫자를 합치지 않는다
    assert score_s1s2({'answer': 'yes', 'prediction': 'no', 'answer_type': 'CLOSED'}) == 0.0   # closed 는 그대로
    assert all(score_s1s2(r) >= score_phi(r) for r in demo)                    # 원본보다 낮아지지 않는다
    ci = bootstrap_ci(demo * 50, n_boot=500)
    assert ci['overall'][1] <= ci['overall'][0] <= ci['overall'][2]
    print()
    report({'demo': demo})
    print('자체 검사 OK')
