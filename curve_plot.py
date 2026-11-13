"""모델 공통 학습 곡선 그림 (2x2). 선택 기준은 항상 val 이고, test 곡선은 진단용으로만 겹쳐 그린다.

  (a) train loss (선택: 학습률)
  (b) val overall vs test overall, 세로선 = val 로 고른 epoch
  (c) epoch 별 open / closed / overall  (실선 = val, 점선 = test, 회색 가로 점선 = 논문 test 값)
  (d) "이 epoch 에서 학습을 멈췄다면" 보고했을 값: 그 시점까지 val overall 이 가장 높았던 epoch 의 test 값 (test 가 있는 epoch 만).
      test 가 없으면 val 의 best-so-far 를 그린다. 어느 쪽이든 선택에 test 는 쓰지 않는다.

입력 형식 (epoch 는 0 부터 센 정수 또는 1 부터 센 정수 중 하나로 통일해서 넣을 것)
  train_loss : {epoch: float}
  lr         : {epoch: float}                (선택)
  val, test  : {epoch: {'open': %, 'closed': %, 'overall': %}}   test 는 평가한 epoch 만 있어도 된다
  paper      : {'open': %, 'closed': %, 'overall': %}            (선택)
"""
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

KEYS = ('open', 'closed', 'overall')
COL = {'open': 'tab:blue', 'closed': 'tab:orange', 'overall': 'tab:green'}


def select_by_val(val):
    """val overall 이 최대인 epoch (동점이면 이른 epoch)."""
    return max(sorted(val), key=lambda e: (val[e]['overall'], -e))


def best_so_far(val):
    """각 epoch 까지의 val-best epoch 를 {epoch: 그때의 best epoch} 로 돌려준다."""
    out, best = {}, None
    for e in sorted(val):
        if best is None or val[e]['overall'] > val[best]['overall']:
            best = e
        out[e] = best
    return out


def _sparse(test, val):
    """test 가 5개 미만(보통 val 선택 epoch 와 마지막 epoch 두 점)이면 곡선이 아니므로 report 레이아웃(막대 비교)을 쓴다.
    5 epoch 마다 평가하는 등 5개 이상이면 점을 선으로 이은 곡선 레이아웃을 쓴다."""
    return len(test) < 5


def _bars(a, groups, series, ylim=None):
    """groups: 막대 묶음 이름 목록, series: [(label, [값...], 스타일 dict)]"""
    n = len(series); w = 0.8 / n
    for j, (lab, vals, kw) in enumerate(series):
        xs = [i + (j - (n - 1) / 2) * w for i in range(len(groups))]
        a.bar(xs, [0 if v is None else v for v in vals], w * 0.95, label=lab, **kw)
        for x, v in zip(xs, vals):
            if v is not None:
                a.text(x, v + 0.8, f'{v:.1f}', ha='center', va='bottom', fontsize=8)
    a.set_xticks(range(len(groups))); a.set_xticklabels(groups); a.set_ylabel('%')
    if ylim:
        a.set_ylim(*ylim)


def _ticks(axes, val, test, train_loss):
    last = max([max(val)] + ([max(test)] if test else []) + ([max(train_loss)] if train_loss else []))
    step = 5 if last >= 15 else 1
    for a in axes:
        a.xaxis.set_major_locator(MultipleLocator(step)); a.xaxis.set_major_formatter(FormatStrFormatter('%d')); a.grid(axis='x', alpha=0.2)


def _plot_report(title, train_loss, lr, val, test, paper, coverage, out_png, sel):
    """test 를 한두 번만 평가한 경우의 레이아웃: 선택 근거(val) + 최종 test 비교 + 후보 coverage."""
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))
    # (a) train loss
    a = ax[0, 0]
    if train_loss:
        xs = sorted(train_loss); a.plot(xs, [train_loss[e] for e in xs], marker='.', label='train loss'); a.set_yscale('log'); a.legend(loc='upper right')
    else:
        a.text(0.5, 0.5, 'no train-loss log found', ha='center', va='center', transform=a.transAxes, color='gray')
    a.set_title('(a) train loss'); a.set_xlabel('epoch')
    # (b) val 곡선 = 선택 근거
    a = ax[0, 1]
    for k in KEYS:
        xs = sorted(val); a.plot(xs, [val[e][k] for e in xs], color=COL[k], lw=1.6 if k == 'overall' else 1.0, label=f'val {k}')
    a.axvline(sel, color='green', lw=1.0, label=f'val-selected epoch {sel}')
    a.set_title('(b) validation per epoch (the only signal used for selection)', fontsize=10); a.set_xlabel('epoch'); a.set_ylabel('%'); a.legend(loc='lower right', fontsize=8)
    # (c) test 비교
    a = ax[1, 0]
    series = []
    if sel in test:
        series.append((f'val-selected (epoch {sel})', [test[sel][k] for k in KEYS], dict(color='tab:blue')))
    last = max(test) if test else None
    if last is not None and last != sel:
        series.append((f'last epoch ({last})', [test[last][k] for k in KEYS], dict(color='tab:cyan')))
    if paper:
        series.append(('paper', [paper.get(k) for k in KEYS], dict(color='lightgray', hatch='//', edgecolor='gray')))
    if series:
        _bars(a, KEYS, series, ylim=(0, 120)); a.legend(loc='upper center', fontsize=8, ncol=3)
    a.set_title('(c) test score, evaluated once per checkpoint (never used for selection)', fontsize=10)
    # (d) candidate coverage
    a = ax[1, 1]
    if coverage:
        cov = [coverage[k]['coverage'] if isinstance(coverage[k], dict) else coverage[k] for k in KEYS]
        _bars(a, KEYS, [('candidate coverage (answer is in the train-only list)', cov, dict(color='lightsteelblue'))], ylim=(0, 120))
        if sel in test:
            a.plot(range(len(KEYS)), [test[sel][k] for k in KEYS], ls='none', marker='D', ms=8, color='tab:blue', markeredgecolor='k', label=f'test score (epoch {sel})')
        a.set_title('(d) candidate coverage of test answers vs achieved score', fontsize=10); a.legend(loc='upper center', fontsize=8, ncol=2)
    else:
        bs = best_so_far(val); xs = sorted(bs)
        for k in KEYS:
            a.step(xs, [val[bs[e]][k] for e in xs], where='post', color=COL[k], label=f'val {k}')
        a.axvline(sel, color='green', lw=1.0)
        a.set_title('(d) val best-so-far (selection trace)', fontsize=10); a.set_xlabel('epoch'); a.set_ylabel('%'); a.legend(loc='lower right', fontsize=8)
    if sel in test:
        t = test[sel]; head = f"val-selected epoch {sel}: test overall {t['overall']:.2f} / open {t['open']:.2f} / closed {t['closed']:.2f}"
    else:
        v = val[sel]; head = f"val-selected epoch {sel}: val overall {v['overall']:.2f} / open {v['open']:.2f} / closed {v['closed']:.2f} (test not evaluated)"
    fig.suptitle(f'{title}  |  {head}', fontsize=11)
    _ticks([ax[0, 0], ax[0, 1]] + ([] if coverage else [ax[1, 1]]), val, test, train_loss)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if out_png:
        fig.savefig(out_png, dpi=130)
    return fig, sel


def closed_confusion(y_true, y_pred):
    """yes/no 문항의 혼동행렬과 클래스별 precision / recall / F1. y_pred 가 yes/no 가 아니면 'other' 로 센다.
    입력은 소문자·구두점 정리된 문자열 목록. 점수(확률)가 아니라 예측 답만 쓴다."""
    cls = ('yes', 'no'); cols = ('yes', 'no', 'other')
    cm = {t: {c: 0 for c in cols} for t in cls}
    for t, p in zip(y_true, y_pred):
        if t in cm:
            cm[t][p if p in ('yes', 'no') else 'other'] += 1
    m = {}
    for c in cls:
        tp = cm[c][c]; pred = sum(cm[t][c] for t in cls); true = sum(cm[c].values())
        pr = tp / pred if pred else 0.0; rc = tp / true if true else 0.0
        m[c] = {'precision': 100 * pr, 'recall': 100 * rc, 'f1': 100 * (2 * pr * rc / (pr + rc) if pr + rc else 0.0), 'support': true}
    m['macro'] = {k: sum(m[c][k] for c in cls) / 2 for k in ('precision', 'recall', 'f1')}
    n = sum(sum(r.values()) for r in cm.values())
    m['accuracy'] = 100 * sum(cm[c][c] for c in cls) / n if n else 0.0
    return cm, m


def plot_closed_f1(title, y_true, y_pred, out_png=None):
    """closed(yes/no) 예측의 혼동행렬 + 클래스별 precision/recall/F1 막대. (ROC 는 확률 점수가 있어야 해서 여기서는 그리지 않는다)"""
    cm, m = closed_confusion(y_true, y_pred)
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    cols = ('yes', 'no', 'other'); rows = ('yes', 'no')
    a = ax[0]; a.imshow([[cm[r][c] for c in cols] for r in rows], cmap='Blues')
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            a.text(j, i, cm[r][c], ha='center', va='center', fontsize=14, color='black')
    a.set_xticks(range(3)); a.set_xticklabels([f'pred {c}' for c in cols]); a.set_yticks(range(2)); a.set_yticklabels([f'true {r}' for r in rows])
    a.set_title(f'confusion matrix (closed, n={sum(sum(v.values()) for v in cm.values())})')
    groups = ('yes', 'no', 'macro')
    _bars(ax[1], groups, [(k, [m[g][k] for g in groups], dict(color=c)) for k, c in (('precision', 'tab:blue'), ('recall', 'tab:orange'), ('f1', 'tab:green'))], ylim=(0, 120))
    ax[1].legend(loc='upper center', fontsize=8, ncol=3); ax[1].set_title(f"precision / recall / F1  (accuracy {m['accuracy']:.1f}%)")
    fig.suptitle(title, fontsize=11); fig.tight_layout(rect=(0, 0, 1, 0.94))
    if out_png:
        fig.savefig(out_png, dpi=130)
    return fig, m


def plot_curves(title, train_loss=None, lr=None, val=None, test=None, paper=None, out_png=None, selected=None, coverage=None):
    val, test = val or {}, test or {}
    sparse = _sparse(test, val)
    if sparse:      # test 가 한두 번만 평가된 경우: 우리 방식(val 선택, test 한 번)에 맞는 레이아웃
        assert val, 'val 곡선이 없습니다 (선택 기준이 필요)'
        return _plot_report(title, train_loss, lr, val, test, paper, coverage, out_png, select_by_val(val) if selected is None else selected)
    tkw = dict(ls='none', marker='o', ms=7) if sparse else dict(marker='o', ms=4)
    assert val, 'val 곡선이 없습니다 (선택 기준이 필요)'
    sel = select_by_val(val) if selected is None else selected
    fig, ax = plt.subplots(2, 2, figsize=(14, 8.5))

    # (a) loss
    a = ax[0, 0]
    if train_loss:
        xs = sorted(train_loss)
        a.plot(xs, [train_loss[e] for e in xs], marker='.', label='train loss')
        a.set_yscale('log')
    if not train_loss:
        a.text(0.5, 0.5, 'no train-loss log found', ha='center', va='center', transform=a.transAxes, color='gray')
    a.set_title('(a) train loss' + (' / learning rate' if lr else '')); a.set_xlabel('epoch')
    h, l = a.get_legend_handles_labels()
    if lr:
        b = a.twinx(); xs = sorted(lr)
        b.plot(xs, [lr[e] for e in xs], ':', color='gray', label='lr'); b.set_ylabel('lr')
        h2, l2 = b.get_legend_handles_labels(); h += h2; l += l2
    if h:
        a.legend(h, l, loc='upper right')

    # (b) val vs test overall
    a = ax[0, 1]
    xs = sorted(val); a.plot(xs, [val[e]['overall'] for e in xs], marker='.', color='tab:orange', label='validation overall')
    if test:
        xt = sorted(test); a.plot(xt, [test[e]['overall'] for e in xt], color='tab:blue', label='test overall (diagnostic)', **tkw)
    a.axvline(sel, color='green', lw=0.8, label=f'val-selected epoch {sel}')
    a.set_title('(b) validation vs test overall'); a.set_xlabel('epoch'); a.set_ylabel('%'); a.legend(loc='lower right')

    # (c) per-epoch metrics
    a = ax[1, 0]
    for k in KEYS:
        xs = sorted(val); a.plot(xs, [val[e][k] for e in xs], color=COL[k], lw=1.2, label=f'val {k}')
        if test:
            xt = sorted(test); a.plot(xt, [test[e][k] for e in xt], color=COL[k], label=f'test {k}', **({'ls': 'none', 'marker': 'o', 'ms': 7} if sparse else {'ls': '--', 'marker': 'o', 'ms': 3, 'lw': 0.9}))
        if paper and paper.get(k) is not None:
            a.axhline(paper[k], color=COL[k], ls=':', lw=0.9, alpha=0.7)
    a.set_title('(c) accuracy per epoch (solid = val, dashed = test, dotted = paper test)'); a.set_xlabel('epoch'); a.set_ylabel('%')
    a.legend(loc='lower right', fontsize=7, ncol=2)

    # (d) reported value if training stopped at this epoch (val-based selection)
    a = ax[1, 1]; bs = best_so_far(val)
    dense = len(test) >= max(5, 0.5 * len(val))      # test 가 epoch 의 절반 이상이어야 '그 시점 val-best 의 test' 계단을 그릴 수 있다
    if test and dense and any(bs[e] in test for e in bs):
        lab = 'test @ val-best-so-far epoch'
        for k in KEYS:
            pts = [(e, test[bs[e]][k]) for e in sorted(bs) if bs[e] in test]
            a.step([p[0] for p in pts], [p[1] for p in pts], where='post', color=COL[k], label=f'{k} (test)')
    else:
        lab = 'lines: val best-so-far | markers: test at evaluated epochs' if test else 'val best-so-far (no per-epoch test)'
        for k in KEYS:
            a.step(sorted(bs), [val[bs[e]][k] for e in sorted(bs)], where='post', color=COL[k], label=f'{k} (val)')
            if test:
                a.plot(sorted(test), [test[e][k] for e in sorted(test)], ls='none', marker='*', ms=11, color=COL[k], markeredgecolor='k', label=f'{k} (test)')
    if paper:
        for k in KEYS:
            if paper.get(k) is not None:
                a.axhline(paper[k], color='gray', ls=':', lw=0.7)
    a.axvline(sel, color='red', lw=0.8)
    a.set_title('(d) reported value if training stopped at this epoch\n[' + lab + ']', fontsize=10); a.set_xlabel('epoch'); a.set_ylabel('%'); a.legend(loc='lower right')

    if sel in test:
        t = test[sel]; head = f"val-selected epoch {sel}: test overall {t['overall']:.2f} / open {t['open']:.2f} / closed {t['closed']:.2f}"
    else:
        v = val[sel]; head = f"val-selected epoch {sel}: val overall {v['overall']:.2f} / open {v['open']:.2f} / closed {v['closed']:.2f} (test not evaluated)"
    fig.suptitle(f'{title}  |  {head}', fontsize=11)
    # x 축 눈금: epoch 가 15 이상이면 5 단위, 그보다 짧으면 1 단위
    last = max([max(val)] + ([max(test)] if test else []) + ([max(train_loss)] if train_loss else []))
    step = 5 if last >= 15 else 1
    for a in ax.ravel():
        a.xaxis.set_major_locator(MultipleLocator(step)); a.xaxis.set_major_formatter(FormatStrFormatter('%d'))
        a.grid(axis='x', alpha=0.2)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if out_png:
        fig.savefig(out_png, dpi=130)
    return fig, sel
