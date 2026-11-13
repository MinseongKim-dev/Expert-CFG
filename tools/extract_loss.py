"""학습 출력에서 epoch 별 'Averaged stats ... loss' 만 뽑아 곡선 셀에 붙여 넣을 수 있는 형태로 출력한다.

사용 예
  python tools/extract_loss.py MUMC_expert_cfg_reproduction.ipynb 19 SLAKE   # 노트북 19 번 셀의 저장된 출력에서
  python tools/extract_loss.py MUMC_expert_cfg_reproduction.ipynb auto SLAKE # 'Averaged stats' 가 가장 많은 셀을 자동 선택
  python tools/extract_loss.py train_output.txt - SLAKE                      # 텍스트 파일에서 (복사해 붙여 넣은 출력)

노트북 출력은 IDE 에서 저장(Ctrl+S)한 뒤에 읽힌다. 출력의 마지막 줄은 곡선 셀 위쪽의 LOSS_MANUAL 에 붙여 넣는 한 줄이다.
"""
import json, re, sys

PAT = re.compile(r'Train Epoch: \[(\d+)\] Total time.*?\nAveraged stats: lr: [\d.eE+-]+\s+loss: ([\d.eE+-]+)')


def from_text(text):
    return {int(e): float(l) for e, l in PAT.findall(text)}


def cell_text(cell):
    return ''.join(''.join(o.get('text', [])) for o in cell.get('outputs', []) if o.get('output_type') == 'stream')


def main():
    if len(sys.argv) < 3:
        print(__doc__); sys.exit(1)
    path, which = sys.argv[1], sys.argv[2]
    label = sys.argv[3] if len(sys.argv) > 3 else 'RUN'
    if path.endswith('.ipynb'):
        cells = json.load(open(path, encoding='utf8'))['cells']
        cand = {i: from_text(cell_text(c)) for i, c in enumerate(cells) if c['cell_type'] == 'code'}
        cand = {i: d for i, d in cand.items() if d}
        if not cand:
            sys.exit('저장된 출력에 Averaged stats 줄이 없습니다. IDE 에서 노트북을 저장했는지, 학습 셀 출력이 남아 있는지 확인하세요.')
        if which == 'auto':
            print('Averaged stats 가 있는 셀:', {i: len(d) for i, d in cand.items()})
            i = max(cand, key=lambda k: len(cand[k]))
        else:
            i = int(which)
            if i not in cand:
                sys.exit(f'{i} 번 셀에는 Averaged stats 줄이 없습니다. 후보: {sorted(cand)}')
        loss = cand[i]
        print(f'셀 {i} 에서 {len(loss)} epoch 읽음')
    else:
        loss = from_text(open(path, encoding='utf8', errors='ignore').read())
        print(f'{path} 에서 {len(loss)} epoch 읽음')
    if not loss:
        sys.exit('읽힌 epoch 가 없습니다')
    missing = [e for e in range(max(loss) + 1) if e not in loss]
    print('빠진 epoch:', missing or '없음', '| 첫/끝 loss:', loss[min(loss)], loss[max(loss)])
    print(f"\nLOSS_MANUAL['{label}'] = {dict(sorted(loss.items()))}")


if __name__ == '__main__':
    main()
