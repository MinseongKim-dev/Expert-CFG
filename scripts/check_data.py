"""
다운로드된 annotation JSON이 Expert-CFG 실행에 필요한 필드를 갖추고 있는지 확인합니다.

Usage:
    python scripts/check_data.py data/RadVQA/test.json
    python scripts/check_data.py data/Slake/test.json
    python scripts/check_data.py data/PathVQA/test.json
"""

import json
import sys
from pathlib import Path


REQUIRED_FIELDS = ["top_k_captions", "highlights"]
SAMPLE_FIELDS = ["question", "answer", "image_name"]


def check_json(path: str):
    p = Path(path)
    if not p.exists():
        print(f"[ERROR] 파일 없음: {path}")
        return False

    with open(p) as f:
        data = json.load(f)

    if isinstance(data, dict):
        items = list(data.values())
    elif isinstance(data, list):
        items = data
    else:
        print(f"[ERROR] 알 수 없는 JSON 구조: {type(data)}")
        return False

    print(f"총 {len(items)}개 항목")
    if not items:
        print("[ERROR] 데이터가 비어있습니다.")
        return False

    sample = items[0]
    ok = True

    print("\n[필수 필드 (Expert-CFG 실행 필요)]")
    for field in REQUIRED_FIELDS:
        if field in sample:
            val = sample[field]
            preview = str(val)[:80] + "..." if len(str(val)) > 80 else str(val)
            print(f"  ✅ {field}: {preview}")
        else:
            print(f"  ❌ {field}: 없음 — Baiduyun 버전이 아닙니다")
            ok = False

    print("\n[기본 필드]")
    for field in SAMPLE_FIELDS:
        status = "✅" if field in sample else "⚠️ "
        print(f"  {status} {field}: {sample.get(field, '없음')}")

    print()
    if ok:
        print("✅ 이 JSON은 Expert-CFG 실행 가능합니다.")
    else:
        print("❌ top_k_captions / highlights 없음 — Baiduyun에서 받은 파일이 필요합니다.")

    return ok


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/check_data.py <path/to/test.json>")
        sys.exit(1)
    check_json(sys.argv[1])
