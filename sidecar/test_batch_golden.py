"""B=1 golden — /synthesize_batch の 1 件が /synthesize とバイト一致するか検証する。

なぜ必要か:
  batch.py は upstream irodori_tts の synthesize() オーケストレーションを「忠実に転記」している。
  B=1 のバッチは row-0 noise ＋ 共有 max 長が単発に収束するため、**バイト一致するはず**。
  upstream の bump で duration/decode/trim の振る舞いがズレるとここで即失敗する（ドリフト検知のカナリア）。

使い方（サイドカーを新コードで起動した状態で）:
  python sidecar/test_batch_golden.py [--url http://127.0.0.1:18080]
"""
import argparse
import base64
import hashlib
import json
import sys
import urllib.request


def _post(url: str, path: str, body: dict, accept: str | None = None) -> bytes:
    headers = {"Content-Type": "application/json"}
    if accept:
        headers["Accept"] = accept
    req = urllib.request.Request(
        url + path, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        method="POST", headers=headers,
    )
    with urllib.request.urlopen(req, timeout=180) as resp:
        return resp.read()


def main() -> int:
    ap = argparse.ArgumentParser(description="B=1 batch == single golden test")
    ap.add_argument("--url", default="http://127.0.0.1:18080", help="サイドカー URL")
    ap.add_argument("--text", default="ゴールデンテスト、バッチと単発の一致を確認します。")
    args = ap.parse_args()

    # no_ref（ref_wav 無し）＋固定 seed の最小 vd リクエスト。声定義に依存せず自己完結。
    payload = {
        "mode": "vd",
        "text": args.text,
        "caption": "落ち着いた女性の声。",
        "seed": 42,
        "num_steps": 24,
        "t_schedule_mode": "sway",
    }
    single = _post(args.url, "/synthesize", payload, accept="audio/wav")
    resp = json.loads(_post(args.url, "/synthesize_batch", {"mode": "vd", "requests": [payload]}))
    batch = base64.b64decode(resp["wavs"][0])

    ok = single == batch
    print(f"single: sha256={hashlib.sha256(single).hexdigest()[:20]} len={len(single)}")
    print(f"batch : sha256={hashlib.sha256(batch).hexdigest()[:20]} len={len(batch)}")
    print(f"GOLDEN {'PASS' if ok else 'FAIL'} (B=1 batch byte-identical to single)")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
