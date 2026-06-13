"""
irodori_server.py — Irodori-TTS サイドカー HTTP サーバ

責務:
  - 重い irodori-tts (torch + CUDA + git 依存) を L2 本体から切り離し、
    別プロセス (irodori の uv venv) で常駐させる。
  - モデルを起動時に 1 回ロードしてウォーム保持し、L2 からの HTTP 合成要求に
    低レイテンシで応答する (実測 ~0.9s/合成、RTF ~0.1 @ RTX 5090)。
  - キャラクター / 感情ロジックは一切持たない「generic な TTS エンドポイント」に
    徹する。声の同一性 (ref_wav / caption) も感情 (絵文字 / caption 修飾) も呼出側
    (L2 の tts._call_irodori) が組み立てて送ってくる。これにより本サーバは他ツール
    からも再利用できる (疎結合・高凝集)。

【なぜ HTTP サイドカーか】
  irodori は torch 2.10 + CUDA + git 依存 (dacvae / silentcipher) の重いスタックで
  独自の uv venv を持つ。軽量な L2 (Python 3.11 / stdlib 中心) にこれを混ぜると uv
  解決が脆くなり GPU メモリも常時占有する。VOICEVOX が Docker コンテナ + HTTP で
  分離されているのと同じ発想で、irodori も別プロセス + HTTP で疎結合にする。

【対応モード】
  "vd" : VoiceDesign (Irodori-TTS-600M-v3-VoiceDesign)。確定版 (PoC) の構成は
         「caption (声質/話し方) + self-ref アンカー (ref_wav) + seed + sway + cfg_scale_speaker」。
         感情は L2 側で本文末の正規絵文字 + caption サフィックスに変換して渡す。
         self-ref アンカーは必須 (no-ref だと台詞ごとに声がドリフトすると PoC で実証)。
         ※ 旧 clone モード (VOICEPEAK 出力を参照クローン) は利用規約リスクのため廃止。

【エンドポイント】
  GET  /health     → {"status":"ok","models":["vd"]}  (listening = 全モデルロード完了)
  POST /synthesize → audio/wav (PCM16) bytes。X-Sample-Rate ヘッダに 48000 を付与。
       リクエスト JSON (vd):
         {
           "mode": "vd",
           "text": "合成するテキスト (本文末に正規絵文字が付くことがある)",
           "caption": "気品のある若い女性の声。… (＋感情サフィックス)",
           "ref_wav": "T:\\\\irodori-tts\\\\reference_voices\\\\mimi_ref.wav",  # self-ref アンカー (必須)
           "duration_scale": 1.0,          # = max(0.85, 100/speed)
           "num_steps": 24,
           "t_schedule_mode": "sway",
           "cfg_scale_speaker": 5.0,
           "seed": 3,                      # キャラ毎に固定
           "seconds": 2.16                 # 任意: 手動 duration。指定時は尺予測器を
                                           #   バイパス (短文の末尾幻聴抑制)。無指定=予測器
         }

【起動方法】
  scripts/run_irodori_sidecar.ps1 から起動するのが通常。直接起動する場合:

    $env:HF_HOME = "T:\\irodori-tts\\hf-cache"
    uv run --directory "T:\\irodori-tts\\Irodori-TTS" --no-sync `
      python <repo>/aibyss-lab-lounge/sidecar/irodori_server.py `
      --port 50080 --models vd

  irodori_tts パッケージは L2_IRODORI_REPO (既定 T:\\irodori-tts\\Irodori-TTS) を
  sys.path に追加して import する。weights は HF_HOME (= hf-cache) からロードされる。
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

# Windows コンソールの既定は cp932 で、日本語ログが文字化けする (別ウィンドウ起動時に顕著)。
# bench スクリプトと同様に stdout/stderr を UTF-8 に切り替えてログを読めるようにする。
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001 — 古い Python / 特殊なストリームでも起動は継続
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("irodori_server")

# ─── irodori_tts を import 可能にする ────────────────────────────────
# 本スクリプトは L2 リポ配下 (aibyss-lab-lounge/sidecar/) に置かれており、
# irodori_tts パッケージ本体は別ディレクトリ (T:\irodori-tts\Irodori-TTS) にある。
# bench スクリプト同様、対象リポを sys.path 先頭に追加してから import する
# (uv run --directory でも import パスに自動で載らないため明示する)。
_IRODORI_REPO = os.environ.get("L2_IRODORI_REPO", r"T:\irodori-tts\Irodori-TTS")
if _IRODORI_REPO and Path(_IRODORI_REPO).is_dir() and _IRODORI_REPO not in sys.path:
    sys.path.insert(0, _IRODORI_REPO)


# ─── モデル定義 ──────────────────────────────────────────────────────
# mode → HuggingFace repo。weights は HF_HOME (hf-cache) に DL 済みの想定。
# 確定版は VoiceDesign 一本 (旧 clone=500M は利用規約リスクのため廃止)。
_MODE_REPOS: dict[str, str] = {
    "vd": "Aratako/Irodori-TTS-600M-v3-VoiceDesign",
}


def _warmup_runtime(runtime) -> None:
    """CUDA カーネル初期化のためダミー 1 合成を流す (PoC bench_load_report 推奨)。

    初回合成は CUDA カーネル初期化でやや遅いため、起動時に 1 回ウォームアップして
    本番の初回発話レイテンシを安定させる。失敗しても起動は継続する。
    """
    from irodori_tts.inference_runtime import SamplingRequest

    try:
        runtime.synthesize(
            SamplingRequest(
                text="ウォームアップ。",
                caption="落ち着いた女性の声。",
                no_ref=True,
                num_steps=24,
                t_schedule_mode="sway",
                seed=0,
            ),
            log_fn=None,
        )
    except Exception as exc:  # noqa: BLE001 — ウォームアップ失敗は致命ではない
        logger.warning("ウォームアップ合成に失敗 (無視して継続): %s", exc)


def _load_runtimes(modes: list[str]) -> dict:
    """
    指定モードの InferenceRuntime を直接インスタンス化し、ウォームアップして dict で返す。

    【なぜ get_cached_runtime を使わないか】
      irodori の get_cached_runtime は「単一エントリ」キャッシュで、別 checkpoint を
      要求すると古い runtime を unload して再ロードしてしまう。将来 mode を増やしても
      再ロードのスラッシングを避けるため、from_key() を mode ごとに呼んで常駐させる
      (VRAM 32GB に対し vd 600M の working set は ~4GB)。

    Args:
        modes: ["vd"] のサブセット

    Returns:
        {mode: InferenceRuntime}
    """
    from huggingface_hub import hf_hub_download
    from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey

    runtimes: dict = {}
    for mode in modes:
        repo = _MODE_REPOS[mode]
        logger.info("[%s] checkpoint 解決中: %s", mode, repo)
        ckpt = hf_hub_download(repo_id=repo, filename="model.safetensors")
        logger.info("[%s] モデルロード中 (cuda / bf16)...", mode)
        t0 = time.perf_counter()
        # bench (bench_char_emotion.py / bench_load.py) と同一の確定本番設定。
        runtimes[mode] = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=ckpt,
                model_device="cuda",
                model_precision="bf16",
                codec_device="cuda",
                codec_precision="bf16",
            )
        )
        logger.info("[%s] ロード完了 (%.1f 秒) → ウォームアップ中...", mode, time.perf_counter() - t0)
        _warmup_runtime(runtimes[mode])
        logger.info("[%s] ウォームアップ完了", mode)
    return runtimes


def _wav_bytes_from_result(result) -> tuple[bytes, int]:
    """
    SamplingResult.audio (torch.Tensor) を PCM16 WAV バイト列に変換する。

    【なぜ PCM16 か】
      L2 側 (_call_irodori) は VOICEVOX 経路と同じく stdlib の `wave` モジュールで
      WAV ヘッダから duration / sample_rate を読む。`wave` は float WAV を扱えない
      ため、PCM16 で返して L2 のコードパスを VOICEVOX と揃える。再生側
      (audio_io.play_audio_file) は soundfile で読むので PCM16 で問題ない。

    Returns:
        (wav_bytes, sample_rate)
    """
    import soundfile as sf
    import torch

    audio_cpu = result.audio.detach().to(device="cpu", dtype=torch.float32)
    # audio shape は (channels, samples)。通常モノラル (1, N)。
    if audio_cpu.ndim == 2 and audio_cpu.shape[0] == 1:
        audio_np = audio_cpu.squeeze(0).numpy()         # (N,)
    elif audio_cpu.ndim == 2:
        audio_np = audio_cpu.T.numpy()                  # (N, channels)
    else:
        audio_np = audio_cpu.numpy()

    sample_rate = int(result.sample_rate)
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue(), sample_rate


class _Handler(BaseHTTPRequestHandler):
    """irodori サイドカーのリクエストハンドラ。

    runtimes は server インスタンス (ThreadingHTTPServer) に attach され、
    全リクエストで共有される。実合成は InferenceRuntime 内部の _infer_lock で
    直列化されるため、同時要求が来てもGPU上は 1 件ずつ処理される。
    """

    # アクセスログを logging 経由に統一 (デフォルトの stderr 直書きを抑制)。
    def log_message(self, fmt: str, *args) -> None:  # noqa: A003
        logger.debug("%s - %s", self.address_string(), fmt % args)

    # ── ヘルパ ──────────────────────────────────────────────
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ── GET /health ─────────────────────────────────────────
    def do_GET(self) -> None:  # noqa: N802
        if self.path.rstrip("/") in ("/health", ""):
            self._send_json(200, {"status": "ok", "models": sorted(self.server.runtimes)})
        else:
            self._send_json(404, {"error": f"unknown path: {self.path}"})

    # ── POST /synthesize ────────────────────────────────────
    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/synthesize":
            self._send_json(404, {"error": f"unknown path: {self.path}"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length) if length > 0 else b""
            req = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            self._send_json(400, {"error": f"invalid JSON body: {exc}"})
            return

        mode = str(req.get("mode", "")).strip().lower()
        runtime = self.server.runtimes.get(mode)
        if runtime is None:
            self._send_json(
                400,
                {"error": f"unknown/unloaded mode: {mode!r}. loaded: {sorted(self.server.runtimes)}"},
            )
            return

        text = req.get("text")
        if not text or not str(text).strip():
            self._send_json(400, {"error": "text is required and must be non-empty"})
            return

        try:
            wav_bytes, sample_rate, synth_sec = self.server.synthesize(mode, req)
        except Exception as exc:  # noqa: BLE001 — どんな失敗も 500 で返し配信を止めない
            logger.exception("合成失敗 (mode=%s)", mode)
            self._send_json(500, {"error": f"{type(exc).__name__}: {exc}"})
            return

        logger.info(
            "synth ok: mode=%s chars=%d wav=%d bytes synth=%.3fs",
            mode, len(str(text)), len(wav_bytes), synth_sec,
        )
        self.send_response(200)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(wav_bytes)))
        self.send_header("X-Sample-Rate", str(sample_rate))
        self.end_headers()
        self.wfile.write(wav_bytes)


class _IrodoriServer(ThreadingHTTPServer):
    """runtimes を保持する ThreadingHTTPServer。

    ThreadingHTTPServer にすることで /health がモデル合成中でもブロックされない。
    実 GPU 処理は runtime._infer_lock で直列化されるため競合は起きない。
    """

    daemon_threads = True

    def __init__(self, addr, runtimes: dict):
        super().__init__(addr, _Handler)
        self.runtimes = runtimes

    def synthesize(self, mode: str, req: dict) -> tuple[bytes, int, float]:
        """1 リクエスト分を VoiceDesign 合成して (wav_bytes, sample_rate, synth_sec) を返す。

        声の同一性・感情・seed・schedule・cfg はすべて呼出側 (L2 tts._call_irodori) が
        決めて渡す。本サーバはそれを SamplingRequest にパススルーするだけ (generic を維持)。

        確定版 (PoC) の構成:
          - caption (必須): 声質/話し方
          - ref_wav (self-ref アンカー): 音色の同一性。無ければ no_ref=True にフォールバック
            するが、確定版では台詞跨ぎドリフト防止のため必ず送る。
          - t_schedule_mode=sway / num_steps=24 / cfg_scale_speaker=5.0 / seed=キャラ毎固定
        """
        from irodori_tts.inference_runtime import SamplingRequest

        runtime = self.runtimes[mode]
        caption = req.get("caption")
        if not caption or not str(caption).strip():
            raise ValueError("vd モードは caption が必須です")
        ref_wav = req.get("ref_wav")
        seed = req.get("seed", None)
        # seconds: 手動 duration (秒)。指定時は irodori の duration predictor を
        # バイパスして尺を直接決める (L2 側で短文の末尾幻聴抑制に使う)。None なら predictor。
        seconds = req.get("seconds", None)
        sampling = SamplingRequest(
            text=str(req["text"]),
            caption=str(caption),
            ref_wav=str(ref_wav) if ref_wav else None,
            no_ref=(not ref_wav),
            duration_scale=float(req.get("duration_scale", 1.0)),
            num_steps=int(req.get("num_steps", 24)),
            t_schedule_mode=str(req.get("t_schedule_mode", "sway")),
            cfg_scale_speaker=float(req.get("cfg_scale_speaker", 5.0)),
            seed=None if seed is None else int(seed),
            seconds=None if seconds is None else float(seconds),
        )

        t0 = time.perf_counter()
        result = runtime.synthesize(sampling, log_fn=None)
        synth_sec = time.perf_counter() - t0
        wav_bytes, sample_rate = _wav_bytes_from_result(result)
        return wav_bytes, sample_rate, synth_sec


def main() -> None:
    parser = argparse.ArgumentParser(description="Irodori-TTS sidecar HTTP server")
    parser.add_argument("--host", default="127.0.0.1", help="bind host (既定 127.0.0.1 = localhost のみ)")
    parser.add_argument("--port", type=int, default=50080, help="bind port (既定 50080)")
    parser.add_argument(
        "--models",
        default="vd",
        help="ロードするモード (カンマ区切り。確定版は 'vd' のみ)",
    )
    args = parser.parse_args()

    modes = [m.strip().lower() for m in args.models.split(",") if m.strip()]
    unknown = [m for m in modes if m not in _MODE_REPOS]
    if unknown:
        parser.error(f"unknown modes: {unknown}. valid: {sorted(_MODE_REPOS)}")
    if not modes:
        parser.error("at least one mode is required")

    logger.info("HF_HOME=%s", os.environ.get("HF_HOME", "(未設定)"))
    logger.info("L2_IRODORI_REPO=%s", _IRODORI_REPO)
    logger.info("ロード対象モデル: %s", modes)

    runtimes = _load_runtimes(modes)

    server = _IrodoriServer((args.host, args.port), runtimes)
    logger.info("════════════════════════════════════════════════════")
    logger.info(" Irodori-TTS サイドカー起動完了")
    logger.info("   listening : http://%s:%d", args.host, args.port)
    logger.info("   models    : %s", sorted(runtimes))
    logger.info("   health    : GET  /health")
    logger.info("   synth     : POST /synthesize")
    logger.info("════════════════════════════════════════════════════")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("停止要求を受信。シャットダウンします。")
    finally:
        server.shutdown()


if __name__ == "__main__":
    main()
