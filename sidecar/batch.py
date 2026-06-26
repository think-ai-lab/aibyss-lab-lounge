"""
batch.py — irodori サイドカーの「同質ボイス バッチ合成」（upstream 無改変）

責務:
  常駐 InferenceRuntime の公開部品を再利用し、同一ボイス
  （caption / ref / seed / steps / cfg / duration_scale を共有し、text のみ可変）の
  N 発話を 1 回の forward で合成する。メモリ帯域律速の本モデルでは、重みを 1 回読んで
  B 発話を処理することでスループットを上げる狙い（複数レプリカより memory traffic を償却）。

設計（fork 回避）:
  `irodori_tts.inference_runtime.InferenceRuntime.synthesize()` は再利用部品
  （tokenizer.batch_encode / _load_reference_latent / model.encode_conditions /
   predict_duration_log_frames / rf.sample_euler_rf_cfg / codec.decode_latent /
   find_flattening_point / watermarker.encode_batch）の orchestration に過ぎない。
  本モジュールはその orchestration を「texts=リスト・duration は per-item 予測→MAX 長で
  生成→各 item を個別 trim」に組み替えて再現する。vendored irodori_tts は読むだけ。

  ※ synthesize() の該当ロジック（inference_runtime.py:1011-1206）を忠実に転記している。
     upstream の bump で API がずれたら golden test（B=1 が単発とバイト一致）が即検知する。

不変条件:
  - 同質性（caption/ref/seed/steps/cfg/duration_scale/uncond_mode 等が全 req で一致）
  - req.seconds is None（手動 duration はバッチ非対応＝client が per-item に回す）
  - req.lora_adapter is None / req.num_candidates == 1（batch 次元は texts に充てる）
"""
from __future__ import annotations

import math
import secrets

import torch

from irodori_tts.codec import unpatchify_latent
from irodori_tts.duration import build_duration_features
from irodori_tts.inference_runtime import (
    SamplingRequest,
    find_flattening_point,
    resolve_cfg_scales,
)
from irodori_tts.rf import sample_euler_rf_cfg
from irodori_tts.text_normalization import normalize_text

# バッチ内で全 req が一致していなければならないフィールド（声の同一性を担保）。
_HOMOGENEOUS_FIELDS = (
    "caption", "ref_wav", "ref_latent", "ref_embed", "no_ref",
    "ref_normalize_db", "ref_ensure_max", "max_ref_seconds",
    "seed", "num_steps", "t_schedule_mode",
    "cfg_scale_text", "cfg_scale_caption", "cfg_scale_speaker", "cfg_scale",
    "cfg_guidance_mode", "cfg_min_t", "cfg_max_t",
    "duration_scale", "min_seconds", "max_seconds",
    "speaker_uncond_mode", "context_kv_cache",
    "speaker_kv_scale", "speaker_kv_min_t", "speaker_kv_max_layers",
    "truncation_factor", "rescale_k", "rescale_sigma",
    "max_text_len", "max_caption_len",
)


def _require_batchable(reqs: list[SamplingRequest]) -> None:
    r0 = reqs[0]
    for j, r in enumerate(reqs):
        if r.seconds is not None:
            raise ValueError(
                f"synthesize_batch: req[{j}] は手動 seconds 指定。バッチ非対応（per-item /synthesize を使う）。"
            )
        if r.lora_adapter is not None:
            raise ValueError(f"synthesize_batch: req[{j}] は lora_adapter 指定。バッチ非対応。")
        if int(r.num_candidates) != 1:
            raise ValueError(f"synthesize_batch: req[{j}] は num_candidates>1。バッチ非対応。")
        for k in _HOMOGENEOUS_FIELDS:
            if getattr(r, k) != getattr(r0, k):
                raise ValueError(
                    f"synthesize_batch: 非同質バッチ（req[{j}].{k} が req[0] と不一致）。"
                    "同一ボイス・同一パラメータのみバッチ可。"
                )


def synthesize_batch(runtime, reqs: list[SamplingRequest]) -> tuple[list[torch.Tensor], int]:
    """同質ボイスの N リクエストを 1 forward で合成。

    Returns: (audios, sample_rate)。audios は各 (1, samples) の CPU テンソルで、入力 reqs と同順。
    runtime._infer_lock を内部で取得（単発合成と GPU を直列化）。
    """
    sample_rate = int(runtime.codec.sample_rate)
    if not reqs:
        return [], sample_rate
    _require_batchable(reqs)

    r0 = reqs[0]
    n = len(reqs)
    model_cfg = runtime.model_cfg
    device = runtime.model_device

    # --- 共有パラメータ（r0 基準。homogeneity は検証済み）---
    duration_scale = float(r0.duration_scale)
    if duration_scale <= 0:
        raise ValueError(f"duration_scale must be > 0, got {duration_scale}")
    min_seconds = float(r0.min_seconds)
    max_seconds = float(r0.max_seconds)
    used_seed = int(r0.seed) if r0.seed is not None else int(secrets.randbits(63))

    text_max_len = runtime.default_text_max_len if r0.max_text_len is None else int(r0.max_text_len)
    caption_max_len = (
        runtime.default_caption_max_len if r0.max_caption_len is None else int(r0.max_caption_len)
    )
    has_caption_text = bool(
        model_cfg.use_caption_condition
        and r0.caption is not None
        and str(r0.caption).strip() != ""
    )
    use_speaker_for_request = bool(model_cfg.use_speaker_condition_resolved and not r0.no_ref)

    cfg_mode = str(r0.cfg_guidance_mode).strip().lower()
    cfg_scale_text, cfg_scale_caption, cfg_scale_speaker, _ = resolve_cfg_scales(
        cfg_guidance_mode=cfg_mode,
        cfg_scale_text=r0.cfg_scale_text,
        cfg_scale_caption=r0.cfg_scale_caption,
        cfg_scale_speaker=r0.cfg_scale_speaker,
        cfg_scale=r0.cfg_scale,
        use_caption_condition=has_caption_text,
        use_speaker_condition=use_speaker_for_request,
    )

    speaker_kv_scale = None if r0.speaker_kv_scale is None else float(r0.speaker_kv_scale)
    speaker_kv_max_layers = (
        None if r0.speaker_kv_max_layers is None else int(r0.speaker_kv_max_layers)
    )
    speaker_kv_min_t = None
    if speaker_kv_scale is not None and use_speaker_for_request:
        speaker_kv_min_t = 0.9 if r0.speaker_kv_min_t is None else float(r0.speaker_kv_min_t)
    elif speaker_kv_scale is not None:
        speaker_kv_scale = None
    truncation_factor = None if r0.truncation_factor is None else float(r0.truncation_factor)
    rescale_k = None if r0.rescale_k is None else float(r0.rescale_k)
    rescale_sigma = None if r0.rescale_sigma is None else float(r0.rescale_sigma)

    # --- 各 text を正規化（単発 synthesize と同じ）---
    texts: list[str] = []
    for j, r in enumerate(reqs):
        t = normalize_text(str(r.text)).strip()
        if t == "":
            raise ValueError(f"synthesize_batch: req[{j}] の text が正規化後に空。")
        texts.append(t)

    with runtime._infer_lock, torch.inference_mode():
        # テキスト/キャプションの batch encode（list → padding + per-row mask）
        text_ids, text_mask = runtime.tokenizer.batch_encode(texts, max_length=text_max_len)
        text_ids = text_ids.to(device)
        text_mask = text_mask.to(device)

        caption_ids = None
        caption_mask = None
        if model_cfg.use_caption_condition:
            if runtime.caption_tokenizer is None:
                raise RuntimeError("caption conditioning 有効だが caption tokenizer 未ロード。")
            caption_text = "" if r0.caption is None else str(r0.caption).strip()
            caption_ids, caption_mask = runtime.caption_tokenizer.batch_encode(
                [caption_text] * n, max_length=caption_max_len
            )
            if caption_text == "":
                caption_mask.zero_()
            caption_ids = caption_ids.to(device)
            caption_mask = caption_mask.to(device)

        # 参照（声）は共有：_load_reference_latent が 1 本の ref を batch_size 分 repeat する。
        speaker_state_override, speaker_mask_override = runtime._load_speaker_embedding_condition(
            req=r0, batch_size=n, messages=[]
        )
        if speaker_state_override is None:
            ref_latent, ref_mask = runtime._load_reference_latent(
                req=r0, batch_size=n, messages=[]
            )
        else:
            ref_latent, ref_mask = None, None

        hop_length = int(runtime.codec.model.hop_length)
        sr = float(runtime.codec.sample_rate)

        # --- duration: per-item 予測（単発の .mean() を per-item に置換）---
        if not model_cfg.use_duration_predictor:
            raise RuntimeError("synthesize_batch: duration predictor 無しの checkpoint は非対応。")

        has_speaker_duration = torch.zeros((n,), dtype=torch.bool, device=device)
        if speaker_mask_override is not None:
            has_speaker_duration = speaker_mask_override.any(dim=1)
        elif model_cfg.use_speaker_condition_resolved and ref_mask is not None:
            has_speaker_duration = ref_mask.any(dim=1)

        duration_features = build_duration_features(
            texts,
            token_counts=text_mask.sum(dim=1),
            max_text_len=text_max_len,
            has_speaker=has_speaker_duration,
        ).to(device)

        (
            d_text_state,
            d_text_mask,
            d_speaker_state,
            d_speaker_mask,
            d_caption_state,
            d_caption_mask,
        ) = runtime.model.encode_conditions(
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state_override,
            speaker_mask_override=speaker_mask_override,
            speaker_uncond_mode=r0.speaker_uncond_mode,
        )
        pred_log_frames = runtime.model.predict_duration_log_frames(
            text_state=d_text_state,
            text_mask=d_text_mask,
            speaker_state=d_speaker_state,
            speaker_mask=d_speaker_mask,
            caption_state=d_caption_state,
            caption_mask=d_caption_mask,
            duration_features=duration_features,
            has_speaker=has_speaker_duration,
            has_caption=(
                torch.full((n,), has_caption_text, dtype=torch.bool, device=device)
                if model_cfg.use_caption_condition
                else None
            ),
        )
        # per-item フレーム数。単発の `expm1(...).mean().item()` を tolist()+round で per-item 化
        # （B=1 では .mean().item() と同一値 → golden test がバイト一致になる）。
        min_frames = max(1, math.ceil(min_seconds * sr / hop_length))
        max_frames = max(1, math.floor(max_seconds * sr / hop_length))
        pred_frames_per = torch.expm1(pred_log_frames).float().tolist()  # list[float], len n
        latent_steps_per: list[int] = []
        for pf in pred_frames_per:
            ls = int(round(pf * duration_scale))
            ls = max(min_frames, min(max_frames, ls))
            latent_steps_per.append(ls)
        latent_steps = max(latent_steps_per)  # バッチ共有の生成長（最長に合わせ、各々を後で trim）
        target_samples_per = [ls * hop_length for ls in latent_steps_per]
        patched_steps = math.ceil(latent_steps / model_cfg.latent_patch_size)

        # --- 拡散サンプリング（1 forward で N row）---
        z_patched = sample_euler_rf_cfg(
            model=runtime.model,
            text_input_ids=text_ids,
            text_mask=text_mask,
            ref_latent=ref_latent,
            ref_mask=ref_mask,
            sequence_length=patched_steps,
            caption_input_ids=caption_ids,
            caption_mask=caption_mask,
            speaker_state_override=speaker_state_override,
            speaker_mask_override=speaker_mask_override,
            speaker_uncond_mode=r0.speaker_uncond_mode,
            num_steps=int(r0.num_steps),
            cfg_scale_text=cfg_scale_text,
            cfg_scale_caption=cfg_scale_caption,
            cfg_scale_speaker=cfg_scale_speaker,
            cfg_guidance_mode=cfg_mode,
            cfg_min_t=float(r0.cfg_min_t),
            cfg_max_t=float(r0.cfg_max_t),
            seed=used_seed,
            truncation_factor=truncation_factor,
            rescale_k=rescale_k,
            rescale_sigma=rescale_sigma,
            use_context_kv_cache=bool(r0.context_kv_cache),
            speaker_kv_scale=speaker_kv_scale,
            speaker_kv_max_layers=speaker_kv_max_layers,
            speaker_kv_min_t=speaker_kv_min_t,
            t_schedule_mode=str(r0.t_schedule_mode),
            sway_coeff=float(r0.sway_coeff),
        )

        z = unpatchify_latent(
            z_patched,
            patch_size=model_cfg.latent_patch_size,
            latent_dim=model_cfg.latent_dim,
        )
        z = z[:, :latent_steps]

        # --- 一括デコード → per-item trim（各 item を自分の予測長＋無音検出で切る）---
        audio_batch = runtime.codec.decode_latent(z).cpu()
        trimmed: list[torch.Tensor] = []
        for i, r in enumerate(reqs):
            audio_i = audio_batch[i]
            max_samples = target_samples_per[i]
            if bool(r.trim_tail):
                flattening_point = find_flattening_point(
                    z[i],
                    window_size=max(1, int(r.tail_window_size)),
                    std_threshold=float(r.tail_std_threshold),
                    mean_threshold=float(r.tail_mean_threshold),
                )
                flattening_samples = int(flattening_point * hop_length)
                if flattening_samples > 0:
                    max_samples = min(max_samples, flattening_samples)
            trimmed.append(audio_i[:, :max_samples])

        if runtime.watermarker.ready:
            trimmed = runtime.watermarker.encode_batch(trimmed, sample_rate=sample_rate)

    return trimmed, sample_rate
