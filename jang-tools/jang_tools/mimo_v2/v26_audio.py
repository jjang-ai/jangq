"""MiMo-V2.6 audio INPUT pipeline in MLX (encoder side only).

Ported from Xiaomi's reference ``modeling_mimo_v2.py`` shipped with
``XiaomiMiMo/MiMo-V2.6-Flash-RL`` (classes ``AudioTokenizerEncoder``,
``ResidualVectorQuantizer``, ``tokenize_audio_batch``, ``MiMoAudioEncoder``,
``AudioProjection``, ``_build_speech_embeddings``, ``_pad_and_group_audio_codes``)
and from the serving front-ends that own the waveform -> mel step (the HF
modeling file takes mels as input and has no frontend):

* SGLang ``python/sglang/srt/multimodal/processors/mimo_audio.py``
  (``MiMoAudioPipeline.preprocess_audio`` / ``compute_audio_token_len``)
* vLLM ``vllm/transformers_utils/processors/mimo_v2_omni.py`` (identical mel
  kwargs) and ``vllm/model_executor/models/mimo_audio.py``.

It deliberately does NOT reuse any MiMo-V2.5 code in this package.

Pipeline::

    wav (any sr, mono or [C, T]) --torchaudio-equivalent sinc resample--> 24 kHz
      -> log-mel  (torchaudio MelSpectrogram: n_fft 960, win 960 periodic Hann,
                   hop 240, center=True reflect pad, power 1.0 (magnitude),
                   HTK mel, norm=None, f 0..12000, 128 mels; log(clip(x, 1e-7)))
      -> [T_mel, 128], T_mel = 1 + N // 240          (100 frames / s)
      -> split into 6000-frame segments (60 s) + remainder, batched <=256000 frames
      -> AudioTokenizerEncoder: conv1(k3,p1)+GELU, conv2(k3,s2,p1)+GELU  (50 Hz)
         24 pre-LN layers, causal; even layers SWA |i-j|<=128, odd layers full;
         RoPE theta 1e4 per segment; skip = output of layer 3 added before final LN
         avg_pooler: Conv1d(k2,s2,no bias)+GELU, LayerNorm   (25 Hz)
      -> 20-stage residual VQ (fp32, codebook rounded to model dtype) -> codes [T, 20]
      -> pad T to multiple of 4 by repeating the last row -> [G, 4, 20]
      -> sum of 20 speech_embeddings (vocab 1280, dim 1024) -> [G, 4, 1024]
      -> input_local_transformer (Qwen2, 6 layers, 16x64 heads, rope 640000,
         bidirectional inside the group, final RMSNorm)
      -> reshape [G, 4096] -> AudioProjection 4096->16384 GELU ->4096 (no bias)
      -> one backbone embedding per group, placed at <|audio_pad|> tokens.

Prompt: ``<|mimo_audio_start|>`` + n_tokens x ``<|audio_pad|>`` + ``<|mimo_audio_end|>``
with n_tokens = ceil(ceil(ceil(T_mel/2)/2)/4) = ceil(T_mel/16).
"""

from __future__ import annotations

import json
import math
import os
import struct
from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

import mlx.core as mx
import mlx.nn as nn

AUDIO_TOKEN_ID = 151669  # <|audio_pad|>
AUDIO_START_TOKEN_ID = 151673  # <|mimo_audio_start|>
AUDIO_END_TOKEN_ID = 151674  # <|mimo_audio_end|>
AUDIO_EOD_TOKEN = "<|mimo_audio_eod|>"  # exists in the vocab; unused for audio INPUT by vLLM/SGLang

SEGMENT_SIZE = 6000  # mel frames (= 60 s), config.audio_config.audio_segment_size
MAX_BATCH_FRAMES = 256000  # _at_group_by_length(max_length) in tokenize_audio_batch


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------


@dataclass
class AudioTokenizerConfig:
    d_model: int = 1024
    encoder_layers: int = 24
    encoder_attention_heads: int = 16
    encoder_ffn_dim: int = 4096
    encoder_skip_layer_id: Optional[int] = 3
    encoder_causal: bool = True
    encoder_attn_window_size: tuple = (128, 0)
    hybrid_attention: bool = True
    hybrid_block_size: int = 8  # present in config, NOT used by the reference encoder
    swa_per_block: int = 2
    kernel_size: int = 3
    stride_size: int = 2
    avg_pooler: int = 2
    rope_theta: float = 10000.0
    n_mels: int = 128
    nfft: int = 960
    hop_length: int = 240
    window_size: int = 960
    sampling_rate: int = 24000
    fmin: float = 0.0
    fmax: Optional[float] = None
    num_quantizers: int = 20
    codebook_size: list = field(default_factory=lambda: [1024, 1024, 256] + [128] * 17)
    ln_type: str = "LayerNorm"
    activation_function: str = "gelu"

    @classmethod
    def from_json(cls, path: str) -> "AudioTokenizerConfig":
        with open(path) as f:
            d = json.load(f)
        known = cls.__dataclass_fields__.keys()
        kw = {k: v for k, v in d.items() if k in known}
        if "encoder_attn_window_size" in kw:
            kw["encoder_attn_window_size"] = tuple(kw["encoder_attn_window_size"])
        cfg = cls(**kw)
        if cfg.ln_type != "LayerNorm" or cfg.activation_function != "gelu":
            raise ValueError(f"unsupported ln_type/activation: {cfg.ln_type}/{cfg.activation_function}")
        cs = list(cfg.codebook_size)
        if len(cs) < cfg.num_quantizers:
            cs += [cs[-1]] * (cfg.num_quantizers - len(cs))
        cfg.codebook_size = cs
        return cfg

    def layer_window(self, i: int) -> int:
        """Left window for layer i (-1 = full). Mirrors AudioTokenizerEncoder.__init__."""
        if self.hybrid_attention:
            if i % self.swa_per_block < self.swa_per_block - 1:
                return int(self.encoder_attn_window_size[0])
            return -1
        return int(self.encoder_attn_window_size[0])

    def conv_out_len(self, mel_len: int) -> int:
        tgt = mel_len + 3 - self.kernel_size
        return (tgt + 2 - self.kernel_size) // self.stride_size + 1

    def code_len(self, mel_len: int) -> int:
        n = self.conv_out_len(mel_len)
        if self.avg_pooler != 1:
            n = n // self.avg_pooler + int(n % self.avg_pooler != 0)
        return n


def _parse_maybe_list(value, length: int) -> list[int]:
    if isinstance(value, str) and "-" in value:
        return [int(x) for x in value.split("-")]
    return [int(value)] * length


@dataclass
class AudioEncoderConfig:
    audio_channels: int = 20
    group_size: int = 4
    input_local_dim: int = 1024
    input_local_layers: int = 6
    input_local_attn_heads: int = 16
    input_local_head_dim: int = 64
    input_local_intermediate_size: int = 4096
    input_full_attention: bool = True
    out_hidden_size: int = 4096
    rope_theta: float = 640000.0
    partial_rotary_factor: float = 1.0
    projection_layers: int = 2
    add_post_norm: bool = True
    audio_segment_size: int = SEGMENT_SIZE
    rms_norm_eps: float = 1e-6  # Qwen2Config default (not overridden by MiMoAudioEncoder)
    speech_vocab_size: list = field(default_factory=lambda: [1280] * 20)
    speech_zeroemb_idx: list = field(default_factory=lambda: [1024] * 20)

    @classmethod
    def from_model_config(cls, path: str) -> "AudioEncoderConfig":
        with open(path) as f:
            d = json.load(f)["audio_config"]
        known = cls.__dataclass_fields__.keys()
        kw = {k: v for k, v in d.items() if k in known and k not in ("speech_vocab_size", "speech_zeroemb_idx")}
        cfg = cls(**kw)
        cfg.speech_vocab_size = _parse_maybe_list(d["speech_vocab_size"], cfg.audio_channels)
        cfg.speech_zeroemb_idx = _parse_maybe_list(d["speech_zeroemb_idx"], cfg.audio_channels)
        if cfg.partial_rotary_factor != 1.0:
            raise ValueError("partial_rotary_factor != 1.0 not implemented")
        if cfg.input_local_dim // cfg.input_local_attn_heads != cfg.input_local_head_dim:
            # Qwen2Config ignores input_local_head_dim and uses hidden/heads.
            raise ValueError("input_local_head_dim disagrees with hidden/heads")
        if cfg.projection_layers not in (1, 2):
            raise ValueError(f"projection_layers={cfg.projection_layers}")
        return cfg


# ---------------------------------------------------------------------------
# Frontend: resampling + log-mel (numpy, float64 internally)
# ---------------------------------------------------------------------------


def _sinc_resample_kernel(orig: int, new: int, lowpass_filter_width: int = 6, rolloff: float = 0.99):
    """torchaudio.functional._get_sinc_resample_kernel (sinc_interp_hann, dtype=None path)."""
    base = min(orig, new) * rolloff
    width = math.ceil(lowpass_filter_width * orig / base)
    idx = np.arange(-width, width + orig, dtype=np.float64)[None, :] / orig
    # torch.arange(0, -new, -1, dtype=None) is float32 and divided in float32
    t0 = (np.arange(0, -new, -1).astype(np.float32) / np.float32(new)).astype(np.float64)
    t = (t0[:, None] + idx) * base
    t = np.clip(t, -lowpass_filter_width, lowpass_filter_width)
    window = np.cos(t * math.pi / lowpass_filter_width / 2) ** 2
    t = t * math.pi
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.where(t == 0, 1.0, np.sin(t) / t)
    k = k * window * (base / orig)
    return k.astype(np.float32), width  # Resample transform caches the kernel as float32


def resample(wav: np.ndarray, orig_sr: int, new_sr: int = 24000) -> np.ndarray:
    """Bit-compatible port of torchaudio.transforms.Resample(orig_sr, new_sr) (default args)."""
    wav = np.asarray(wav, dtype=np.float32)
    if int(orig_sr) == int(new_sr):
        return wav
    if int(orig_sr) != orig_sr or int(new_sr) != new_sr:
        raise ValueError("integer sample rates required")
    g = math.gcd(int(orig_sr), int(new_sr))
    orig, new = int(orig_sr) // g, int(new_sr) // g
    kernel, width = _sinc_resample_kernel(orig, new)
    shape = wav.shape
    x = wav.reshape(-1, shape[-1]).astype(np.float64)
    length = x.shape[1]
    xp = np.pad(x, ((0, 0), (width, width + orig)))
    n_out = (xp.shape[1] - kernel.shape[1]) // orig + 1
    # frames[b, n, k] = xp[b, n*orig + k]
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(xp.shape[0], n_out, kernel.shape[1]),
        strides=(xp.strides[0], xp.strides[1] * orig, xp.strides[1]),
    )
    out = np.einsum("bnk,jk->bnj", frames, kernel.astype(np.float64))  # [B, n_out, new]
    out = out.reshape(x.shape[0], -1)
    target = int(math.ceil(new * length / orig))
    out = out[:, :target].astype(np.float32)
    return out.reshape(shape[:-1] + (out.shape[-1],))


def _hz_to_mel_htk(f):
    return 2595.0 * np.log10(1.0 + np.asarray(f, dtype=np.float64) / 700.0)


def _mel_to_hz_htk(m):
    return 700.0 * (10.0 ** (np.asarray(m, dtype=np.float64) / 2595.0) - 1.0)


def mel_filterbank(sr: int = 24000, n_fft: int = 960, n_mels: int = 128, f_min: float = 0.0,
                   f_max: Optional[float] = None) -> np.ndarray:
    """torchaudio.functional.melscale_fbanks(norm=None, mel_scale='htk'); returns [n_freqs, n_mels]."""
    f_max = float(sr // 2) if f_max is None else float(f_max)
    n_freqs = n_fft // 2 + 1
    all_freqs = np.linspace(0, sr // 2, n_freqs)
    m_pts = np.linspace(_hz_to_mel_htk(f_min), _hz_to_mel_htk(f_max), n_mels + 2)
    f_pts = _mel_to_hz_htk(m_pts)
    f_diff = f_pts[1:] - f_pts[:-1]
    slopes = f_pts[None, :] - all_freqs[:, None]
    down = -slopes[:, :-2] / f_diff[:-1]
    up = slopes[:, 2:] / f_diff[1:]
    return np.maximum(0.0, np.minimum(down, up))


def num_mel_frames(n_samples: int, hop_length: int = 240) -> int:
    """center=True STFT frame count."""
    return 1 + int(n_samples) // hop_length


def log_mel(wav24k: np.ndarray, cfg: Optional[AudioTokenizerConfig] = None) -> np.ndarray:
    """Waveform at 24 kHz (1-D) -> log-mel [T_mel, n_mels] float32.

    Equivalent to ``log(clip(MelSpectrogram(sr, n_fft, hop, win, f_min, f_max,
    n_mels, power=1.0, center=True)(wav), 1e-7)).T`` as used by SGLang/vLLM.
    """
    cfg = cfg or AudioTokenizerConfig()
    x = np.asarray(wav24k, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError("log_mel expects a mono 1-D waveform")
    n_fft, hop, win = cfg.nfft, cfg.hop_length, cfg.window_size
    pad = n_fft // 2
    if x.shape[0] <= pad:
        raise ValueError(f"audio too short for reflect padding ({x.shape[0]} <= {pad} samples)")
    xp = np.pad(x, (pad, pad), mode="reflect")
    n_frames = 1 + (xp.shape[0] - n_fft) // hop
    frames = np.lib.stride_tricks.as_strided(
        xp, shape=(n_frames, n_fft), strides=(xp.strides[0] * hop, xp.strides[0])
    )
    k = np.arange(win, dtype=np.float64)
    window = 0.5 - 0.5 * np.cos(2.0 * math.pi * k / win)  # torch.hann_window(periodic=True)
    if win < n_fft:  # torch.stft centers a shorter window inside n_fft
        left = (n_fft - win) // 2
        window = np.pad(window, (left, n_fft - win - left))
    spec = np.abs(np.fft.rfft(frames * window[None, :], n=n_fft, axis=-1))  # power=1.0
    fb = mel_filterbank(cfg.sampling_rate, n_fft, cfg.n_mels, cfg.fmin, cfg.fmax)
    mel = spec @ fb
    return np.log(np.maximum(mel, 1e-7)).astype(np.float32)


def to_mono_24k(wav: np.ndarray, sr: int, target_sr: int = 24000) -> np.ndarray:
    """Match SGLang/vLLM order: resample every channel first, then average channels.

    ``wav`` is [T] or channels-first [C, T] (torchcodec/torchaudio layout).
    """
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim not in (1, 2):
        raise ValueError(f"wav must be [T] or [C, T], got {wav.shape}")
    if int(sr) != target_sr:
        wav = resample(wav, int(sr), target_sr)
    if wav.ndim == 2:
        wav = wav.mean(axis=0, dtype=np.float32)
    return wav


# ---------------------------------------------------------------------------
# Token-count formula
# ---------------------------------------------------------------------------


def audio_token_count_from_mel(mel_len: int, kernel_size: int = 3, stride: int = 2, avg_pooler: int = 2,
                               group_size: int = 4) -> int:
    """SGLang MiMoAudioPipeline.compute_audio_token_len (== vLLM item_token_lens)."""
    n = mel_len + 3 - kernel_size
    n = (n + 2 - kernel_size) // stride + 1
    n = n // avg_pooler + int(n % avg_pooler != 0)
    return math.ceil(n / group_size)


def audio_token_count(n_samples_24k: int) -> int:
    """Number of <|audio_pad|> tokens for N samples at 24 kHz: ceil((1 + N//240) / 16)."""
    return audio_token_count_from_mel(num_mel_frames(n_samples_24k))


def build_audio_prompt_ids(n_tokens: int) -> list[int]:
    return [AUDIO_START_TOKEN_ID] + [AUDIO_TOKEN_ID] * int(n_tokens) + [AUDIO_END_TOKEN_ID]


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def _gelu(x: mx.array) -> mx.array:
    """Exact (erf) GELU evaluated in fp32, result in x.dtype -- what torch does for bf16
    (opmath). Evaluating it in bf16 is catastrophic here: AudioProjection's fc1 output is
    overwhelmingly large-negative, so GELU lives in its tail where x*(1+erf(x/sqrt2))/2
    cancels; bf16 GELU alone moved the final embeddings by ~8.9% rel vs fp32."""
    return nn.gelu(x.astype(mx.float32)).astype(x.dtype)


def _silu(x: mx.array) -> mx.array:
    return nn.silu(x.astype(mx.float32)).astype(x.dtype)


def _rotate_half(x: mx.array) -> mx.array:
    h = x.shape[-1] // 2
    return mx.concatenate([-x[..., h:], x[..., :h]], axis=-1)


def _rope_cos_sin(positions: int, dim: int, theta: float, dtype, inv_freq_dtype=None) -> tuple[mx.array, mx.array]:
    """HF-style non-interleaved rope; computed in fp32 then cast to the activation dtype.

    ``inv_freq_dtype``: dtype the ``inv_freq`` buffer was rounded to before the fp32
    matmul. The reference tokenizer loader calls ``model.to(bfloat16)``, which rounds
    this non-persistent buffer too (HF ``load_audio_tokenizer`` and vLLM
    ``_load_audio_tokenizer`` both do it), so tokenizer rope angles use a
    bf16-rounded inv_freq in production.
    """
    inv = 1.0 / (theta ** (np.arange(0, dim, 2, dtype=np.int64).astype(np.float32) / np.float32(dim)))
    inv = mx.array(inv.astype(np.float32))
    if inv_freq_dtype is not None:
        inv = inv.astype(inv_freq_dtype).astype(mx.float32)
    pos = mx.arange(positions, dtype=mx.float32)
    freqs = pos[:, None] * inv[None, :]
    emb = mx.concatenate([freqs, freqs], axis=-1)
    return mx.cos(emb).astype(dtype), mx.sin(emb).astype(dtype)


# ---------------------------------------------------------------------------
# Audio tokenizer encoder (MLX)
# ---------------------------------------------------------------------------


class _ATAttention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads = heads
        self.head_dim = d // heads
        self.scale = self.head_dim ** -0.5
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.out_proj = nn.Linear(d, d, bias=True)

    def __call__(self, x, cos, sin, mask):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.heads, self.head_dim)
        k = self.k_proj(x).reshape(B, L, self.heads, self.head_dim)
        v = self.v_proj(x).reshape(B, L, self.heads, self.head_dim)
        c, s = cos[None, :, None, :], sin[None, :, None, :]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        return self.out_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))


class _ATLayer(nn.Module):
    def __init__(self, cfg: AudioTokenizerConfig, window: int):
        super().__init__()
        d = cfg.d_model
        self.window = window
        self.self_attn = _ATAttention(d, cfg.encoder_attention_heads)
        self.self_attn_layer_norm = nn.LayerNorm(d, eps=1e-5)
        self.fc1 = nn.Linear(d, cfg.encoder_ffn_dim)
        self.fc2 = nn.Linear(cfg.encoder_ffn_dim, d)
        self.final_layer_norm = nn.LayerNorm(d, eps=1e-5)

    def __call__(self, x, cos, sin, mask):
        x = x + self.self_attn(self.self_attn_layer_norm(x), cos, sin, mask)
        return x + self.fc2(_gelu(self.fc1(self.final_layer_norm(x))))


class _DownSample(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        self.conv = nn.Conv1d(d, d, kernel_size=k, stride=k, bias=False)

    def __call__(self, x):
        return _gelu(self.conv(x))


class AudioTokenizerEncoderMLX(nn.Module):
    """AudioTokenizerEncoder + ResidualVectorQuantizer (encode only)."""

    def __init__(self, cfg: AudioTokenizerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.conv1 = nn.Conv1d(cfg.n_mels, d, kernel_size=cfg.kernel_size, padding=1)
        self.conv2 = nn.Conv1d(d, d, kernel_size=cfg.kernel_size, stride=cfg.stride_size, padding=1)
        self.layers = [_ATLayer(cfg, cfg.layer_window(i)) for i in range(cfg.encoder_layers)]
        self.layer_norm = nn.LayerNorm(d, eps=1e-5)
        self.down_sample_layer = _DownSample(d, cfg.avg_pooler) if cfg.avg_pooler != 1 else None
        self.down_sample_norm = nn.LayerNorm(d, eps=1e-5) if cfg.avg_pooler != 1 else None
        # fp32 codebooks, set by the loader (rounded to model dtype for reference parity)
        self.codebooks: list[mx.array] = [mx.zeros((n, d), dtype=mx.float32) for n in cfg.codebook_size]

    @property
    def dtype(self):
        return self.conv1.weight.dtype

    def _mask(self, L: int, window: int):
        if not self.cfg.encoder_causal and window <= 0:
            return None
        if window <= 0:
            return "causal"
        i = mx.arange(L)[:, None]
        j = mx.arange(L)[None, :]
        keep = mx.abs(i - j) <= window
        if self.cfg.encoder_causal:
            keep = keep & (j <= i)
        return keep

    def _transformer(self, h: mx.array) -> mx.array:
        """h: [n, L, D] -- n segments of identical length L, positions reset per segment."""
        L = h.shape[1]
        cos, sin = _rope_cos_sin(L, self.cfg.d_model // self.cfg.encoder_attention_heads,
                                 self.cfg.rope_theta, h.dtype, inv_freq_dtype=h.dtype)
        skip = None
        for idx, layer in enumerate(self.layers):
            h = layer(h, cos, sin, self._mask(L, layer.window))
            if self.cfg.encoder_skip_layer_id is not None and idx == self.cfg.encoder_skip_layer_id - 1:
                skip = h
        if skip is not None:
            h = h + skip
        return self.layer_norm(h)

    def features(self, seg_mels: Sequence[np.ndarray]) -> list[mx.array]:
        """One reference ``encode`` call on a batch of segments. Returns packed hidden per segment.

        Reproduces the reference's *batched* semantics exactly: segments are
        zero-padded to the longest one before conv1/conv2 (so an odd-length
        segment shorter than the batch max sees GELU(conv1(pad)) at its conv2
        edge), and before the avg-pooler each segment is padded with copies of its
        own last frame (``unpacking_index``) up to the batch length, then one zero
        frame if that length is odd.
        """
        cfg = self.cfg
        lens = [int(m.shape[0]) for m in seg_mels]
        Lmax = max(lens)
        x = np.zeros((len(seg_mels), Lmax, cfg.n_mels), dtype=np.float32)
        for b, m in enumerate(seg_mels):
            x[b, : lens[b]] = m
        h = mx.array(x).astype(self.dtype)
        h = _gelu(self.conv1(h))
        h = _gelu(self.conv2(h))  # [B, T, D]
        T = h.shape[1]
        out_lens = [cfg.conv_out_len(l) for l in lens]

        # transformer per segment; batch segments sharing a length
        packed: list[Optional[mx.array]] = [None] * len(lens)
        by_len: dict[int, list[int]] = {}
        for b, ol in enumerate(out_lens):
            by_len.setdefault(ol, []).append(b)
        for ol, bs in by_len.items():
            hb = mx.stack([h[b, :ol] for b in bs]) if len(bs) > 1 else h[bs[0], :ol][None]
            hb = self._transformer(hb)
            for i, b in enumerate(bs):
                packed[b] = hb[i]

        if self.down_sample_layer is None:
            return packed
        k = cfg.avg_pooler
        rows = []
        for b, hb in enumerate(packed):
            ol = out_lens[b]
            if ol < T:
                hb = mx.concatenate([hb, mx.repeat(hb[ol - 1: ol], T - ol, axis=0)], axis=0)
            rows.append(hb)
        hp = mx.stack(rows)  # [B, T, D]
        if T % k:
            hp = mx.concatenate([hp, mx.zeros((hp.shape[0], k - T % k, hp.shape[2]), dtype=hp.dtype)], axis=1)
        hp = self.down_sample_layer(hp)
        outs = []
        for b, ol in enumerate(out_lens):
            pl = ol // k + int(ol % k != 0)
            outs.append(self.down_sample_norm(hp[b, :pl]))
        return outs

    def quantize(self, hidden: mx.array, n_q: Optional[int] = None) -> mx.array:
        """Residual VQ in fp32. hidden [T, D] -> codes [T, n_q] int32."""
        x = hidden.astype(mx.float32)
        residual = x
        codes = []
        for E in self.codebooks[: (n_q or len(self.codebooks))]:
            # same expression/order as EuclideanCodebook.quantize
            dist = -(mx.sum(residual * residual, axis=1, keepdims=True)
                     - 2 * (residual @ E.T)
                     + mx.sum(E * E, axis=1)[None, :])
            idx = mx.argmax(dist, axis=-1)
            residual = residual - E[idx]
            codes.append(idx)
        return mx.stack(codes, axis=1).astype(mx.int32)

    def tokenize(self, mels: Sequence[np.ndarray], segment_size: int = SEGMENT_SIZE,
                 max_batch_frames: int = MAX_BATCH_FRAMES) -> list[mx.array]:
        """``tokenize_audio_batch``: list of [T_mel, n_mels] -> list of codes [T_codes, 20]."""
        seg_mels, seg_counts = [], []
        for m in mels:
            m = np.asarray(m, dtype=np.float32)
            L = m.shape[0]
            segs = [segment_size] * (L // segment_size)
            if L % segment_size:
                segs.append(L % segment_size)
            off = 0
            for s in segs:
                seg_mels.append(m[off: off + s])
                off += s
            seg_counts.append(len(segs))
        # _at_group_by_length
        groups, cur, cur_sum = [], [], 0
        for sm in seg_mels:
            l = sm.shape[0]
            if cur_sum + l > max_batch_frames and cur_sum > 0:
                groups.append(cur)
                cur, cur_sum = [sm], l
            else:
                cur.append(sm)
                cur_sum += l
        if cur:
            groups.append(cur)
        seg_codes = []
        for g in groups:
            hs = self.features(g)
            codes = self.quantize(mx.concatenate(hs, axis=0))
            mx.eval(codes)
            seg_codes.append(codes)
        codes = mx.concatenate(seg_codes, axis=0)
        out, off, si = [], 0, 0
        for n in seg_counts:
            t = sum(self.cfg.code_len(sm.shape[0]) for sm in seg_mels[si: si + n])
            out.append(codes[off: off + t])
            off += t
            si += n
        return out


# ---------------------------------------------------------------------------
# MiMoAudioEncoder (speech embeddings + input local transformer + projection)
# ---------------------------------------------------------------------------


class _QAttn(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.heads, self.hd = heads, d // heads
        self.scale = self.hd ** -0.5
        self.q_proj = nn.Linear(d, d, bias=True)
        self.k_proj = nn.Linear(d, d, bias=True)
        self.v_proj = nn.Linear(d, d, bias=True)
        self.o_proj = nn.Linear(d, d, bias=False)

    def __call__(self, x, cos, sin, causal: bool):
        B, L, D = x.shape
        q = self.q_proj(x).reshape(B, L, self.heads, self.hd).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, L, self.heads, self.hd).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, L, self.heads, self.hd).transpose(0, 2, 1, 3)
        c, s = cos[None, None], sin[None, None]
        q = q * c + _rotate_half(q) * s
        k = k * c + _rotate_half(k) * s
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask="causal" if causal else None)
        return self.o_proj(o.transpose(0, 2, 1, 3).reshape(B, L, D))


class _QMLP(nn.Module):
    def __init__(self, d: int, f: int):
        super().__init__()
        self.gate_proj = nn.Linear(d, f, bias=False)
        self.up_proj = nn.Linear(d, f, bias=False)
        self.down_proj = nn.Linear(f, d, bias=False)

    def __call__(self, x):
        return self.down_proj(_silu(self.gate_proj(x)) * self.up_proj(x))


class _QLayer(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        d = cfg.input_local_dim
        self.self_attn = _QAttn(d, cfg.input_local_attn_heads)
        self.mlp = _QMLP(d, cfg.input_local_intermediate_size)
        self.input_layernorm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(d, eps=cfg.rms_norm_eps)

    def __call__(self, x, cos, sin, causal):
        x = x + self.self_attn(self.input_layernorm(x), cos, sin, causal)
        return x + self.mlp(self.post_attention_layernorm(x))


class _InputLocalTransformer(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.layers = [_QLayer(cfg) for _ in range(cfg.input_local_layers)]
        self.norm = nn.RMSNorm(cfg.input_local_dim, eps=cfg.rms_norm_eps) if cfg.add_post_norm else None


class _AudioProjection(nn.Module):
    def __init__(self, i: int, h: int, o: int):
        super().__init__()
        self.fc1 = nn.Linear(i, h, bias=False)
        self.fc2 = nn.Linear(h, o, bias=False)

    def __call__(self, x):
        return self.fc2(_gelu(self.fc1(x)))


class MiMoAudioEncoderMLX(nn.Module):
    def __init__(self, cfg: AudioEncoderConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.input_local_dim
        self.speech_embeddings = [nn.Embedding(cfg.speech_vocab_size[i], d) for i in range(cfg.audio_channels)]
        self.input_local_transformer = _InputLocalTransformer(cfg)
        pin = d * cfg.group_size
        if cfg.projection_layers == 2:
            self.projection = _AudioProjection(pin, pin * 4, cfg.out_hidden_size)
        else:
            self.projection = nn.Linear(pin, cfg.out_hidden_size, bias=False)

    def group_codes(self, codes) -> mx.array:
        """_pad_and_group_audio_codes: [T, C] -> [G, group, C], padding by repeating the last row."""
        codes = mx.array(np.asarray(codes)) if not isinstance(codes, mx.array) else codes
        codes = codes[:, : self.cfg.audio_channels].astype(mx.int32)
        T, g = codes.shape[0], self.cfg.group_size
        pT = ((T + g - 1) // g) * g
        if pT > T:
            codes = mx.concatenate([codes, mx.repeat(codes[-1:], pT - T, axis=0)], axis=0)
        return codes.reshape(pT // g, g, self.cfg.audio_channels)

    def embed_grouped(self, grouped: mx.array) -> mx.array:
        """[G, group, C] int -> [G, out_hidden_size]."""
        dtype = self.speech_embeddings[0].weight.dtype
        x = mx.zeros((grouped.shape[0], grouped.shape[1], self.cfg.input_local_dim), dtype=dtype)
        for i in range(self.cfg.audio_channels):
            x = x + self.speech_embeddings[i](grouped[:, :, i])
        cos, sin = _rope_cos_sin(grouped.shape[1], self.cfg.input_local_dim // self.cfg.input_local_attn_heads,
                                 self.cfg.rope_theta, x.dtype)
        causal = not self.cfg.input_full_attention
        for layer in self.input_local_transformer.layers:
            x = layer(x, cos, sin, causal)
        if self.input_local_transformer.norm is not None:
            x = self.input_local_transformer.norm(x)
        return self.projection(x.reshape(x.shape[0], -1))

    def __call__(self, codes) -> mx.array:
        """codes [T, C] for ONE clip -> [ceil(T/4), out_hidden_size]."""
        return self.embed_grouped(self.group_codes(codes))


# ---------------------------------------------------------------------------
# Weight loading (reads only the needed tensors; bf16-safe)
# ---------------------------------------------------------------------------

_ST_DTYPES = {"BF16": (np.uint16, mx.bfloat16), "F16": (np.float16, None), "F32": (np.float32, None),
              "I64": (np.int64, None), "I32": (np.int32, None)}


def _read_safetensors(path: str, want) -> dict[str, mx.array]:
    """Read selected tensors from a safetensors file by byte range. ``want(name) -> bool``."""
    out = {}
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        for name, meta in header.items():
            if name == "__metadata__" or not want(name):
                continue
            np_dt, mx_view = _ST_DTYPES[meta["dtype"]]
            s, e = meta["data_offsets"]
            f.seek(base + s)
            buf = np.frombuffer(f.read(e - s), dtype=np_dt).reshape(meta["shape"])
            a = mx.array(buf)
            if mx_view is not None:
                a = a.view(mx_view)
            out[name] = a
    return out


def load_audio_tokenizer_encoder(src_dir: str, dtype=mx.bfloat16,
                                 codebook_round_to_dtype: bool = True) -> AudioTokenizerEncoderMLX:
    """Load ``<src>/audio_tokenizer`` encoder + quantizer.

    The reference ``load_audio_tokenizer`` casts the whole tokenizer (including the
    fp32 ``_codebook.embed`` buffers) to ``dtype`` and ``encode`` then calls
    ``quantizer.float()`` -- so the official codebooks are ``dtype``-rounded fp32.
    ``codebook_round_to_dtype`` reproduces that (a no-op for fp32).
    """
    tdir = os.path.join(src_dir, "audio_tokenizer")
    cfg = AudioTokenizerConfig.from_json(os.path.join(tdir, "config.json"))
    raw = _read_safetensors(os.path.join(tdir, "model.safetensors"), lambda k: k.startswith("encoder."))
    model = AudioTokenizerEncoderMLX(cfg)
    weights, codebooks = [], {}
    for k, v in raw.items():
        k = k[len("encoder."):]
        if k.startswith("quantizer."):
            if k.endswith("._codebook.embed"):
                i = int(k.split(".")[3])
                cb = v.astype(dtype).astype(mx.float32) if codebook_round_to_dtype else v.astype(mx.float32)
                codebooks[i] = cb
            continue  # cluster_size / embed_avg / inited are training-only buffers
        if k in ("conv1.weight", "conv2.weight"):
            v = v.transpose(0, 2, 1)  # torch [O, I, K] -> mlx [O, K, I]
        elif k == "down_sample_layer.0.weight":
            k, v = "down_sample_layer.conv.weight", v.transpose(0, 2, 1)
        weights.append((k, v.astype(dtype)))
    if sorted(codebooks) != list(range(cfg.num_quantizers)):
        raise ValueError(f"missing codebooks: have {sorted(codebooks)}")
    _strict_load(model, weights, skip=("codebooks",))
    model.codebooks = [codebooks[i] for i in range(cfg.num_quantizers)]
    for i, cb in enumerate(model.codebooks):
        if cb.shape[0] != cfg.codebook_size[i]:
            raise ValueError(f"codebook {i}: {cb.shape} vs config {cfg.codebook_size[i]}")
    mx.eval(model.parameters(), model.codebooks)
    return model


def _strict_load(model: nn.Module, weights: list, skip: tuple = ()) -> None:
    from mlx.utils import tree_flatten

    have = {k for k, _ in tree_flatten(model.parameters()) if not k.startswith(skip)}
    got = {k for k, _ in weights}
    missing, extra = have - got, got - have
    if missing or extra:
        raise ValueError(f"weight mismatch: missing={sorted(missing)[:8]} extra={sorted(extra)[:8]}")
    model.load_weights(weights, strict=False)


def load_audio_encoder(src_dir: str, dtype=mx.bfloat16) -> MiMoAudioEncoderMLX:
    """Load ``speech_embeddings.*`` + ``audio_encoder.*`` from the main shards (via the index)."""
    cfg = AudioEncoderConfig.from_model_config(os.path.join(src_dir, "config.json"))
    with open(os.path.join(src_dir, "model.safetensors.index.json")) as f:
        wmap = json.load(f)["weight_map"]
    keys = [k for k in wmap if k.startswith(("speech_embeddings.", "audio_encoder."))]
    shards = sorted({wmap[k] for k in keys})
    raw = {}
    kset = set(keys)
    for sh in shards:
        raw.update(_read_safetensors(os.path.join(src_dir, sh), lambda k: k in kset))
    model = MiMoAudioEncoderMLX(cfg)
    weights = []
    for k, v in raw.items():
        if k.startswith("audio_encoder."):
            k = k[len("audio_encoder."):]
            k = k.replace("projection.mlp.0.", "projection.fc1.").replace("projection.mlp.2.", "projection.fc2.")
            if k.startswith("input_local_transformer.embed_tokens."):
                continue  # unused (inputs_embeds path); absent from checkpoint anyway
        weights.append((k, v.astype(dtype)))
    _strict_load(model, weights)
    mx.eval(model.parameters())
    return model


# ---------------------------------------------------------------------------
# High-level helpers
# ---------------------------------------------------------------------------


def prepare_audio(wav: np.ndarray, sr: int, tokenizer: AudioTokenizerEncoderMLX,
                  segment_size: int = SEGMENT_SIZE) -> tuple[mx.array, int]:
    """wav ([T] or [C, T], any integer sr) -> (codes [T_codes, 20] int32, n_audio_pad_tokens)."""
    wav24 = to_mono_24k(wav, sr, tokenizer.cfg.sampling_rate)
    mel = log_mel(wav24, tokenizer.cfg)
    codes = tokenizer.tokenize([mel], segment_size=segment_size)[0]
    n_tokens = audio_token_count_from_mel(mel.shape[0], tokenizer.cfg.kernel_size, tokenizer.cfg.stride_size,
                                          tokenizer.cfg.avg_pooler, 4)
    if math.ceil(codes.shape[0] / 4) != n_tokens:
        raise AssertionError(f"token count mismatch: codes {codes.shape[0]} vs formula {n_tokens}")
    return codes, n_tokens


def embed_audio(codes, encoder: MiMoAudioEncoderMLX) -> mx.array:
    """codes [T, 20] -> [n_tokens, 4096]; row i goes to the i-th <|audio_pad|> of that clip."""
    return encoder(codes)


def merge_audio_embeddings(input_ids: mx.array, inputs_embeds: mx.array, audio_embeds: mx.array,
                           audio_token_id: int = AUDIO_TOKEN_ID) -> mx.array:
    """Scatter audio embeddings into ``inputs_embeds`` at ``<|audio_pad|>`` positions (1-D ids)."""
    ids = np.asarray(input_ids).reshape(-1)
    pos = np.nonzero(ids == audio_token_id)[0]
    if len(pos) != audio_embeds.shape[0]:
        raise ValueError(f"{len(pos)} <|audio_pad|> slots vs {audio_embeds.shape[0]} audio embeddings")
    if len(pos) == 0:
        return inputs_embeds
    flat = inputs_embeds.reshape(-1, inputs_embeds.shape[-1])
    flat[mx.array(pos)] = audio_embeds.astype(flat.dtype)
    return flat.reshape(inputs_embeds.shape)


__all__ = [
    "AudioTokenizerConfig", "AudioEncoderConfig", "AudioTokenizerEncoderMLX", "MiMoAudioEncoderMLX",
    "resample", "log_mel", "mel_filterbank", "to_mono_24k", "num_mel_frames", "audio_token_count",
    "audio_token_count_from_mel", "build_audio_prompt_ids", "load_audio_tokenizer_encoder",
    "load_audio_encoder", "prepare_audio", "embed_audio", "merge_audio_embeddings",
    "AUDIO_TOKEN_ID", "AUDIO_START_TOKEN_ID", "AUDIO_END_TOKEN_ID", "SEGMENT_SIZE",
]
