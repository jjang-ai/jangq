#!/usr/bin/env python3
"""Restamp and prepare Muse Glimmer JANG bundles for OsaurusAI publication.

This prepares local files only.  It intentionally does not create or upload a
Hub repository; the project runtime gate must be satisfied before publication.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from jang_tools.capabilities import build_capabilities
from jang_tools.convert import _muse_glimmer_chat_metadata


SOURCE_REVISION = "f84ecc3a0ea984a4c04542a84269e3d065350a6e"
SOURCE_REPO = "meta-models/Muse-Glimmer-30B"


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _model_card(
    profile: str,
    weight_bytes: int,
    bits: float,
    shards: int,
    tensor_keys: int,
    *,
    text_runtime_verified: bool = False,
) -> str:
    profile_tag = profile.lower().replace("_", "-")
    profile_policy = {
        "JANG_2D": "4-bit attention/lm_head; 3-bit embeddings and MLP gate/down; 2-bit MLP up; FP16 vision",
        "JANG_4M": "8-bit critical; 4-bit important/compress; FP16 vision",
        "JANG_6M": "8-bit critical; 6-bit important/compress; FP16 vision",
    }[profile]
    size_gb = weight_bytes / 1_000_000_000
    size_gib = weight_bytes / (1024**3)
    runtime_status = (
        """**PARTIAL / text generation verified in the target runtime.** On 2026-08-10,
the exact local JANG_2D bundle was loaded by Osaurus/vmlx-swift with all 418
per-module mixed-bit overrides. A request with no sampling or reasoning override,
`Reply with exactly: FOUR`, returned visible content `FOUR`, a separate non-empty
`reasoning_content`, and `finish_reason=stop` at 36.87 tokens/s. This covers
coherent text generation, the native default reasoning path, reasoning-channel
separation, and EOS handling.

Still unverified: grounded image input, grounded video input, ATEM tool round
trips, multi-turn behavior, and prefix/partial-block/suffix cache reuse."""
        if text_runtime_verified
        else """**PARTIAL / runtime unverified.** Current validation covers source identity,
safetensor headers, index integrity, mixed-bit metadata, exact processor/chat
sidecars, deployment generation metadata, FP16 vision preservation, and selected
dequantized-vs-BF16 tensor comparisons. It does not cover coherent target-runtime
generation, image/video grounding, reasoning streaming, ATEM tools, multi-turn
behavior, or prefix/partial-block/suffix cache reuse."""
    )
    korean_runtime = (
        "실제 Osaurus/vmlx-swift에서 기본 추론 경로의 일관된 텍스트 생성, "
        "분리된 reasoning_content 및 EOS 종료를 확인했습니다. 이미지/비디오, ATEM "
        "도구 왕복, 멀티턴 및 캐시 재사용은 아직 검증되지 않았습니다."
        if text_runtime_verified
        else "실제 vmlx-swift 생성, 이미지/비디오, 추론 스트리밍, 도구 왕복 및 "
        "캐시 재사용은 아직 검증되지 않았습니다."
    )
    return f"""---
language:
- en
library_name: mlx
license: apache-2.0
pipeline_tag: image-text-to-text
base_model: {SOURCE_REPO}
base_model_relation: quantized
inference: false
tags:
- mlx
- jang
- {profile_tag}
- quantized
- apple-silicon
- muse-glimmer
- vision-language
- reasoning
- tool-use
- atem
quantization_config:
  family: jang
  weight_format: jang_affine
  profile: {profile}
  group_size: 64
  source_qat: false
  imatrix_applied: false
  awq_applied: false
---

<p align="center"><a href="https://osaurus.ai"><img src="./osaurus-x-banner.png" alt="Osaurus AI"></a></p>

# Muse-Glimmer-30B-{profile}

JANG mixed-affine quantization of
[`{SOURCE_REPO}`](https://huggingface.co/{SOURCE_REPO}) for Apple Silicon
runtimes that implement Muse Glimmer's dense multimodal architecture.

## Bundle

| Field | Value |
|---|---|
| Source revision | `{SOURCE_REVISION}` |
| Format | `jang_affine` / JANG v2 |
| Profile | `{profile}` |
| Bit policy | {profile_policy} |
| Effective language-weight bits | {bits:.2f} |
| Indexed weights | {size_gb:.2f} GB / {size_gib:.2f} GiB |
| Safetensor shards | {shards} |
| Indexed tensor keys | {tensor_keys} |
| Language layers | 52: 39 sliding + 13 full attention |
| Vision tensors | 809 FP16 passthrough tensors |
| Assistant / DFlash | Not included; maintained as a separate artifact |

This is a post-training quantization baseline. The upstream BF16 checkpoint
contains no QAT weights or QAT scale metadata. GPTQ, imatrix, and AWQ were not
applied, and the metadata says so explicitly.

## Native model contract

- Text input and text output are part of the source architecture.
- The vision tower, adapter, projection, image processor, and video processor
  sidecars are preserved. Image/video execution in vMLX remains unverified.
- Reasoning uses `reasoning_strength=low|medium|high|xhigh`; omission defaults
  to `high` in the shipped chat template.
- Reasoning output is an assistant `to=self` channel. Visible content is
  addressed `to=user`.
- Tools use the ATEM function-call grammar. A Muse-specific incremental
  reasoning parser and ATEM tool parser are required.
- Deployment sampling follows the upstream model-card recommendation:
  `do_sample=true`, `temperature=1.0`, `top_p=0.95`, and `top_k=64`. BOS is
  `200000`, EOS is `[200001, 200008]`, pad is `200018`, and maximum length is
  `131072`.
- Cache topology is heterogeneous: rotating KV with window 2048 on 39 layers
  and unbounded KV on full-attention layers `3, 7, ..., 51`.

## Runtime status

{runtime_status}

Do not interpret repository availability or structural loading as production
readiness. This format is not a uniform `mlx_lm` quant; loaders must honor every
per-module entry in `config.json.quantization`.

## Files

- `config.json`: Muse architecture plus per-module affine overrides.
- `jang_config.json`: source revision, profile, capability, reasoning/tool,
  modality, generation, and mixed full/sliding cache metadata.
- `chat_template.jinja`: exact upstream Muse channel/ATEM template.
- `processor_config.json`: exact upstream image/video processor contract.
- `generation_config.json`: upstream-recommended deployment sampling plus the
  source token IDs and maximum length.
- `LICENSE` and `USAGE_POLICY.md`: copied from the pinned upstream source.

## Download

```bash
hf download OsaurusAI/Muse-Glimmer-30B-{profile} \\
  --local-dir ~/models/OsaurusAI/Muse-Glimmer-30B-{profile}
```

## Verification

From the JANG repository:

```bash
PYTHONPATH=jang-tools uv run --no-project \\
  --with mlx --with numpy --with safetensors --with tqdm \\
  python jang-tools/scripts/verify_muse_glimmer_artifact.py \\
  ~/models/OsaurusAI/Muse-Glimmer-30B-{profile} \\
  --profile {profile} --dequant
```

## Korean summary

이 번들은 공식 `{SOURCE_REPO}` BF16 체크포인트를 Apple Silicon용 JANG
혼합 affine 형식으로 변환한 `{profile}` PTQ 모델입니다. 52개 언어 레이어는
슬라이딩/전체 어텐션 구조를 유지하며, 비전 타워·어댑터·프로젝션 809개 텐서는
FP16으로 보존됩니다. 기본 추론 강도는 `high`이고 도구 호출은 ATEM 형식입니다.
현재 파일 구조, 메타데이터, 사이드카 및 일부 역양자화 비교를 확인했습니다.
{korean_runtime} 별도 5레이어 DFlash assistant는 포함하지 않습니다.

## License and use

Apache 2.0 license and the upstream Muse Glimmer Usage Policy apply. Review
`LICENSE` and `USAGE_POLICY.md` before use.

## Contact

eric@osaurus.ai
"""


def prepare(
    bundle: Path,
    source: Path,
    banner: Path,
    profile: str,
    *,
    text_runtime_verified: bool = False,
) -> dict:
    if text_runtime_verified and profile != "JANG_2D":
        raise ValueError("the recorded target-runtime proof applies only to JANG_2D")
    config = _load(bundle / "config.json")
    jang = _load(bundle / "jang_config.json")
    index = _load(bundle / "model.safetensors.index.json")["weight_map"]
    manifest = jang["quantization"]["tensor_quantization_manifest"]

    if jang.get("source_model", {}).get("revision") != SOURCE_REVISION:
        raise ValueError("bundle source revision is not the pinned Muse Glimmer revision")
    if jang.get("quantization", {}).get("profile") != profile:
        raise ValueError(f"bundle profile is not {profile}")

    quant_keys = {
        entry[key]
        for entry in manifest.values()
        for key in ("weight_key", "scales_key", "biases_key")
    }
    missing_quant = quant_keys - set(index)
    if missing_quant:
        raise ValueError(f"manifest keys missing from index: {sorted(missing_quant)[:3]}")
    passthrough_count = len(set(index) - quant_keys)

    jang["weight_format"] = "jang_affine"
    jang["quantization"]["passthrough_tensor_count"] = passthrough_count
    jang["modalities"] = {"text": True, "vision": True, "audio": False, "video": True}
    jang["chat"] = _muse_glimmer_chat_metadata(source)

    config["weight_format"] = "jang_affine"
    config["has_vision"] = True
    config["has_audio"] = False
    config["has_video"] = True
    config["modalities"] = dict(jang["modalities"])

    caps = build_capabilities(jang, config, bundle)
    if caps is None:
        raise ValueError("could not resolve Muse Glimmer capabilities")
    jang["capabilities"] = caps
    config["capabilities"] = caps

    _write(bundle / "jang_config.json", jang)
    _write(bundle / "config.json", config)
    _write(bundle / "generation_config.json", jang["chat"]["generation_defaults"])
    shutil.copy2(source / "LICENSE", bundle / "LICENSE")
    shutil.copy2(source / "USAGE_POLICY.md", bundle / "USAGE_POLICY.md")
    banner_target = bundle / "osaurus-x-banner.png"
    if banner.resolve() != banner_target.resolve():
        shutil.copy2(banner, banner_target)

    shards = len(set(index.values()))
    weight_bytes = sum((bundle / shard).stat().st_size for shard in set(index.values()))
    actual_bits = float(jang["quantization"]["actual_bits"])
    (bundle / "README.md").write_text(
        _model_card(
            profile,
            weight_bytes,
            actual_bits,
            shards,
            len(index),
            text_runtime_verified=text_runtime_verified,
        ),
        encoding="utf-8",
    )
    return {
        "bundle": str(bundle),
        "profile": profile,
        "weight_format": "jang_affine",
        "passthrough_tensor_count": passthrough_count,
        "shards": shards,
        "tensor_keys": len(index),
        "publication_status": (
            "prepared_text_runtime_verified" if text_runtime_verified
            else "prepared_not_uploaded_runtime_gate_open"
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--banner", type=Path, required=True)
    parser.add_argument("--profile", choices=("JANG_2D", "JANG_4M", "JANG_6M"), required=True)
    parser.add_argument("--text-runtime-verified", action="store_true")
    args = parser.parse_args()
    print(json.dumps(prepare(
        args.bundle.expanduser(),
        args.source.expanduser(),
        args.banner.expanduser(),
        args.profile,
        text_runtime_verified=args.text_runtime_verified,
    ), indent=2))


if __name__ == "__main__":
    main()
