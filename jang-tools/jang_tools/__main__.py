"""
JANG Tools CLI — Mixed-Precision Importance Quantization for Apple Silicon
Created by Jinho Jang (eric@jangq.ai)
"""

import argparse
import json
import sys
import time

from . import __version__
from .progress import ProgressEmitter, make_noop

BANNER = f"""
  ╔══════════════════════════════════════════════════════╗
  ║  JANG Tools v{__version__:<43s}║
  ║  Mixed-Precision Importance Quantization             ║
  ║  for Apple Silicon                                   ║
  ║                                                      ║
  ║  Created by Jinho Jang (eric@jangq.ai)                ║
  ╚══════════════════════════════════════════════════════╝
"""


def cmd_inspect(args):
    """Inspect a JANG model — show bit allocation, quality metrics, size."""
    from .format.reader import load_jang_model

    model = load_jang_model(args.model)
    summary = model.summary()

    print(f"\n  Model: {summary['source_model']}")
    print(f"  Target bits: {summary['target_bits']}")
    print(f"  Actual bits: {summary['actual_bits']}")
    print(f"  Block size: {summary['block_size']}")
    print(f"  Total blocks: {summary['total_blocks']:,}")
    print(f"  Weight tensors: {summary['total_weight_names']}")
    print(f"  Total size: {summary['total_qweight_gb']} GB")
    print()
    print("  Bit allocation:")
    for bw, info in summary["histogram"].items():
        bar = "█" * int(info["percent"] / 2)
        print(f"    {bw:>5s}: {info['count']:>8,} blocks ({info['percent']:>5.1f}%) {bar}")
    print()

    # Show quality metrics if available
    metrics = model.jang_config.get("quality_metrics", {})
    if metrics:
        print("  Quality metrics:")
        for key, val in metrics.items():
            print(f"    {key}: {val}")
        print()


def cmd_validate(args):
    """Validate a JANG model directory."""
    from .format.reader import is_jang_model, load_jang_model

    path = args.model
    if not is_jang_model(path):
        print(f"  ERROR: {path} is not a valid JANG model directory")
        sys.exit(1)

    try:
        model = load_jang_model(path)
        summary = model.summary()
        print(f"  VALID: {path}")
        print(f"  Source: {summary['source_model']}")
        print(f"  Bits: {summary['actual_bits']}")
        print(f"  Blocks: {summary['total_blocks']:,}")
        print(f"  Size: {summary['total_qweight_gb']} GB")
    except Exception as e:
        print(f"  ERROR: {e}")
        sys.exit(1)


def cmd_estimate(args):
    """Estimate JANG model size for a given parameter count and bit width."""
    from .format.spec import estimate_model_size

    # Parse parameter count (e.g., "70B", "14B", "7B")
    param_str = args.params.upper().replace(",", "")
    multipliers = {"B": 1e9, "M": 1e6, "K": 1e3}
    for suffix, mult in multipliers.items():
        if param_str.endswith(suffix):
            num_params = int(float(param_str[:-1]) * mult)
            break
    else:
        num_params = int(param_str)

    print(f"\n  Model: {args.params} parameters")
    print(f"  {'Profile':<12s} {'Nominal':>8s} {'Effective':>10s} {'Size (GB)':>10s}")
    print(f"  {'─' * 42}")

    for target in [2.0, 2.5, 3.0, 4.0, 6.0, 8.0]:
        info = estimate_model_size(num_params, target)
        print(
            f"  JANG-{target:<5.1f}  {info['nominal_bits']:>7.1f}b  "
            f"{info['effective_bits']:>9.2f}b  {info['weight_gb']:>9.1f}"
        )
    print()


def cmd_upgrade(args):
    """Upgrade a JANG v1 model to v2 format (MLX-native, instant load)."""
    try:
        from .loader import upgrade_v1_to_v2, is_jang_model, _is_v2_model
    except ImportError:
        print("  ERROR: 'jang upgrade' requires MLX (Apple Silicon only).")
        print("  Install with: pip install 'jang[mlx]'")
        sys.exit(1)
    from pathlib import Path

    path = Path(args.model)
    if not is_jang_model(path):
        print(f"  ERROR: {path} is not a JANG model directory")
        sys.exit(1)

    if _is_v2_model(path):
        print(f"  Already v2 format — loads instantly via mx.load() mmap")
        sys.exit(0)

    upgrade_v1_to_v2(path)


def cmd_profile(args):
    """Collect expert routing profile for TurboSmelt SSD inference."""
    from .routing_profile import collect_routing_profile

    result = collect_routing_profile(
        model_path=args.model,
        output_path=args.output,
        n_samples=args.samples,
        seq_len=args.seq_len,
    )

    print(f"\n  Routing profile saved: {result['file']}")
    print(f"  Size: {result['size_mb']} MB")
    print(f"  Tokens profiled: {result['n_calibration_tokens']:,}")
    print(f"  MoE layers: {result['n_moe_layers']}")


def cmd_convert(args):
    """Convert a HuggingFace model to JANG format."""
    from .convert import convert_model
    from .allocate import JANG_PROFILES, JANG_K_TARGETS, profile_for_bits, is_k_quant, k_quant_target

    # Resolve profile
    raw = args.profile
    if raw.isdigit():
        profile = profile_for_bits(int(raw))
        print(f"  Bit target {raw} → profile {profile}")
    else:
        profile = raw.upper()

    if profile not in JANG_PROFILES and not is_k_quant(profile):
        print(f"  ERROR: Unknown profile '{profile}'")
        all_profiles = sorted(JANG_PROFILES.keys()) + sorted(JANG_K_TARGETS.keys())
        print(f"  Available: {', '.join(all_profiles)}")
        print(f"  Or use a number 1-8 (e.g., jang convert model -p 2)")
        sys.exit(1)

    # Derive target_bits from profile
    if is_k_quant(profile):
        target_bits = k_quant_target(profile)
    else:
        # Extract from profile name: JANG_2S → 2.0, JANG_4M → 4.0
        for ch in profile.replace("JANG_", ""):
            if ch.isdigit():
                target_bits = float(ch)
                break
        else:
            target_bits = 4.0

    # Output path
    output = args.output
    if not output:
        import os
        name = os.path.basename(args.model.rstrip("/"))
        for suffix in ["-BF16", "-bf16", "-FP16", "-fp16"]:
            if name.endswith(suffix):
                name = name[:-len(suffix)]
        output = f"{name}-{profile}"

    from .convert import DEFAULT_BLOCK_SIZE
    block_size = args.block_size if args.block_size > 0 else DEFAULT_BLOCK_SIZE
    result = convert_model(
        model_path=args.model,
        output_path=output,
        target_bits=target_bits,
        profile=profile,
        quantization_method=args.method,
        hadamard=args.hadamard,
        block_size=block_size,
        force_dtype=args.force_dtype,
        apply_mlp_asymmetry=not args.no_mlp_asymmetry_floor,
        expert_prune_keep=args.expert_prune_keep,
        expert_prune_map=args.expert_prune_map,
        split_gate_up_quant=args.split_gate_up_quant,
        split_gate_bits=args.split_gate_bits,
        split_up_bits=args.split_up_bits,
        n2_down_bits=args.n2_down_bits,
        progress_emitter=getattr(args, "progress_emitter", None),
    )

    print(f"\n  Profile: {profile}")
    print(f"  Actual bits: {result['actual_bits']}")
    print(f"  Weight size: {result['total_weight_gb']} GB")
    print(f"  Output: {output}")


def main():
    # Early-intercept passthrough subcommands BEFORE the strict top-level
    # argparse runs. These forward their own --flags to a vendored tool's
    # argparse main(); argparse.REMAINDER can't capture a leading --flag (the
    # parent parser consumes it), so we slice the raw argv here instead. They
    # are still registered below for `jang --help` discoverability.
    if len(sys.argv) >= 2 and sys.argv[1] == "minimax-m3":
        from .minimax_m3.cli import dispatch as _m3_dispatch
        sys.exit(_m3_dispatch(sys.argv[2:]))
    if len(sys.argv) >= 2 and sys.argv[1] == "prune-n2-experts":
        from .prune_n2_jang_experts import dispatch as _n2_dispatch
        sys.exit(_n2_dispatch(sys.argv[2:]))

    parser = argparse.ArgumentParser(
        prog="jang",
        description="JANG: Mixed-Precision Importance Quantization for Apple Silicon",
    )
    parser.add_argument("--version", action="version", version=f"jang-tools {__version__}")
    parser.add_argument(
        "--progress", choices=["json", "off"], default="off",
        help="Emit JSONL progress events on stderr (for GUIs).",
    )
    parser.add_argument(
        "--quiet-text", action="store_true",
        help="Suppress human-readable phase/progress prints on stdout.",
    )
    subparsers = parser.add_subparsers(dest="command")

    # inspect
    p_inspect = subparsers.add_parser("inspect", help="Inspect a JANG model")
    p_inspect.add_argument("model", help="Path to JANG model directory")
    p_inspect.set_defaults(func=cmd_inspect)

    # validate
    p_validate = subparsers.add_parser("validate", help="Validate a JANG model directory")
    p_validate.add_argument("model", help="Path to JANG model directory")
    p_validate.set_defaults(func=cmd_validate)

    # estimate
    p_estimate = subparsers.add_parser("estimate", help="Estimate JANG model sizes")
    p_estimate.add_argument("params", help="Parameter count (e.g., 70B, 14B, 7B)")
    p_estimate.set_defaults(func=cmd_estimate)

    # convert
    p_convert = subparsers.add_parser("convert", help="Convert a HuggingFace model to JANG format")
    p_convert.add_argument("model", help="Path to HuggingFace model directory")
    p_convert.add_argument("-o", "--output", help="Output directory (default: auto)")
    p_convert.add_argument("-p", "--profile", default="2",
                          help="JANG profile (e.g., JANG_2L, JANG_3M) or number 1-8 (default: 2)")
    p_convert.add_argument("-m", "--method", default="mse", choices=["mse", "rtn", "mse-all"],
                          help="Quantization method (default: mse)")
    p_convert.add_argument("-b", "--block-size", type=int, default=0,
                          help="Quantization group size in weights per block. 0 = auto (default). "
                               "Typical values: 32, 64, 128. Large-expert MoE models (150+) auto-pick "
                               "128 unless this flag overrides.")
    p_convert.add_argument("--force-dtype", choices=["bf16", "fp16", "fp8"], default=None,
                          help="Treat source tensors as this dtype during load. Overrides the per-tensor "
                               "safetensors-header sniff. Useful when the header is stripped or mislabeled. "
                               "Default: auto-detect per tensor.")
    p_convert.add_argument("--hadamard", action="store_true",
                          help="Apply Hadamard rotation before quantization (QuIP# style, ~0.5-1 bit quality gain)")
    p_convert.add_argument("--no-mlp-asymmetry-floor", action="store_true",
                          help="Disable routed-expert gate/down precision floors for legacy compact 2-bit MoE builds")
    p_convert.add_argument("--expert-prune-keep", type=int, default=None,
                          help="Keep the top-K routed experts per layer by router row L2 while converting")
    p_convert.add_argument("--expert-prune-map", type=str, default=None,
                          help="Activation-guided expert keep map/profile JSON; use with --expert-prune-keep")
    p_convert.add_argument("--split-gate-up-quant", action="store_true",
                          help="For fused MoE gate_up_proj tensors, quantize gate and up halves separately")
    p_convert.add_argument("--split-gate-bits", type=int, default=4, choices=[2, 3, 4, 6, 8],
                          help="Bit width for the gate half when --split-gate-up-quant is enabled")
    p_convert.add_argument("--split-up-bits", type=int, default=2, choices=[2, 3, 4, 6, 8],
                          help="Bit width for the up half when --split-gate-up-quant is enabled")
    p_convert.add_argument("--n2-down-bits", type=int, default=None, choices=[2, 3, 4, 6, 8],
                          help="Force Nex/N2 fused expert down_proj tensors to this bit width")
    p_convert.set_defaults(func=cmd_convert)

    # profile
    p_profile = subparsers.add_parser("profile",
        help="Collect expert routing profile for TurboSmelt SSD inference")
    p_profile.add_argument("model", help="Path to JANG or MLX MoE model directory")
    p_profile.add_argument("-o", "--output", help="Output directory (default: model dir)")
    p_profile.add_argument("-n", "--samples", type=int, default=256,
                          help="Number of calibration samples (default: 256)")
    p_profile.add_argument("--seq-len", type=int, default=512,
                          help="Max sequence length per sample (default: 512)")
    p_profile.set_defaults(func=cmd_profile)

    # upgrade
    p_upgrade = subparsers.add_parser("upgrade",
        help="Upgrade JANG v1 model to v2 (MLX-native, instant load)")
    p_upgrade.add_argument("model", help="Path to JANG v1 model directory")
    p_upgrade.set_defaults(func=cmd_upgrade)

    # --- spec subcommand (jang-spec bundle tooling) ---
    from .jangspec.cli import register_subparsers as _register_spec
    p_spec = subparsers.add_parser("spec", help="jang-spec bundle tooling")
    _register_spec(p_spec)

    from .inspect_source import register as _register_inspect_source
    _register_inspect_source(subparsers)

    from .examples import register as _register_examples
    _register_examples(subparsers)

    from .modelcard import register as _register_modelcard
    _register_modelcard(subparsers)

    from .inference import register as _register_inference
    _register_inference(subparsers)

    from .profiles_cli import register as _register_profiles
    _register_profiles(subparsers)

    from .capabilities_cli import register as _register_capabilities
    _register_capabilities(subparsers)

    from .estimate_model import register as _register_estimate_model
    _register_estimate_model(subparsers)

    from .publish import register as _register_publish
    _register_publish(subparsers)

    from .recommend import register as _register_recommend
    _register_recommend(subparsers)

    from .minimax_m3.cli import register as _register_minimax_m3
    _register_minimax_m3(subparsers)

    from .prune_n2_jang_experts import register as _register_prune_n2
    _register_prune_n2(subparsers)

    args = parser.parse_args()

    if args.command is None:
        print(BANNER)
        parser.print_help()
        sys.exit(0)

    if not hasattr(args, "func"):
        parser.print_help()
        return

    suppress_banner = args.quiet_text or (
        args.command in ("inspect-source", "examples", "modelcard", "inference",
                         "profiles", "capabilities", "estimate-model", "publish", "recommend")
        and getattr(args, "json", False)
    )
    if not suppress_banner:
        print(BANNER)
    progress = ProgressEmitter(
        json_to_stderr=(args.progress == "json"),
        quiet_text=args.quiet_text,
    )
    args.progress_emitter = progress
    t0 = time.time()
    try:
        args.func(args)
        progress.done(ok=True, elapsed_s=time.time() - t0)
    except SystemExit as e:
        progress.done(ok=False, error=f"exit-code-{e.code}", elapsed_s=time.time() - t0)
        raise
    except Exception as e:
        progress.done(ok=False, error=f"{type(e).__name__}: {e}", elapsed_s=time.time() - t0)
        raise


if __name__ == "__main__":
    main()
