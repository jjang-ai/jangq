// JANGStudio/JANGStudio/Runner/CLIArgsBuilder.swift
import Foundation

/// Builds the `python -m ...` argument list for a given ConversionPlan.
/// Pure function — no side effects — so it can be unit tested exhaustively.
enum CLIArgsBuilder {
    /// Returns the argument list to pass to `python3`. Returns [] if sourceURL or
    /// outputURL are missing from the plan.
    static func args(for plan: ConversionPlan) -> [String] {
        guard let src = plan.sourceURL?.path, let out = plan.outputURL?.path else { return [] }
        switch plan.family {
        case .jang:
            var args = ["-m", "jang_tools", "convert", src, "-o", out, "-p", plan.profile,
                        "-m", plan.method.rawValue, "--progress=json", "--quiet-text"]
            if plan.hadamard { args.append("--hadamard") }
            // Advanced overrides — propagated only when explicitly set by the
            // user (Architecture → Advanced overrides). Omitted otherwise so
            // Python's auto-detect stays in charge.
            if let bs = plan.overrides.forceBlockSize, bs > 0 {
                args.append(contentsOf: ["-b", String(bs)])
            }
            if let fd = plan.overrides.forceDtype, let alias = dtypeFlagValue(for: fd) {
                args.append(contentsOf: ["--force-dtype", alias])
            }
            return args
        case .jangtq:
            return jangtqArgs(modelType: plan.detected?.modelType ?? "",
                              src: src,
                              out: out,
                              profile: plan.profile)
        }
    }

    private static func jangtqArgs(modelType: String, src: String, out: String, profile: String) -> [String] {
        switch modelType {
        case "qwen3_5_moe":
            return positionalProgress("jang_tools.convert_qwen35_jangtq", src: src, out: out, profile: profile)
        case "minimax_m2", "minimax_m2_5", "minimax":
            return positionalProgress("jang_tools.convert_minimax_jangtq", src: src, out: out, profile: profile)
        case "hy_v3":
            return positionalProgress("jang_tools.convert_hy3_jangtq", src: src, out: out, profile: profile)
        case "bailing_hybrid", "bailing_moe_v2_5":
            return positionalProgress("jang_tools.convert_ling_jangtq", src: src, out: out, profile: profile)
        case "nemotron_h", "nemotron_h_v2":
            return positionalProgress("jang_tools.convert_nemotron_jangtq", src: src, out: out, profile: profile)
        case "zaya":
            return positionalProgress("jang_tools.convert_zaya_jangtq", src: src, out: out, profile: profile)
        case "zaya1_vl":
            return positionalProgress("jang_tools.convert_zaya1_vl_jangtq", src: src, out: out, profile: profile)
        case "laguna":
            return positionalProgress("jang_tools.convert_laguna_jangtq", src: src, out: out, profile: profile)
        case "mistral3", "mistral4":
            return positionalProgress("jang_tools.convert_mistral3_jangtq", src: src, out: out, profile: profile)
        case "deepseek_v4":
            return dsv4Args(src: src, out: out, profile: profile)
        case "kimi_k25":
            return ["-m", "jang_tools.kimi_prune.convert_kimi_jangtq",
                    "--src", src, "--dst", out, "--profile", profile]
        default:
            return []
        }
    }

    private static func positionalProgress(_ module: String, src: String, out: String, profile: String) -> [String] {
        ["-m", module, "--progress=json", "--quiet-text", src, out, profile]
    }

    private static func dsv4Args(src: String, out: String, profile: String) -> [String] {
        let profileBits: String
        let variant: String
        switch profile {
        case "JANGTQ_K":
            profileBits = "2"
            variant = "K"
        case "JANGTQ3":
            profileBits = "3"
            variant = "V3"
        case "JANGTQ4":
            profileBits = "4"
            variant = "V3"
        default:
            profileBits = "2"
            variant = "V3"
        }
        return ["-m", "jang_tools.dsv4.convert_dsv4_jangtq",
                "--src", src, "--dst", out,
                "--profile", profileBits,
                "--variant", variant]
    }

    /// Map SourceDtype → the alias Python's `jang_tools convert --force-dtype`
    /// accepts. Returns nil for values that don't make sense to force (unknown,
    /// jangV2 — the model is already JANG format, you shouldn't be reconverting).
    private static func dtypeFlagValue(for d: SourceDtype) -> String? {
        switch d {
        case .bf16: return "bf16"
        case .fp16: return "fp16"
        case .fp8: return "fp8"
        case .unknown, .jangV2: return nil
        }
    }
}
