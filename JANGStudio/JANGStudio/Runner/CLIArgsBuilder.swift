// JANGStudio/JANGStudio/Runner/CLIArgsBuilder.swift
import Foundation

/// Builds the `python -m ...` argument list for a given ConversionPlan.
/// Pure function — no side effects — so it can be unit tested exhaustively.
enum CLIArgsBuilder {
    /// `model_type` / `text_model_type` token → `python -m` module.
    /// Intentionally separate from `Capabilities.jangtqWhitelist`:
    /// whitelist = UI allow list; this map = which converter binary to run.
    /// When whitelist grows without a module entry, `args(for:)` returns `[]`
    /// (hard-fail) rather than silently routing to the Qwen converter.
    static let jangtqModuleByModelType: [String: String] = [
        "qwen3_5_moe": "jang_tools.convert_qwen35_jangtq",
        "qwen3_5_moe_text": "jang_tools.convert_qwen35_jangtq",
        "minimax_m2": "jang_tools.convert_minimax_jangtq",
    ]

    /// First matching `architectureModelTypes` entry wins (order: modelType
    /// then textModelType — see `ArchitectureSummary.architectureModelTypes`).
    static func jangtqModule(for plan: ConversionPlan) -> String? {
        guard let detected = plan.detected else { return nil }
        for token in detected.architectureModelTypes {
            if let mod = jangtqModuleByModelType[token] { return mod }
        }
        return nil
    }

    /// Pure reason string for empty argv — unit-testable without RunStep UI.
    /// Returns `nil` when `args(for:)` would be non-empty.
    static func failureReason(for plan: ConversionPlan) -> String? {
        if plan.sourceURL == nil || plan.outputURL == nil {
            return "Cannot build converter argv: sourceURL and outputURL are required."
        }
        if plan.family == .jangtq, jangtqModule(for: plan) == nil {
            let types = plan.detected?.architectureModelTypes.joined(separator: ", ") ?? "(none)"
            return "Cannot build converter argv: JANGTQ has no module mapping for model types [\(types)]. Use a JANG profile, or pick a supported architecture."
        }
        return nil
    }

    /// Returns the argument list to pass to `python3`. Returns [] if sourceURL or
    /// outputURL are missing, or if family is JANGTQ with no module mapping for
    /// the detected architecture.
    static func args(for plan: ConversionPlan) -> [String] {
        guard let src = plan.sourceURL?.path, let out = plan.outputURL?.path else { return [] }
        switch plan.family {
        case .jang:
            var args = ["-m", "jang_tools", "--progress=json", "--quiet-text",
                        "convert", src, "-o", out, "-p", plan.profile,
                        "-m", plan.method.rawValue]
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
            guard let mod = jangtqModule(for: plan) else { return [] }
            return ["-m", mod, "--progress=json", "--quiet-text", src, out, plan.profile]
        }
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
