// JANGStudio/JANGStudio/Verify/PreflightRunner.swift
import CryptoKit
import Foundation
import JANGExpertLab

struct PreflightRunner {
    func run(plan: ConversionPlan,
             capabilities: Capabilities = .frozen,
             profiles: Profiles = .frozen) -> [PreflightCheck] {
        var out: [PreflightCheck] = []
        let src = plan.sourceURL
        let dst = plan.outputURL

        out.append(Self.sourceReadable(src))
        out.append(Self.configValid(src))
        out.append(Self.outputUsable(src: src, dst: dst))
        // M141 (iter 63): diskSpace was being called with `estimated: 0`,
        // which makes the function short-circuit to `.pass` unconditionally.
        // The gate was inert. Compute a profile-aware estimate from the
        // source bytes × (avgBits / 16) × 1.05 metadata overhead — same
        // formula as `estimate_model.predict` on the Python side, keeping
        // the two size-estimates aligned across the Swift⇄Python boundary
        // (M140 meta-lesson about cross-boundary decision-overlap).
        let estimated = Self.estimateOutputBytes(plan: plan, profiles: profiles)
        out.append(Self.diskSpace(dst: dst, estimated: estimated))
        out.append(Self.ramAdequate(plan: plan))
        out.append(Self.jangtqArchSupported(plan: plan, whitelist: capabilities.jangtqWhitelist))
        out.append(Self.jangtqSourceDtype(plan: plan))
        out.append(Self.bf16For512Experts(plan: plan, types: capabilities.knownExpert512Types))
        out.append(Self.hadamardVsLowBits(plan: plan, profiles: profiles))
        if let reviewed = Self.reviewedPruneVerifiedCheck(plan: plan) {
            out.append(reviewed)
        }
        out.append(Self.bundledPythonHealthy())
        return out
    }

    /// M141 (iter 63): profile-aware output-size estimator for the
    /// preflight disk-space gate. Returns 0 when the source hasn't been
    /// inspected yet (preserves the pre-iter-63 pass-through behavior on
    /// the initial empty-plan render).
    ///
    /// Formula mirrors `jang_tools/estimate_model.predict` so the preflight
    /// warning, the wizard's predicted-size banner, and the Python-side
    /// downstream are all in agreement. Assumes source is BF16 (16 bits /
    /// weight) — for an FP8 source the real output will be slightly
    /// smaller than this estimate, but conservative-over predicts are OK
    /// (the disk-space gate is an INEQUALITY: "have at least N free").
    static func estimateOutputBytes(plan: ConversionPlan, profiles: Profiles) -> Int64 {
        guard let srcBytes = plan.detected?.totalBytes, srcBytes > 0 else { return 0 }
        let avgBits = avgBitsForProfile(plan.profile, profiles: profiles)
        guard avgBits > 0 else { return 0 }
        // M173 (iter 99): the divisor is source-dtype-dependent, not a
        // hardcoded 16. Pre-M173 every source was assumed 16-bit — correct
        // for BF16/FP16 (the common case) but WRONG for FP8 sources like
        // DeepSeek V3/V3.2 where src_bytes = weights × 1 (not × 2). A 100 GB
        // FP8 source converting to JANG_4K would be predicted as 26 GB (half
        // the real 52 GB need) → user sees "plenty of disk" → convert fails
        // mid-way on disk-full. `sourceBytesPerWeight` maps the detected
        // dtype to bytes-per-weight; the formula becomes
        //   srcBytes × (avgBits / 8) / bytesPerWeight × 1.05
        // which is equivalent to `srcBytes × avgBits / (8 × bytesPerWeight)`.
        // Unknown dtype falls back to 2 (BF16 assumption) — conservative
        // over-estimate is safer than under-estimate for a disk-space gate.
        let bytesPerWeight = Self.sourceBytesPerWeight(plan.detected?.dtype ?? .unknown)
        return Int64(Double(srcBytes) * avgBits / (8.0 * Double(bytesPerWeight)) * 1.05)
    }

    /// M173 (iter 99): bytes-per-weight for each supported source dtype.
    /// Keeps the Swift preflight estimator aligned with Python's
    /// `estimate_model.predict` per iter-63 M141's cross-boundary contract.
    /// Unknown → 2 (BF16/FP16 default) for safety.
    static func sourceBytesPerWeight(_ dtype: SourceDtype) -> Int {
        switch dtype {
        case .bf16, .fp16: return 2
        case .fp8: return 1
        case .jangV2:
            // Already-quantized source — the preflight estimator is
            // fundamentally off for this case since jangV2 carries variable
            // bit-width, not a uniform byte-per-weight mapping. Treat as
            // BF16-equivalent; this class of source shouldn't normally
            // reach the convert preflight anyway (requantization is
            // atypical). Safer to over-estimate than under-estimate.
            return 2
        case .unknown: return 2   // conservative over-estimate fallback
        }
    }

    /// Look up the avg bits/weight for a profile from either JANG or JANGTQ
    /// tables. Returns 0 on unknown profile (caller falls back to pass).
    static func avgBitsForProfile(_ profile: String, profiles: Profiles) -> Double {
        if let p = profiles.jang.first(where: { $0.name == profile }) { return p.avgBits }
        if let p = profiles.jangtq.first(where: { $0.name == profile }) { return Double(p.bits) }
        return 0
    }

    /// M142 (iter 64): the authoritative "is this profile a 2-bit compress
    /// tier?" answer — used by hadamardVsLowBits. Returns the compress-tier
    /// bits for JANG profiles (criticalBits/importantBits stay high while
    /// compressBits drives MLP quality, which is what Hadamard rotation
    /// affects) and the uniform bits for JANGTQ. Returns nil on unknown
    /// profile so callers can fall back to pass instead of guessing.
    ///
    /// JANG_NK (K-quant) profiles expose criticalBits=nil in the schema;
    /// for those we derive from avgBits as a robust fallback (JANG_4K has
    /// avgBits=4.0 → compress-equivalent 4).
    static func compressBitsForProfile(_ profile: String, profiles: Profiles) -> Int? {
        if let p = profiles.jang.first(where: { $0.name == profile }) {
            if let cb = p.compressBits { return cb }
            // K-quant profiles have nil compressBits — the compress tier is
            // the uniform avg (no separate tiers).
            return Int(p.avgBits.rounded())
        }
        if let p = profiles.jangtq.first(where: { $0.name == profile }) {
            return p.bits
        }
        return nil
    }

    private static func sourceReadable(_ url: URL?) -> PreflightCheck {
        guard let url else { return .init(id: .sourceReadable, title: "Source dir exists", status: .fail, hint: "No source selected") }
        let ok = FileManager.default.isReadableFile(atPath: url.path)
        return .init(id: .sourceReadable, title: "Source dir exists",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "\(url.path) is not readable")
    }

    private static func configValid(_ url: URL?) -> PreflightCheck {
        guard let url else { return .init(id: .configJSONValid, title: "config.json parses", status: .fail, hint: nil) }
        let cfg = url.appendingPathComponent("config.json")
        guard let data = try? Data(contentsOf: cfg),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
              (obj["model_type"] as? String) != nil || ((obj["text_config"] as? [String: Any])?["model_type"] as? String) != nil
        else {
            return .init(id: .configJSONValid, title: "config.json parses", status: .fail,
                         hint: "config.json missing or no model_type")
        }
        return .init(id: .configJSONValid, title: "config.json parses", status: .pass, hint: nil)
    }

    private static func outputUsable(src: URL?, dst: URL?) -> PreflightCheck {
        guard let dst else { return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Pick an output folder") }
        if dst == src { return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Output cannot equal source") }
        // M139 (iter 61): reject nested src/dst. If output lives INSIDE the
        // source tree (or source inside output), the two directories share
        // safetensors shards in the same subtree. Recursive greps / future
        // cleanup passes could touch the wrong set. Also confuses users who
        // later `rm -rf source/` and discover their output went with it.
        // The plain-equal check above doesn't cover this case because the
        // paths differ — one is a strict prefix of the other.
        if let s = src {
            let srcPath = s.standardizedFileURL.path
            let dstPath = dst.standardizedFileURL.path
            // Use path + "/" to prevent sibling-prefix matches
            // (e.g. `/a/b` is NOT inside `/a/bc`).
            if dstPath.hasPrefix(srcPath + "/") {
                return .init(id: .outputUsable, title: "Output dir valid", status: .fail,
                             hint: "Output cannot be inside the source folder")
            }
            if srcPath.hasPrefix(dstPath + "/") {
                return .init(id: .outputUsable, title: "Output dir valid", status: .fail,
                             hint: "Source cannot be inside the output folder")
            }
        }
        if dst.path.contains(".app/Contents") {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Do not write inside an .app")
        }
        let parent = dst.deletingLastPathComponent()
        if !FileManager.default.isWritableFile(atPath: parent.path) {
            return .init(id: .outputUsable, title: "Output dir valid", status: .fail, hint: "Parent not writable")
        }
        return .init(id: .outputUsable, title: "Output dir valid", status: .pass, hint: nil)
    }

    private static func diskSpace(dst: URL?, estimated: Int64) -> PreflightCheck {
        guard let dst else { return .init(id: .diskSpace, title: "Free disk space", status: .fail, hint: nil) }
        let parent = dst.deletingLastPathComponent()
        let rv = try? parent.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        let free = Int64(rv?.volumeAvailableCapacityForImportantUsage ?? 0)
        // M05 (iter 101): pre-M05, when `estimated <= 0` (source not yet
        // inspected, or unknown profile blocking the estimator), this branch
        // returned `.pass` with a plain "X GB free" hint — same UI state as
        // a real positive check. User couldn't tell whether the system had
        // actually verified sufficient space vs. simply couldn't compute an
        // estimate yet. Now returns `.warn` with an explicit "(no estimate)"
        // marker so the UX makes the uncheckable state visible. The gate
        // stays functional (warn doesn't block preflight like fail would)
        // but the user knows they should come back after Profile is picked.
        if estimated <= 0 {
            return .init(id: .diskSpace, title: "Free disk space", status: .warn,
                         hint: "\(free / 1_000_000_000) GB free (no estimate yet — pick source + profile for a real check)")
        }
        let ok = free >= estimated
        return .init(id: .diskSpace, title: "Free disk space",
                     status: ok ? .pass : .fail,
                     hint: ok ? "\(free / 1_000_000_000) GB free" : "Need ~\(estimated / 1_000_000_000) GB, have \(free / 1_000_000_000) GB")
    }

    private static func ramAdequate(plan: ConversionPlan) -> PreflightCheck {
        let ram = Int64(ProcessInfo.processInfo.physicalMemory)
        // M175 (iter 102): sibling of M05 — pre-M175 this returned
        // `.pass` with `nil` hint when totalBytes was unknown (pre-
        // inspection). Same ambiguous-pass anti-pattern M05 closed on
        // diskSpace. RAM OOM mid-convert is even worse than disk-full
        // (convert may get killed by OS instead of surfacing a
        // readable error). Promote to `.warn` with an "uncheckable"
        // hint so the user knows to re-check after inspection lands.
        guard let srcBytes = plan.detected?.totalBytes, srcBytes > 0 else {
            return .init(id: .ramAdequate, title: "RAM adequate", status: .warn,
                         hint: "\(ram / 1_000_000_000) GB installed (no estimate yet — pick source for a real check)")
        }
        let needed = Int64(Double(srcBytes) * 1.5)
        let ok = ram >= needed
        return .init(id: .ramAdequate, title: "RAM adequate",
                     status: ok ? .pass : .warn,
                     hint: ok ? nil : "~\(needed / 1_000_000_000) GB needed; you have \(ram / 1_000_000_000) GB. Conversion may swap or OOM.")
    }

    /// JANGTQ is allowed only when the architecture is on the capabilities
    /// whitelist **and** Studio has a concrete converter module mapping
    /// (`CLIArgsBuilder.jangtqModule`). Whitelist alone is not enough: a
    /// future arch can land on the whitelist before a Studio module entry
    /// ships; without the module check, preflight would go green while
    /// `CLIArgsBuilder.args` returns `[]` and Run fails late.
    private static func jangtqArchSupported(plan: ConversionPlan, whitelist: [String]) -> PreflightCheck {
        if plan.family != .jangtq {
            return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported", status: .pass, hint: nil)
        }
        let types = plan.detected?.architectureModelTypes ?? []
        let typesDesc = types.isEmpty ? (plan.detected?.modelType ?? "?") : types.joined(separator: ", ")
        let allowed = plan.isJANGTQAllowed(for: whitelist)
        let module = CLIArgsBuilder.jangtqModule(for: plan)
        if allowed, module != nil {
            return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                         status: .pass, hint: nil)
        }
        if !allowed {
            return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                         status: .fail,
                         hint: "JANGTQ supports \(whitelist.joined(separator: ", ")); detected [\(typesDesc)]")
        }
        // Whitelisted (or would be via textModelType) but no converter module.
        return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                     status: .fail,
                     hint: "No JANGTQ converter module mapped for [\(typesDesc)] in this Studio build. Use a JANG profile.")
    }

    private static func jangtqSourceDtype(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqSourceDtype, title: "Source dtype supported", status: .pass, hint: nil) }
        let d = plan.detected?.dtype ?? .unknown
        let ok = (d == .bf16 || d == .fp16 || d == .fp8)
        return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ expects BF16, FP16, or FP8 source; detected \(d.rawValue)")
    }

    private static func bf16For512Experts(plan: ConversionPlan, types: [String]) -> PreflightCheck {
        let mt = plan.detected?.modelType ?? ""
        // M140 (iter 62): preflight-side of the M131 fix. Iter 53 made
        // `_recommend_dtype` (Python) dynamically promote any MoE with
        // `expert_count >= 512` to bfloat16 instead of relying on a
        // hardcoded `{minimax_m2, glm_moe_dsa}` name list. The preflight
        // check here had the same decision-overlap bug — it only flagged
        // types in `knownExpert512Types` (a hardcoded list in
        // capabilities_cli.py). A future 512+ expert family (e.g., a
        // future Qwen / DeepSeek variant) would skip this warning despite
        // needing bfloat16 for the same float16-overflow reason.
        //
        // Fix: check BOTH the named-family list AND the dynamic expert
        // count. Mirrors the Python-side recommend.py fix exactly.
        let dynamic512 = (plan.detected?.numExperts ?? 0) >= 512
        guard types.contains(mt) || dynamic512 else {
            return .init(id: .bf16For512Experts, title: "512+ expert dtype guard", status: .pass, hint: nil)
        }
        if plan.overrides.forceDtype == .fp16 {
            let expertStr = dynamic512 ? "\(plan.detected?.numExperts ?? 0) experts" : mt
            return .init(id: .bf16For512Experts, title: "512+ expert dtype guard", status: .warn,
                         hint: "\(expertStr) — bfloat16 strongly recommended over float16 to avoid overflow")
        }
        return .init(id: .bf16For512Experts, title: "512+ expert dtype guard", status: .pass, hint: nil)
    }

    private static func hadamardVsLowBits(plan: ConversionPlan, profiles: Profiles) -> PreflightCheck {
        // M142 (iter 64): use the profile's authoritative compress-bits
        // field instead of `plan.profile.contains("_2")` substring match.
        // The substring match is brittle to:
        //   - Future "JANG_20" / "JANGTQ_2X" / similar profile names where
        //     "_2" wouldn't mean 2-bit.
        //   - Profiles that should be flagged but don't contain "_2"
        //     (current JANG_1L is specifically hardcoded to work around
        //     this; a future JANG_0L would need the same treatment).
        // With compressBitsForProfile, the check is driven by the profile
        // data structure — same source of truth as ProfilesService.frozen
        // and as jang-tools' Python-side allocate.py JANG_PROFILES.
        let compress = Self.compressBitsForProfile(plan.profile, profiles: profiles)
        let is2bit = (compress ?? 99) <= 2
        if plan.hadamard && is2bit {
            return .init(id: .hadamardVsLowBits, title: "Hadamard rotation sanity", status: .warn,
                         hint: "Hadamard rotation hurts quality at 2-bit and below. Turn off for this profile.")
        }
        return .init(id: .hadamardVsLowBits, title: "Hadamard rotation sanity", status: .pass, hint: nil)
    }

    static func reviewedPruneVerifiedCheck(plan: ConversionPlan) -> PreflightCheck? {
        let hasReviewedPruneProvenance = plan.expertReviewOriginalSourceURL != nil
            || plan.expertReviewPrunedSourceURL != nil
            || plan.expertReviewPrunePlanURL != nil
            || plan.expertReviewPruneReportURL != nil
        guard hasReviewedPruneProvenance else { return nil }

        let title = "Reviewed BF16/F16 prune verified"
        guard let pruned = plan.expertReviewPrunedSourceURL else {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Reviewed prune provenance is missing the pruned source path")
        }
        guard plan.sourceURL == pruned else {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Final conversion source must be the verified pruned BF16/F16 directory")
        }
        let planURL = Self.reviewedPrunePlanURL(plan: plan, prunedSource: pruned)
        guard FileManager.default.isReadableFile(atPath: planURL.path) else {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Reviewed prune plan is missing or unreadable")
        }
        let verificationURL = pruned.appendingPathComponent("verification.json")
        guard let data = try? Data(contentsOf: verificationURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Missing or unreadable verification.json in pruned BF16/F16 source")
        }
        let verification = Self.readPrunedSourceVerification(from: obj)
        guard verification.ok else {
            let errors = (obj["errors"] as? [String])?.joined(separator: "; ")
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: errors?.isEmpty == false ? errors : verification.hint)
        }
        let suiteEvidence = Self.readPrunedSourceReviewEvidence(from: pruned)
        guard suiteEvidence.ok else {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail, hint: suiteEvidence.hint)
        }
        guard let prunePlan = readJSONObject(planURL) else {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Reviewed prune plan JSON is unreadable")
        }
        if let issue = reviewedPrunePlanIssue(prunePlan) {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Reviewed prune plan failed: \(issue)")
        }
        if let issue = reviewedPrunePlanSidecarConsistencyIssue(plan: prunePlan, prunedSource: pruned) {
            return .init(id: .reviewedPruneVerified, title: title, status: .fail,
                         hint: "Reviewed prune plan failed: \(issue)")
        }
        return .init(id: .reviewedPruneVerified, title: title, status: .pass,
                     hint: "verification.json and same-suite Expert Lab review sidecars passed")
    }

    private static func reviewedPrunePlanURL(plan _: ConversionPlan, prunedSource: URL) -> URL {
        return prunedSource.appendingPathComponent("prune_plan.json")
    }

    private static func readPrunedSourceReviewEvidence(from prunedSource: URL) -> (ok: Bool, hint: String) {
        let fm = FileManager.default
        let summaryURL = prunedSource.appendingPathComponent("expert_lab_review_summary.json")
        guard let data = try? Data(contentsOf: summaryURL),
              let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return (false, "Missing or unreadable expert_lab_review_summary.json in pruned BF16/F16 source")
        }
        guard obj["same_suite_verification_ready"] as? Bool == true else {
            return (false, "Expert Lab same-suite review sidecars are incomplete; rerun BF16/F16 prune from the reviewed plan")
        }
        guard let reviewPrunedSource = stringValue(obj["pruned_source"] ?? obj["prunedSource"]),
              !reviewPrunedSource.isEmpty else {
            return (false, "Expert Lab review summary is missing pruned BF16/F16 source path evidence")
        }
        if canonicalPath(reviewPrunedSource) != canonicalPath(prunedSource.path) {
            return (false, "Expert Lab review summary pruned source path does not match the selected pruned BF16/F16 source")
        }
        let embeddedSidecars = embeddedReviewSidecars(summary: obj, prunedSource: prunedSource)
        if let issue = embeddedSidecars.issue {
            return (false, issue)
        }
        let sidecars = embeddedSidecars.urls
        let requiredPaths: [(String, URL?)] = [
            ("suite_jsonl", sidecars["suite_jsonl"]),
            ("comparison_summary", sidecars["comparison_summary"]),
            ("eval_jsonl", sidecars["eval_jsonl"]),
            ("eval_trace_jsonl", sidecars["eval_trace_jsonl"]),
            ("eval_index", sidecars["eval_index"]),
            ("mask_json", sidecars["mask_json"]),
        ]
        let missing = requiredPaths.compactMap { key, url -> String? in
            guard let url,
                  fm.isReadableFile(atPath: url.path) else {
                return key
            }
            return nil
        }
        if !missing.isEmpty {
            return (false, "Missing Expert Lab review sidecar files: \(missing.joined(separator: ", "))")
        }
        let expectedLayerCount = expectedReviewedLayerCount(summary: obj)
        var comparison: [String: Any]?
        if let comparisonURL = sidecars["comparison_summary"] {
            guard let loadedComparison = readJSONObject(comparisonURL) else {
                return (false, "Expert Lab same-suite comparison summary is unreadable")
            }
            comparison = loadedComparison
            if let issue = reviewedPruneComparisonGateIssue(
                comparison: loadedComparison,
                tracedPromptCount: intValue(obj["prompt_count"])
            ) {
                return (false, "Expert Lab same-suite comparison failed: \(issue)")
            }
            if let issue = comparisonSafeDropMaskIssue(
                comparison: loadedComparison,
                maskURL: sidecars["mask_json"]
            ) {
                return (false, "Expert Lab same-suite comparison failed: \(issue)")
            }
        }
        if let evalIndexURL = sidecars["eval_index"] {
            guard let index = readJSONObject(evalIndexURL) else {
                return (false, "Expert Lab per-prompt eval index is unreadable")
            }
            if let issue = reviewedPruneEvalIndexIssue(
                index: index,
                comparedPromptCount: comparisonPromptCount(from: obj),
                tracedPromptCount: intValue(obj["prompt_count"]),
                comparison: comparison,
                suiteURL: sidecars["suite_jsonl"],
                evalURL: sidecars["eval_jsonl"],
                evalTraceURL: sidecars["eval_trace_jsonl"],
                maskURL: sidecars["mask_json"],
                sourceModelPath: stringValue(obj["source_model_path"] ?? obj["source_model"]),
                expectedLayerCount: expectedLayerCount
            ) {
                return (false, "Expert Lab per-prompt eval index failed: \(issue)")
            }
        }
        if let issue = prunedSourceSuiteVerificationIssue(
            summary: obj,
            prunedSource: prunedSource,
            tracedPromptCount: intValue(obj["prompt_count"]),
            suiteURL: sidecars["suite_jsonl"],
            expectedLayerCount: expectedLayerCount
        ) {
            return (false, "Pruned BF16/F16 prompt-suite verification failed: \(issue)")
        }
        return (true, summaryURL.path)
    }

    private static let minimumReviewedPrunePromptCount = 50
    private static let minimumReviewedPruneMeanTokens: Double = 8

    /// Intent Prune plans (`jang-intent-prune-plan-v1`) share the reviewed hard-prune path.
    /// Hybrid scorer fields are evidence; `layers` keep lists are operational.
    private static let intentPrunePlanSchema = "jang-intent-prune-plan-v1"

    private static func isIntentPrunePlan(_ plan: [String: Any]) -> Bool {
        if let schema = stringValue(plan["schema"]), schema == intentPrunePlanSchema {
            return true
        }
        // Accept hybrid_v1 (and future scorers) when schema omitted but scorer is set
        // and Expert Lab `method` is absent — do not reject unknown scorer strings.
        if stringValue(plan["scorer"]) != nil, stringValue(plan["method"]) == nil {
            return true
        }
        return false
    }

    private static func planTracedPromptCount(_ plan: [String: Any]) -> Int? {
        if let count = intValue(plan["promptCount"] ?? plan["prompt_count"]) {
            return count
        }
        if let suite = plan["suite"] as? [String: Any] {
            return intValue(suite["prompt_count"] ?? suite["promptCount"])
        }
        return nil
    }

    private static func intentPruneFingerprintIssue(_ plan: [String: Any]) -> String? {
        guard isIntentPrunePlan(plan) else { return nil }

        if let suite = plan["suite"] as? [String: Any], !suite.isEmpty {
            let hasName = stringValue(suite["name"])?.isEmpty == false
            let hasSHA = stringValue(suite["sha256"])?.isEmpty == false
            let hasCount = intValue(suite["prompt_count"] ?? suite["promptCount"]) != nil
            if !hasName && !hasSHA && !hasCount {
                return "suite fingerprint is incomplete (need name, sha256, or prompt_count)"
            }
        }

        let stance = (stringValue(plan["safety_stance"] ?? plan["safetyStance"]) ?? "").lowercased()
        let crackPack = plan["crack_pack"] as? [String: Any]
        let crackNonEmpty = crackPack.map { !$0.isEmpty } ?? false

        if stance == "crack" {
            guard let crackPack, crackNonEmpty else {
                return "intent prune plan with safety_stance=crack is missing crack_pack fingerprint"
            }
            let hasName = stringValue(crackPack["name"])?.isEmpty == false
            let hasSHA = stringValue(crackPack["sha256"])?.isEmpty == false
            let hasCount = intValue(crackPack["prompt_count"] ?? crackPack["promptCount"]) != nil
            if !hasName {
                return "crack_pack is missing name"
            }
            if !hasSHA && !hasCount {
                return "crack_pack is missing sha256 or prompt_count fingerprint"
            }
        } else if crackNonEmpty, let crackPack {
            // Optional CRACK fingerprint: when present, require a usable shape.
            let hasName = stringValue(crackPack["name"])?.isEmpty == false
            let hasSHA = stringValue(crackPack["sha256"])?.isEmpty == false
            if !hasName && !hasSHA {
                return "crack_pack fingerprint is incomplete (need name or sha256)"
            }
        }
        return nil
    }

    private static func reviewedPrunePlanIssue(_ plan: [String: Any]) -> String? {
        guard let safety = plan["safety"] as? [String: Any] else {
            return "prune_plan.json is missing top-k safety evidence"
        }
        guard jsonBool(safety["passed"]) == true else {
            return "embedded safety block did not pass"
        }
        if let issues = safety["issues"] as? [String], !issues.isEmpty {
            return issues.joined(separator: " ")
        }
        guard let minimumActive = intValue(
            safety["minimum_active_experts_per_layer"] ?? safety["minimumActiveExpertsPerLayer"]
        ) else {
            return "safety block is missing minimum active experts"
        }
        guard let keep = declaredKeepExperts(in: plan) else {
            return "prune plan is missing keep experts per layer"
        }
        if minimumActive != keep {
            return "safety declares \(minimumActive) active experts but plan keeps \(keep)"
        }
        guard let trainedTopK = maxTrainedTopK(in: safety) else {
            return "safety block is missing trained top-k evidence"
        }
        if keep < trainedTopK {
            return "plan keeps \(keep) experts but trained top-k is \(trainedTopK)"
        }
        if let issue = intentPruneFingerprintIssue(plan) {
            return issue
        }
        guard let comparison = plan["comparison_summary"] as? [String: Any], !comparison.isEmpty else {
            return "prune_plan.json is missing embedded same-suite comparison evidence"
        }
        let tracedPromptCount = planTracedPromptCount(plan)
        if let issue = reviewedPruneComparisonGateIssue(
            comparison: comparison,
            tracedPromptCount: tracedPromptCount
        ) {
            return issue
        }
        if let issue = reviewedPrunePlanDropEvidenceIssue(plan: plan, comparison: comparison) {
            return issue
        }
        guard let evalIndex = plan["eval_index"] as? [String: Any], !evalIndex.isEmpty else {
            return "prune_plan.json is missing embedded per-prompt eval_index evidence"
        }
        if let issue = reviewedPruneEvalIndexIssue(
            index: evalIndex,
            comparedPromptCount: intValue(comparison["promptCount"] ?? comparison["prompt_count"]),
            tracedPromptCount: tracedPromptCount,
            comparison: comparison,
            sourceModelPath: stringValue(plan["source_model"] ?? plan["sourceModelPath"]),
            expectedLayerCount: expectedReviewedLayerCount(summary: plan)
        ) {
            return issue
        }
        // Intent plans use hybrid scorer evidence + suite gates; Expert Lab per-expert
        // atlas semantic rows are not required (and flat keep lists have no evidence).
        if !isIntentPrunePlan(plan), let issue = reviewedPruneSemanticEvidenceIssue(plan) {
            return issue
        }
        return nil
    }

    private static func reviewedPrunePlanSidecarConsistencyIssue(
        plan: [String: Any],
        prunedSource: URL
    ) -> String? {
        let summaryURL = prunedSource.appendingPathComponent("expert_lab_review_summary.json")
        guard let summary = readJSONObject(summaryURL) else {
            return "expert_lab_review_summary.json is unreadable for prune_plan.json consistency"
        }
        let embeddedSidecars = embeddedReviewSidecars(summary: summary, prunedSource: prunedSource)
        if let issue = embeddedSidecars.issue {
            return issue
        }
        let sidecars = embeddedSidecars.urls

        guard let planComparison = plan["comparison_summary"] as? [String: Any] else {
            return nil
        }
        guard let comparisonURL = sidecars["comparison_summary"],
              let sidecarComparison = readJSONObject(comparisonURL) else {
            return "expert_lab_comparison_summary.json is unreadable for prune_plan.json consistency"
        }
        if let issue = comparisonEvidenceConsistencyIssue(plan: planComparison, sidecar: sidecarComparison) {
            return "prune_plan.json embedded comparison_summary does not match expert_lab_comparison_summary.json: \(issue)"
        }

        guard let planEvalIndex = plan["eval_index"] as? [String: Any] else {
            return nil
        }
        guard let evalIndexURL = sidecars["eval_index"],
              let sidecarEvalIndex = readJSONObject(evalIndexURL) else {
            return "expert_lab_eval_index.json is unreadable for prune_plan.json consistency"
        }
        if let issue = evalIndexEvidenceConsistencyIssue(plan: planEvalIndex, sidecar: sidecarEvalIndex) {
            return "prune_plan.json embedded eval_index does not match expert_lab_eval_index.json: \(issue)"
        }
        return nil
    }

    private static func comparisonEvidenceConsistencyIssue(
        plan: [String: Any],
        sidecar: [String: Any]
    ) -> String? {
        if let issue = intConsistencyIssue(
            label: "prompt count",
            plan: plan["promptCount"] ?? plan["prompt_count"],
            sidecar: sidecar["promptCount"] ?? sidecar["prompt_count"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "baseline pass rate",
            plan: plan["passRateBaseline"] ?? plan["pass_rate_baseline"],
            sidecar: sidecar["passRateBaseline"] ?? sidecar["pass_rate_baseline"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "masked pass rate",
            plan: plan["passRateMasked"] ?? plan["pass_rate_masked"],
            sidecar: sidecar["passRateMasked"] ?? sidecar["pass_rate_masked"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "mean text delta",
            plan: plan["meanTextDelta"] ?? plan["mean_text_delta"],
            sidecar: sidecar["meanTextDelta"] ?? sidecar["mean_text_delta"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "mean latency delta",
            plan: plan["meanLatencyDeltaPct"] ?? plan["mean_latency_delta_pct"],
            sidecar: sidecar["meanLatencyDeltaPct"] ?? sidecar["mean_latency_delta_pct"]
        ) {
            return issue
        }
        if let issue = stringSetConsistencyIssue(
            label: "high-risk domains",
            plan: plan["highRiskDomains"] ?? plan["high_risk_domains"],
            sidecar: sidecar["highRiskDomains"] ?? sidecar["high_risk_domains"]
        ) {
            return issue
        }
        if let issue = intConsistencyIssue(
            label: "validator-available prompt count",
            plan: plan["validatorAvailablePromptCount"] ?? plan["validator_available_prompt_count"],
            sidecar: sidecar["validatorAvailablePromptCount"] ?? sidecar["validator_available_prompt_count"]
        ) {
            return issue
        }
        if let issue = intConsistencyIssue(
            label: "baseline-qualified prompt count",
            plan: plan["baselineQualifiedPromptCount"] ?? plan["baseline_qualified_prompt_count"],
            sidecar: sidecar["baselineQualifiedPromptCount"] ?? sidecar["baseline_qualified_prompt_count"]
        ) {
            return issue
        }
        if let issue = stringArrayExactConsistencyIssue(
            label: "baseline-qualified prompt IDs",
            plan: plan["baselineQualifiedPromptIDs"] ?? plan["baseline_qualified_prompt_ids"],
            sidecar: sidecar["baselineQualifiedPromptIDs"] ?? sidecar["baseline_qualified_prompt_ids"]
        ) {
            return issue
        }
        if let issue = stringArrayExactConsistencyIssue(
            label: "degraded prompt IDs",
            plan: plan["degradedPromptIDs"] ?? plan["degraded_prompt_ids"],
            sidecar: sidecar["degradedPromptIDs"] ?? sidecar["degraded_prompt_ids"]
        ) {
            return issue
        }
        if let issue = doubleConsistencyIssue(
            label: "baseline-qualified masked pass rate",
            plan: plan["baselineQualifiedMaskedPassRate"] ?? plan["baseline_qualified_masked_pass_rate"],
            sidecar: sidecar["baselineQualifiedMaskedPassRate"] ?? sidecar["baseline_qualified_masked_pass_rate"]
        ) {
            return issue
        }
        if let issue = stringSetConsistencyIssue(
            label: "baseline-qualified semantic coverage",
            plan: plan["baselineQualifiedSemanticCoverage"] ?? plan["baseline_qualified_semantic_coverage"],
            sidecar: sidecar["baselineQualifiedSemanticCoverage"] ?? sidecar["baseline_qualified_semantic_coverage"]
        ) {
            return issue
        }
        if let issue = stringSetConsistencyIssue(
            label: "missing baseline-qualified semantic coverage",
            plan: plan["missingBaselineQualifiedSemanticCoverage"] ?? plan["missing_baseline_qualified_semantic_coverage"],
            sidecar: sidecar["missingBaselineQualifiedSemanticCoverage"] ?? sidecar["missing_baseline_qualified_semantic_coverage"]
        ) {
            return issue
        }
        if let issue = stringConsistencyIssue(
            label: "regression severity",
            plan: plan["regressionSeverity"] ?? plan["regression_severity"],
            sidecar: sidecar["regressionSeverity"] ?? sidecar["regression_severity"]
        ) {
            return issue
        }
        return coordinateSetConsistencyIssue(
            label: "safe-drop candidates",
            plan: plan["safeDropCandidates"] ?? plan["safe_drop_candidates"],
            sidecar: sidecar["safeDropCandidates"] ?? sidecar["safe_drop_candidates"]
        )
    }

    private static func evalIndexEvidenceConsistencyIssue(
        plan: [String: Any],
        sidecar: [String: Any]
    ) -> String? {
        let checks: [String?] = [
            intConsistencyIssue(
                label: "prompt count",
                plan: plan["prompt_count"] ?? plan["promptCount"],
                sidecar: sidecar["prompt_count"] ?? sidecar["promptCount"]
            ),
            stringArrayExactConsistencyIssue(
                label: "prompt IDs",
                plan: plan["prompt_ids"] ?? plan["promptIDs"],
                sidecar: sidecar["prompt_ids"] ?? sidecar["promptIDs"]
            ),
            stringSetConsistencyIssue(
                label: "risky prompt IDs",
                plan: plan["risky_prompt_ids"] ?? plan["riskyPromptIDs"],
                sidecar: sidecar["risky_prompt_ids"] ?? sidecar["riskyPromptIDs"]
            ),
            stringSetConsistencyIssue(
                label: "high-risk domains",
                plan: plan["high_risk_domains"] ?? plan["highRiskDomains"],
                sidecar: sidecar["high_risk_domains"] ?? sidecar["highRiskDomains"]
            ),
            stringConsistencyIssue(
                label: "validator schema",
                plan: plan["validator_schema"] ?? plan["validatorSchema"],
                sidecar: sidecar["validator_schema"] ?? sidecar["validatorSchema"]
            ),
            intConsistencyIssue(
                label: "validator-available prompt count",
                plan: plan["validator_available_prompt_count"] ?? plan["validatorAvailablePromptCount"],
                sidecar: sidecar["validator_available_prompt_count"] ?? sidecar["validatorAvailablePromptCount"]
            ),
            intConsistencyIssue(
                label: "baseline-qualified prompt count",
                plan: plan["baseline_qualified_prompt_count"] ?? plan["baselineQualifiedPromptCount"],
                sidecar: sidecar["baseline_qualified_prompt_count"] ?? sidecar["baselineQualifiedPromptCount"]
            ),
            stringArrayExactConsistencyIssue(
                label: "baseline-qualified prompt IDs",
                plan: plan["baseline_qualified_prompt_ids"] ?? plan["baselineQualifiedPromptIDs"],
                sidecar: sidecar["baseline_qualified_prompt_ids"] ?? sidecar["baselineQualifiedPromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "baseline-invalid prompt IDs",
                plan: plan["baseline_invalid_prompt_ids"] ?? plan["baselineInvalidPromptIDs"],
                sidecar: sidecar["baseline_invalid_prompt_ids"] ?? sidecar["baselineInvalidPromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "inconclusive prompt IDs",
                plan: plan["inconclusive_prompt_ids"] ?? plan["inconclusivePromptIDs"],
                sidecar: sidecar["inconclusive_prompt_ids"] ?? sidecar["inconclusivePromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "preserved prompt IDs",
                plan: plan["preserved_prompt_ids"] ?? plan["preservedPromptIDs"],
                sidecar: sidecar["preserved_prompt_ids"] ?? sidecar["preservedPromptIDs"]
            ),
            stringArrayExactConsistencyIssue(
                label: "degraded prompt IDs",
                plan: plan["degraded_prompt_ids"] ?? plan["degradedPromptIDs"],
                sidecar: sidecar["degraded_prompt_ids"] ?? sidecar["degradedPromptIDs"]
            ),
            doubleConsistencyIssue(
                label: "baseline-qualified masked pass rate",
                plan: plan["baseline_qualified_masked_pass_rate"] ?? plan["baselineQualifiedMaskedPassRate"],
                sidecar: sidecar["baseline_qualified_masked_pass_rate"] ?? sidecar["baselineQualifiedMaskedPassRate"]
            ),
            stringSetConsistencyIssue(
                label: "baseline-qualified semantic coverage",
                plan: plan["baseline_qualified_semantic_coverage"] ?? plan["baselineQualifiedSemanticCoverage"],
                sidecar: sidecar["baseline_qualified_semantic_coverage"] ?? sidecar["baselineQualifiedSemanticCoverage"]
            ),
            stringSetConsistencyIssue(
                label: "missing baseline-qualified semantic coverage",
                plan: plan["missing_baseline_qualified_semantic_coverage"] ?? plan["missingBaselineQualifiedSemanticCoverage"],
                sidecar: sidecar["missing_baseline_qualified_semantic_coverage"] ?? sidecar["missingBaselineQualifiedSemanticCoverage"]
            ),
            stringSetConsistencyIssue(
                label: "semantic coverage",
                plan: plan["semantic_coverage"] ?? plan["semanticCoverage"],
                sidecar: sidecar["semantic_coverage"] ?? sidecar["semanticCoverage"]
            ),
            stringSetConsistencyIssue(
                label: "missing semantic coverage",
                plan: plan["missing_semantic_coverage"] ?? plan["missingSemanticCoverage"],
                sidecar: sidecar["missing_semantic_coverage"] ?? sidecar["missingSemanticCoverage"]
            ),
            doubleConsistencyIssue(
                label: "minimum baseline tokens",
                plan: plan["min_baseline_tokens"] ?? plan["minBaselineTokens"],
                sidecar: sidecar["min_baseline_tokens"] ?? sidecar["minBaselineTokens"]
            ),
            doubleConsistencyIssue(
                label: "minimum masked tokens",
                plan: plan["min_masked_tokens"] ?? plan["minMaskedTokens"],
                sidecar: sidecar["min_masked_tokens"] ?? sidecar["minMaskedTokens"]
            ),
            doubleConsistencyIssue(
                label: "mean baseline tokens",
                plan: plan["mean_baseline_tokens"] ?? plan["meanBaselineTokens"],
                sidecar: sidecar["mean_baseline_tokens"] ?? sidecar["meanBaselineTokens"]
            ),
            doubleConsistencyIssue(
                label: "mean masked tokens",
                plan: plan["mean_masked_tokens"] ?? plan["meanMaskedTokens"],
                sidecar: sidecar["mean_masked_tokens"] ?? sidecar["meanMaskedTokens"]
            ),
            intConsistencyIssue(
                label: "baseline route record count",
                plan: plan["baseline_route_record_count"] ?? plan["baselineRouteRecordCount"],
                sidecar: sidecar["baseline_route_record_count"] ?? sidecar["baselineRouteRecordCount"]
            ),
            intConsistencyIssue(
                label: "masked route record count",
                plan: plan["masked_route_record_count"] ?? plan["maskedRouteRecordCount"],
                sidecar: sidecar["masked_route_record_count"] ?? sidecar["maskedRouteRecordCount"]
            ),
            boolConsistencyIssue(
                label: "generation settings checked",
                plan: plan["generation_settings_checked"] ?? plan["generationSettingsChecked"],
                sidecar: sidecar["generation_settings_checked"] ?? sidecar["generationSettingsChecked"]
            ),
            stringConsistencyIssue(
                label: "runtime mode",
                plan: plan["runtime_mode"] ?? plan["runtimeMode"],
                sidecar: sidecar["runtime_mode"] ?? sidecar["runtimeMode"]
            ),
            stringConsistencyIssue(
                label: "runtime backend",
                plan: plan["runtime_backend"] ?? plan["runtimeBackend"],
                sidecar: sidecar["runtime_backend"] ?? sidecar["runtimeBackend"]
            ),
            stringConsistencyIssue(
                label: "runtime device",
                plan: plan["runtime_device"] ?? plan["runtimeDevice"],
                sidecar: sidecar["runtime_device"] ?? sidecar["runtimeDevice"]
            ),
            boolConsistencyIssue(
                label: "runtime Metal flag",
                plan: plan["runtime_metal_enabled"] ?? plan["runtimeMetalEnabled"],
                sidecar: sidecar["runtime_metal_enabled"] ?? sidecar["runtimeMetalEnabled"]
            ),
            boolConsistencyIssue(
                label: "hook coverage flag",
                plan: plan["hook_coverage_complete"] ?? plan["hookCoverageComplete"],
                sidecar: sidecar["hook_coverage_complete"] ?? sidecar["hookCoverageComplete"]
            ),
            intConsistencyIssue(
                label: "hooked MoE layers",
                plan: plan["hooked_moe_layers"] ?? plan["hookedMOELayers"],
                sidecar: sidecar["hooked_moe_layers"] ?? sidecar["hookedMOELayers"]
            ),
            intConsistencyIssue(
                label: "expected MoE layers",
                plan: plan["expected_moe_layers"] ?? plan["expectedMOELayers"],
                sidecar: sidecar["expected_moe_layers"] ?? sidecar["expectedMOELayers"]
            ),
            stringConsistencyIssue(
                label: "JANG tools version",
                plan: plan["jang_tools_version"] ?? plan["jangToolsVersion"],
                sidecar: sidecar["jang_tools_version"] ?? sidecar["jangToolsVersion"]
            ),
            stringConsistencyIssue(
                label: "MLX version",
                plan: plan["mlx_version"] ?? plan["mlxVersion"],
                sidecar: sidecar["mlx_version"] ?? sidecar["mlxVersion"]
            ),
            stringConsistencyIssue(
                label: "MLX-LM version",
                plan: plan["mlx_lm_version"] ?? plan["mlxLMVersion"],
                sidecar: sidecar["mlx_lm_version"] ?? sidecar["mlxLMVersion"]
            ),
            normalizedPathConsistencyIssue(
                label: "source model path",
                plan: plan["source_model_path"] ?? plan["sourceModelPath"],
                sidecar: sidecar["source_model_path"] ?? sidecar["sourceModelPath"]
            ),
            boolConsistencyIssue(
                label: "mask applied flag",
                plan: plan["mask_applied"] ?? plan["maskApplied"],
                sidecar: sidecar["mask_applied"] ?? sidecar["maskApplied"]
            ),
            intConsistencyIssue(
                label: "disabled expert count",
                plan: plan["disabled_expert_count"] ?? plan["disabledExpertCount"],
                sidecar: sidecar["disabled_expert_count"] ?? sidecar["disabledExpertCount"]
            ),
            intConsistencyIssue(
                label: "top-k override",
                plan: plan["top_k_override"] ?? plan["topKOverride"],
                sidecar: sidecar["top_k_override"] ?? sidecar["topKOverride"]
            )
        ]
        return checks.first { $0 != nil } ?? nil
    }

    private static func intConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = intValue(plan),
              let sidecarValue = intValue(sidecar) else {
            return nil
        }
        return planValue == sidecarValue ? nil : "\(label) differs"
    }

    private static func doubleConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = doubleValue(plan),
              let sidecarValue = doubleValue(sidecar) else {
            return nil
        }
        return doubleEqual(planValue, sidecarValue) ? nil : "\(label) differs"
    }

    private static func boolConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = jsonBool(plan),
              let sidecarValue = jsonBool(sidecar) else {
            return nil
        }
        return planValue == sidecarValue ? nil : "\(label) differs"
    }

    private static func stringConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = trimmedString(plan),
              let sidecarValue = trimmedString(sidecar) else {
            return nil
        }
        return planValue == sidecarValue ? nil : "\(label) differs"
    }

    private static func normalizedPathConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValue = trimmedString(plan),
              let sidecarValue = trimmedString(sidecar) else {
            return nil
        }
        return normalizedPath(planValue) == normalizedPath(sidecarValue) ? nil : "\(label) differs"
    }

    private static func stringSetConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValues = stringArrayValue(plan),
              let sidecarValues = stringArrayValue(sidecar) else {
            return nil
        }
        return Set(planValues) == Set(sidecarValues) ? nil : "\(label) differ"
    }

    private static func stringArrayExactConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planValues = stringArrayValue(plan),
              let sidecarValues = stringArrayValue(sidecar) else {
            return nil
        }
        return planValues == sidecarValues ? nil : "\(label) differ"
    }

    private static func coordinateSetConsistencyIssue(label: String, plan: Any?, sidecar: Any?) -> String? {
        guard let planCoordinates = coordinateSet(plan),
              let sidecarCoordinates = coordinateSet(sidecar) else {
            return nil
        }
        return planCoordinates == sidecarCoordinates ? nil : "\(label) differ"
    }

    private static func coordinateSet(_ value: Any?) -> Set<String>? {
        guard let rows = value as? [Any] else { return nil }
        var coordinates = Set<String>()
        for row in rows {
            guard let object = row as? [String: Any],
                  let layer = intValue(object["layer"] ?? object["layer_id"] ?? object["layerID"]),
                  let expert = intValue(object["expert"] ?? object["expert_id"] ?? object["expertID"]) else {
                return nil
            }
            coordinates.insert("\(layer):\(expert)")
        }
        return coordinates
    }

    private static func comparisonSafeDropMaskIssue(
        comparison: [String: Any],
        maskURL: URL?
    ) -> String? {
        guard let safeDrops = coordinateSet(comparison["safeDropCandidates"] ?? comparison["safe_drop_candidates"]),
              !safeDrops.isEmpty else {
            return nil
        }
        guard let maskURL,
              let disabled = disabledCoordinateSet(fromMaskURL: maskURL) else {
            return "mask.json is unreadable"
        }
        guard !disabled.isEmpty else {
            return "mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning"
        }
        if safeDrops != disabled {
            return "comparison_summary.json safe-drop candidates do not match mask.json disabled experts: safe \(previewCoordinates(safeDrops)); mask \(previewCoordinates(disabled))"
        }
        return nil
    }

    private static func reviewedPrunePlanDropEvidenceIssue(
        plan: [String: Any],
        comparison: [String: Any]
    ) -> String? {
        guard let safeDrops = coordinateSet(comparison["safeDropCandidates"] ?? comparison["safe_drop_candidates"]) else {
            return nil
        }
        guard let plannedDrops = planDropCoordinateSet(plan) else {
            return "prune_plan.json planned drop list is unreadable"
        }
        let unsafeDrops = plannedDrops.subtracting(safeDrops)
        if !unsafeDrops.isEmpty {
            return "prune_plan.json drops experts outside same-suite safe-drop candidates: \(previewCoordinates(unsafeDrops))"
        }
        return nil
    }

    private static func disabledCoordinateSet(fromMaskURL url: URL) -> Set<String>? {
        guard let disabledByLayer = disabledExpertsByLayer(fromMaskURL: url) else { return nil }
        return Set(disabledByLayer.flatMap { layer, experts in
            experts.map { "\(layer):\($0)" }
        })
    }

    private static func planSourceExpertCount(_ plan: [String: Any], layer: [String: Any]? = nil) -> Int? {
        if let n = intValue(plan["num_experts_source"] ?? plan["numExpertsSource"] ?? plan["sourceNumExperts"]) {
            return n
        }
        if let layer {
            return intValue(layer["num_source_experts"] ?? layer["numSourceExperts"])
        }
        return nil
    }

    private static func planLayerKeepList(_ value: Any) -> (layerObject: [String: Any]?, keep: [Int])? {
        if let layer = value as? [String: Any] {
            guard let rawKeep = layer["keep"] as? [Any] else { return nil }
            let keep = rawKeep.compactMap(intValue)
            guard keep.count == rawKeep.count else { return nil }
            return (layer, keep)
        }
        if let rawKeep = value as? [Any] {
            // jang-intent-prune-plan-v1: layers map to flat keep lists
            let keep = rawKeep.compactMap(intValue)
            guard keep.count == rawKeep.count else { return nil }
            return (nil, keep)
        }
        return nil
    }

    private static func planDropCoordinateSet(_ plan: [String: Any]) -> Set<String>? {
        guard let layers = plan["layers"] as? [String: Any] else { return nil }
        var coordinates = Set<String>()
        for key in layers.keys {
            guard let value = layers[key],
                  let parsed = planLayerKeepList(value),
                  let layerID = intValue(parsed.layerObject?["layer"]) ?? intValue(key) else {
                return nil
            }
            if let drops = parsed.layerObject?["drop"] as? [Any] {
                for drop in drops {
                    guard let expert = intValue(drop) else { return nil }
                    coordinates.insert("\(layerID):\(expert)")
                }
                continue
            }
            // Flat keep lists (intent plans) or keep-only objects: derive drops from source expert count.
            guard let numExperts = planSourceExpertCount(plan, layer: parsed.layerObject), numExperts > 0 else {
                return nil
            }
            let keepSet = Set(parsed.keep)
            for expert in 0..<numExperts where !keepSet.contains(expert) {
                coordinates.insert("\(layerID):\(expert)")
            }
        }
        return coordinates
    }

    private static func previewCoordinates(_ coordinates: Set<String>) -> String {
        let display = coordinates.sorted().map { coordinate -> String in
            let parts = coordinate.split(separator: ":", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { return coordinate }
            return "L\(parts[0]) E\(parts[1])"
        }
        let head = display.prefix(5).joined(separator: ", ")
        let remaining = max(0, display.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func reviewedPruneSemanticEvidenceIssue(_ plan: [String: Any]) -> String? {
        guard let layers = plan["layers"] as? [String: Any], !layers.isEmpty else {
            return "prune_plan.json is missing layer evidence rows"
        }
        var checkedRows = 0
        for layerKey in layers.keys.sorted() {
            guard let layer = layers[layerKey] as? [String: Any] else {
                return "prune_plan.json layer \(layerKey) is unreadable"
            }
            guard let evidenceRows = layer["evidence"] as? [[String: Any]], !evidenceRows.isEmpty else {
                return "prune_plan.json layer \(layerKey) is missing expert evidence rows"
            }
            for row in evidenceRows {
                let label = stringValue(row["label"]) ?? ""
                let normalizedLabel = label.lowercased()
                let hits = intValue(row["hits"]) ?? 0
                let domains = row["domains"] as? [String: Any] ?? [:]
                let isUnobserved = normalizedLabel.contains("unobserved") && hits == 0 && domains.isEmpty
                if isUnobserved { continue }

                checkedRows += 1
                let layerID = stringValue(layer["layer"]) ?? layerKey
                let expertID = stringValue(row["expert"]) ?? "?"
                let coordinate = "L\(layerID) E\(expertID)"

                guard doubleValue(row["router_mass"] ?? row["routerMass"] ?? row["probabilityMass"] ?? row["probability_mass"]) != nil else {
                    return "prune_plan.json evidence row \(coordinate) is missing gate mass evidence"
                }
                guard doubleValue(row["ablation_delta"] ?? row["ablationDelta"]) != nil else {
                    return "prune_plan.json evidence row \(coordinate) is missing masked-output impact evidence"
                }
                guard let impactScope = stringValue(row["masked_impact_scope"] ?? row["maskedImpactScope"]),
                      !impactScope.isEmpty else {
                    return "prune_plan.json evidence row \(coordinate) is missing masked-output impact scope evidence"
                }
                guard jsonBool(row["reviewed_mask_member"] ?? row["reviewedMaskMember"]) != nil else {
                    return "prune_plan.json evidence row \(coordinate) is missing reviewed mask membership evidence"
                }
                guard let domainLift = row["domain_lift"] as? [String: Any],
                      domainLift.contains(where: { doubleValue($0.value) != nil }) else {
                    return "prune_plan.json evidence row \(coordinate) is missing activation lift evidence"
                }
                guard let promptEvidence = row["prompt_evidence"] as? [[String: Any]], !promptEvidence.isEmpty else {
                    return "prune_plan.json evidence row \(coordinate) is missing prompt example evidence"
                }
                let hasPromptProof = promptEvidence.contains { prompt in
                    stringValue(prompt["promptID"] ?? prompt["prompt_id"]) != nil &&
                    stringValue(prompt["domain"]) != nil &&
                    stringValue(prompt["promptExcerpt"] ?? prompt["prompt_excerpt"]) != nil &&
                    (prompt["tags"] as? [Any])?.isEmpty == false &&
                    (intValue(prompt["hits"]) ?? 0) > 0
                }
                if !hasPromptProof {
                    return "prune_plan.json evidence row \(coordinate) has incomplete prompt tags/examples"
                }
            }
        }
        return checkedRows == 0 ? "prune_plan.json has no semantic expert evidence rows" : nil
    }

    private static func declaredKeepExperts(in plan: [String: Any]) -> Int? {
        if let keep = intValue(plan["keepExpertsPerLayer"] ?? plan["keep_experts_per_layer"]) {
            return keep
        }
        if let target = plan["target"] as? [String: Any],
           let keep = intValue(target["keep_experts_per_layer"] ?? target["keepExpertsPerLayer"]) {
            return keep
        }
        guard let layers = plan["layers"] as? [String: Any] else { return nil }
        let keepCounts = Set(layers.values.compactMap { value -> Int? in
            planLayerKeepList(value)?.keep.count
        })
        return keepCounts.count == 1 ? keepCounts.first : nil
    }

    private static func maxTrainedTopK(in safety: [String: Any]) -> Int? {
        // jang-intent-prune-plan-v1 uses scalar trained_top_k; Expert Lab uses by-layer maps.
        if let scalar = intValue(safety["trained_top_k"] ?? safety["trainedTopK"]) {
            return scalar
        }
        let raw = safety["trained_top_k_by_layer"] ?? safety["trainedTopKByLayer"]
        guard let topKByLayer = raw as? [String: Any] else { return nil }
        let values = topKByLayer.values.compactMap(intValue)
        return values.isEmpty ? nil : values.max()
    }

    private static func reviewedPruneComparisonGateIssue(
        comparison: [String: Any],
        tracedPromptCount: Int?
    ) -> String? {
        let promptCount = intValue(comparison["promptCount"] ?? comparison["prompt_count"]) ?? 0
        if promptCount < minimumReviewedPrunePromptCount {
            return "compare at least \(minimumReviewedPrunePromptCount) prompts before final quantization"
        }
        if let tracedPromptCount,
           tracedPromptCount > 0,
           promptCount != tracedPromptCount {
            return "rerun A/B compare for all \(tracedPromptCount) traced prompts"
        }
        if let highRiskDomains = comparison["highRiskDomains"] as? [String],
           !highRiskDomains.isEmpty {
            return "masked outputs regressed in high-risk domains: \(highRiskDomains.sorted().joined(separator: ", "))"
        }
        if let highRiskDomains = comparison["high_risk_domains"] as? [String],
           !highRiskDomains.isEmpty {
            return "masked outputs regressed in high-risk domains: \(highRiskDomains.sorted().joined(separator: ", "))"
        }
        if isBlockingRegressionSeverity(comparison["regressionSeverity"] ?? comparison["regression_severity"]) {
            return "masked comparison regression severity is high or critical"
        }
        let safeDropCandidates = comparison["safeDropCandidates"] ?? comparison["safe_drop_candidates"]
        guard let candidates = safeDropCandidates as? [Any] else {
            return "comparison summary is missing A/B-safe candidates"
        }
        if candidates.isEmpty {
            return "A/B comparison found no safe drop candidates"
        }
        if let issue = reviewedPruneComparisonValidatorIssue(comparison) {
            return issue
        }
        return nil
    }

    private static func reviewedPruneComparisonValidatorIssue(_ comparison: [String: Any]) -> String? {
        guard intValue(comparison["validatorAvailablePromptCount"] ?? comparison["validator_available_prompt_count"]) != nil,
              dictionaryValue(comparison["classificationCounts"] ?? comparison["prompt_classification_counts"]) != nil else {
            return "comparison summary is missing validator classification evidence"
        }
        guard let baselineQualified = intValue(
            comparison["baselineQualifiedPromptCount"] ?? comparison["baseline_qualified_prompt_count"]
        ),
              baselineQualified > 0 else {
            return "comparison summary has no baseline-qualified validator prompts"
        }
        if let baselineQualifiedIDs = stringArrayValue(
            comparison["baselineQualifiedPromptIDs"] ?? comparison["baseline_qualified_prompt_ids"]
        ),
            baselineQualifiedIDs.count != baselineQualified {
            return "comparison summary baseline-qualified prompt IDs do not match the baseline-qualified count"
        }
        let missingCoverage = stringArrayValue(
            comparison["missingBaselineQualifiedSemanticCoverage"]
                ?? comparison["missing_baseline_qualified_semantic_coverage"]
        ) ?? []
        if !missingCoverage.isEmpty {
            return "baseline-qualified prompts are missing semantic coverage: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        let degraded = stringArrayValue(comparison["degradedPromptIDs"] ?? comparison["degraded_prompt_ids"]) ?? []
        if !degraded.isEmpty {
            return "baseline-qualified prompts degraded after masking: \(previewIDs(Set(degraded)))"
        }
        guard let coverage = stringArrayValue(
            comparison["baselineQualifiedSemanticCoverage"]
                ?? comparison["baseline_qualified_semantic_coverage"]
        ),
              !coverage.isEmpty else {
            return "comparison summary is missing baseline-qualified semantic coverage evidence"
        }
        if let passRate = doubleValue(
            comparison["baselineQualifiedMaskedPassRate"] ?? comparison["baseline_qualified_masked_pass_rate"]
        ),
            passRate < 1.0 {
            return "masked validator pass rate is below 100% on baseline-qualified prompts"
        }
        return nil
    }

    private static func comparisonPromptCount(from summary: [String: Any]) -> Int? {
        guard let prunedSourcePath = stringValue(summary["pruned_source"]),
              let comparisonURL = embeddedSidecarURL(
                summary["comparison_summary"],
                prunedSource: URL(fileURLWithPath: prunedSourcePath),
                fallbackName: "expert_lab_comparison_summary.json"
              ),
              let comparison = readJSONObject(comparisonURL) else {
            return nil
        }
        return intValue(comparison["promptCount"] ?? comparison["prompt_count"])
    }

    private static func embeddedReviewSidecars(
        summary: [String: Any],
        prunedSource: URL
    ) -> (urls: [String: URL], issue: String?) {
        let specs: [(key: String, value: Any?, fileName: String)] = [
            ("suite_jsonl", summary["suite_jsonl"], "expert_lab_suite.jsonl"),
            ("comparison_summary", summary["comparison_summary"], "expert_lab_comparison_summary.json"),
            ("eval_jsonl", summary["eval_jsonl"], "expert_lab_eval.jsonl"),
            ("eval_trace_jsonl", summary["eval_trace_jsonl"], "expert_lab_eval_trace.jsonl"),
            ("eval_index", summary["eval_index"], "expert_lab_eval_index.json"),
            ("mask_json", summary["mask_json"] ?? summary["mask"] ?? summary["maskJSON"], "mask.json"),
        ]
        var urls: [String: URL] = [:]
        var outside: [String] = []
        for spec in specs {
            if let url = embeddedSidecarURL(spec.value, prunedSource: prunedSource, fallbackName: spec.fileName) {
                urls[spec.key] = url
            } else {
                outside.append(spec.key)
            }
        }
        if !outside.isEmpty {
            return (
                urls,
                "Expert Lab review summary sidecar paths must be embedded in the pruned BF16/F16 source: \(outside.joined(separator: ", "))"
            )
        }
        return (urls, nil)
    }

    private static func embeddedSidecarURL(_ value: Any?, prunedSource: URL, fallbackName: String) -> URL? {
        let expected = prunedSource.appendingPathComponent(fallbackName)
        guard let raw = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !raw.isEmpty else {
            return expected
        }
        let expanded = (raw as NSString).expandingTildeInPath
        let recorded = (expanded as NSString).isAbsolutePath
            ? URL(fileURLWithPath: expanded)
            : prunedSource.appendingPathComponent(expanded)
        return canonicalPath(recorded.path) == canonicalPath(expected.path) ? expected : nil
    }

    private static func reviewedPruneEvalIndexIssue(
        index: [String: Any],
        comparedPromptCount: Int?,
        tracedPromptCount: Int?,
        comparison: [String: Any]? = nil,
        suiteURL: URL? = nil,
        evalURL: URL? = nil,
        evalTraceURL: URL? = nil,
        maskURL: URL? = nil,
        sourceModelPath: String? = nil,
        expectedLayerCount: Int? = nil
    ) -> String? {
        let promptCount = intValue(index["prompt_count"] ?? index["promptCount"]) ?? 0
        let promptIDs = stringArrayValue(index["prompt_ids"] ?? index["promptIDs"]) ?? []
        if promptIDs.count != promptCount {
            return "eval_index.json lists \(promptIDs.count) prompt IDs for \(promptCount) indexed prompts"
        }
        if Set(promptIDs).count < promptIDs.count {
            return "eval_index.json contains duplicate prompt IDs"
        }
        if let issue = evalIndexSemanticCoverageIssue(index) {
            return issue
        }
        let indexedIDs = Set(promptIDs)
        var evalRowsForDecodeSettings: [[String: Any]]?
        if let suiteURL {
            guard let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]) else {
                return "suite.jsonl prompt IDs are unreadable"
            }
            if Set(suiteIDs).count < suiteIDs.count {
                return "suite.jsonl contains duplicate prompt IDs"
            }
            if let issue = suiteSemanticCoverageIssue(suiteURL) {
                return issue
            }
            let suiteSet = Set(suiteIDs)
            let missing = suiteSet.subtracting(indexedIDs)
            if !missing.isEmpty {
                return "eval_index.json prompt IDs missing suite.jsonl prompts: \(previewIDs(missing))"
            }
            let unexpected = indexedIDs.subtracting(suiteSet)
            if !unexpected.isEmpty {
                return "eval_index.json prompt IDs outside suite.jsonl: \(previewIDs(unexpected))"
            }
            if promptIDs != suiteIDs {
                return "eval_index.json prompt order does not match suite.jsonl"
            }
        }
        if let evalURL {
            guard let evalIDs = jsonlStringIDs(evalURL, keys: ["promptID", "prompt_id", "id"]) else {
                return "eval.jsonl prompt IDs are unreadable"
            }
            if Set(evalIDs).count < evalIDs.count {
                return "eval.jsonl contains duplicate prompt IDs"
            }
            let evalSet = Set(evalIDs)
            let missing = indexedIDs.subtracting(evalSet)
            let unexpected = evalSet.subtracting(indexedIDs)
            if !missing.isEmpty, !unexpected.isEmpty {
                return "eval.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpected)); eval_index.json prompt IDs missing from eval.jsonl: \(previewIDs(missing))"
            }
            if !missing.isEmpty {
                return "eval_index.json prompt IDs missing from eval.jsonl: \(previewIDs(missing))"
            }
            if !unexpected.isEmpty {
                return "eval.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpected))"
            }
            if evalIDs != promptIDs {
                return "eval.jsonl prompt order does not match eval_index.json"
            }
            guard let evalRows = jsonlObjects(evalURL) else {
                return "eval.jsonl is unreadable"
            }
            evalRowsForDecodeSettings = evalRows
            if let issue = evalRowEvidenceIssue(
                rows: evalRows,
                expectedPromptIDs: promptIDs,
                sourceModelPath: sourceModelPath
            ) {
                return issue
            }
        }
        let expectedDisabledByLayer: [Int: Set<Int>]?
        if let maskURL {
            guard let mask = disabledExpertsByLayer(fromMaskURL: maskURL) else {
                return "mask.json is unreadable"
            }
            let disabledCount = mask.values.reduce(0) { $0 + $1.count }
            guard disabledCount > 0 else {
                return "mask.json does not disable any experts; top-k-only comparisons cannot authorize hard pruning"
            }
            if let indexDisabledCount = intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
               indexDisabledCount != disabledCount {
                return "eval_index.json disabled expert count \(indexDisabledCount) does not match mask.json \(disabledCount)"
            }
            expectedDisabledByLayer = mask
        } else {
            expectedDisabledByLayer = nil
        }
        guard let baselineRouteRecordCount = intValue(index["baseline_route_record_count"] ?? index["baselineRouteRecordCount"]),
              let maskedRouteRecordCount = intValue(index["masked_route_record_count"] ?? index["maskedRouteRecordCount"]),
              baselineRouteRecordCount >= promptCount,
              maskedRouteRecordCount >= promptCount else {
            return "eval_index.json is missing routing record evidence for every indexed prompt"
        }
        if let evalTraceURL {
            guard let traceIDs = jsonlStringIDs(evalTraceURL, keys: ["promptID", "prompt_id", "id"]) else {
                return "eval_trace.jsonl prompt IDs are unreadable"
            }
            if traceIDs.isEmpty {
                return "eval_trace.jsonl has no routing records"
            }
            let traceSet = Set(traceIDs)
            let missing = indexedIDs.subtracting(traceSet)
            if !missing.isEmpty {
                return "eval_index.json prompt IDs missing from eval_trace.jsonl: \(previewIDs(missing))"
            }
            let unexpected = traceSet.subtracting(indexedIDs)
            if !unexpected.isEmpty {
                return "eval_trace.jsonl prompt IDs outside eval_index.json: \(previewIDs(unexpected))"
            }
            guard let traceRows = jsonlObjects(evalTraceURL) else {
                return "eval_trace.jsonl is unreadable"
            }
            if let issue = evalTraceVariantIssue(
                rows: traceRows,
                expectedPromptIDs: promptIDs,
                disabledExpertCount: intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
                topKOverride: intValue(index["top_k_override"] ?? index["topKOverride"]),
                expectedDisabledByLayer: expectedDisabledByLayer,
                expectedBaselineRouteRecordCount: baselineRouteRecordCount,
                expectedMaskedRouteRecordCount: maskedRouteRecordCount
            ) {
                return issue
            }
        }
        if let comparedPromptCount, promptCount != comparedPromptCount {
            return "eval_index.json covers \(promptCount) of \(comparedPromptCount) compared prompts"
        }
        if let tracedPromptCount, tracedPromptCount > 0, promptCount != tracedPromptCount {
            return "eval_index.json covers \(promptCount) of \(tracedPromptCount) traced prompts"
        }
        if let risky = index["risky_prompt_ids"] as? [Any], !risky.isEmpty {
            return "eval_index.json still has risky prompt IDs"
        }
        if let risky = index["riskyPromptIDs"] as? [Any], !risky.isEmpty {
            return "eval_index.json still has risky prompt IDs"
        }
        if isBlockingRegressionSeverity(index["regression_severity"] ?? index["regressionSeverity"]) {
            return "eval_index.json regression severity is high or critical"
        }
        if let highRiskDomains = index["high_risk_domains"] as? [Any], !highRiskDomains.isEmpty {
            return "eval_index.json still has high-risk domains"
        }
        if let highRiskDomains = index["highRiskDomains"] as? [Any], !highRiskDomains.isEmpty {
            return "eval_index.json still has high-risk domains"
        }
        if let issue = evalComparisonConsistencyIssue(
            comparison: comparison,
            index: index,
            evalRows: evalRowsForDecodeSettings
        ) {
            return issue
        }
        guard let meanBaselineTokens = doubleValue(index["mean_baseline_tokens"] ?? index["meanBaselineTokens"]),
              let meanMaskedTokens = doubleValue(index["mean_masked_tokens"] ?? index["meanMaskedTokens"]) else {
            return "eval_index.json is missing generation-depth token evidence"
        }
        let shallow = min(meanBaselineTokens, meanMaskedTokens)
        if shallow < minimumReviewedPruneMeanTokens {
            return String(
                format: "eval_index.json average generated depth %.1f tokens is below %.0f",
                shallow,
                minimumReviewedPruneMeanTokens
            )
        }
        if let issue = evalIndexLayerStatsCoverageIssue(index: index, promptCount: promptCount) {
            return issue
        }
        guard stringValue(index["eval_trace_jsonl"] ?? index["evalTraceJSONL"]) != nil else {
            return "eval_index.json is missing eval_trace.jsonl evidence"
        }
        guard let runtimeMode = index["runtime_mode"] as? String ?? index["runtimeMode"] as? String,
              !runtimeMode.isEmpty,
              let runtimeDevice = index["runtime_device"] as? String ?? index["runtimeDevice"] as? String,
              !runtimeDevice.isEmpty,
              let runtimeMetalEnabled = jsonBool(index["runtime_metal_enabled"] ?? index["runtimeMetalEnabled"]) else {
            return "eval_index.json is missing runtime device evidence"
        }
        if runtimeMetalEnabled != true {
            return "eval_index.json did not record a Metal runtime"
        }
        if runtimeMode != "bf16_vmlx" {
            return "eval_index.json did not record BF16/vMLX runtime evidence"
        }
        if stringValue(index["runtime_backend"] ?? index["runtimeBackend"]) != "vmlx" {
            return "eval_index.json did not record vMLX backend evidence"
        }
        if jsonBool(index["hook_coverage_complete"] ?? index["hookCoverageComplete"]) == false {
            return "eval_index.json recorded incomplete vMLX routed-layer hook coverage"
        }
        if let expectedLayerCount {
            guard let hookedLayers = intValue(index["hooked_moe_layers"] ?? index["hookedMOELayers"]) else {
                return "eval_index.json is missing vMLX routed-layer hook evidence"
            }
            if hookedLayers < expectedLayerCount {
                return "eval_index.json vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers"
            }
        }
        if let expectedMOELayers = intValue(index["expected_moe_layers"] ?? index["expectedMOELayers"]),
           let hookedLayers = intValue(index["hooked_moe_layers"] ?? index["hookedMOELayers"]),
           hookedLayers < expectedMOELayers {
            return "eval_index.json vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers"
        }
        guard let jangToolsVersion = stringValue(index["jang_tools_version"] ?? index["jangToolsVersion"]),
              !jangToolsVersion.isEmpty,
              let mlxVersion = stringValue(index["mlx_version"] ?? index["mlxVersion"]),
              !mlxVersion.isEmpty,
              let mlxLMVersion = stringValue(index["mlx_lm_version"] ?? index["mlxLMVersion"]),
              !mlxLMVersion.isEmpty else {
            return "eval_index.json is missing vMLX package version evidence"
        }
        guard let evalSourcePath = stringValue(index["source_model_path"] ?? index["sourceModelPath"]) else {
            return "eval_index.json is missing source model path evidence"
        }
        if let sourceModelPath,
           normalizedPath(evalSourcePath) != normalizedPath(sourceModelPath) {
            return "eval_index.json source model path does not match reviewed source"
        }
        guard jsonBool(index["mask_applied"] ?? index["maskApplied"]) == true else {
            return "eval_index.json did not record an applied BF16/vMLX mask"
        }
        guard let disabledExpertCount = intValue(index["disabled_expert_count"] ?? index["disabledExpertCount"]),
              disabledExpertCount > 0 else {
            return "eval_index.json did not record disabled expert evidence; top-k-only comparisons cannot authorize hard pruning"
        }
        if let evalRowsForDecodeSettings,
           let issue = evalDecodeSettingsIssue(index: index, rows: evalRowsForDecodeSettings) {
            return issue
        }
        if let issue = evalIndexValidatorEvidenceIssue(index) {
            return issue
        }
        if let suiteURL {
            guard let expectedSuiteSHA256 = fileSHA256(suiteURL) else {
                return "eval_index.json suite.jsonl fingerprint could not be computed"
            }
            guard let recordedSuiteSHA256 = stringValue(index["suite_sha256"] ?? index["suiteSHA256"]),
                  !recordedSuiteSHA256.isEmpty else {
                return "eval_index.json is missing suite.jsonl fingerprint evidence"
            }
            if recordedSuiteSHA256 != expectedSuiteSHA256 {
                return "eval_index.json suite.jsonl fingerprint does not match suite.jsonl"
            }
        }
        return nil
    }

    private static func evalRowEvidenceIssue(
        rows: [[String: Any]],
        expectedPromptIDs: [String],
        sourceModelPath: String?
    ) -> String? {
        if rows.count != expectedPromptIDs.count {
            return "eval.jsonl has \(rows.count) rows for \(expectedPromptIDs.count) indexed prompts"
        }
        let rowIDs = rows.compactMap { promptID(in: $0) }
        if rowIDs.count != rows.count {
            return "eval.jsonl prompt IDs are unreadable"
        }
        if rowIDs != expectedPromptIDs {
            return "eval.jsonl prompt order does not match eval_index.json"
        }
        if rows.contains(where: {
            trimmedString($0["baselineText"] ?? $0["baseline_text"]) == nil
                || trimmedString($0["maskedText"] ?? $0["masked_text"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt baseline/masked output text"
        }
        if rows.contains(where: {
            guard let textDelta = doubleValue($0["textDelta"] ?? $0["text_delta"]),
                  textDelta.isFinite else {
                return true
            }
            return false
        }) {
            return "eval.jsonl is missing per-prompt text delta evidence"
        }
        if rows.contains(where: {
            (intValue($0["baselineTokenCount"] ?? $0["baseline_token_count"]) ?? 0) <= 0
                || (intValue($0["maskedTokenCount"] ?? $0["masked_token_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt token count evidence"
        }
        if rows.contains(where: {
            (intValue($0["baselineRouteRecordCount"] ?? $0["baseline_route_record_count"]) ?? 0) <= 0
                || (intValue($0["maskedRouteRecordCount"] ?? $0["masked_route_record_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt routing record evidence"
        }
        if let issue = evalRowLayerStatsCoverageIssue(rows: rows) {
            return issue
        }
        if rows.contains(where: {
            trimmedString($0["runtimeMode"] ?? $0["runtime_mode"]) == nil
                || trimmedString($0["runtimeDevice"] ?? $0["runtime_device"]) == nil
                || jsonBool($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt runtime device evidence"
        }
        if rows.contains(where: {
            jsonBool($0["runtimeMetalEnabled"] ?? $0["runtime_metal_enabled"]) != true
        }) {
            return "eval.jsonl did not record a Metal runtime"
        }
        if rows.contains(where: {
            trimmedString($0["runtimeMode"] ?? $0["runtime_mode"]) != "bf16_vmlx"
        }) {
            return "eval.jsonl did not record BF16/vMLX runtime evidence"
        }
        if rows.contains(where: {
            trimmedString($0["runtimeBackend"] ?? $0["runtime_backend"]) != "vmlx"
        }) {
            return "eval.jsonl did not record per-prompt vMLX backend evidence"
        }
        if rows.contains(where: {
            trimmedString($0["jangToolsVersion"] ?? $0["jang_tools_version"]) == nil
                || trimmedString($0["mlxVersion"] ?? $0["mlx_version"]) == nil
                || trimmedString($0["mlxLMVersion"] ?? $0["mlx_lm_version"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt vMLX package version evidence"
        }
        if rows.contains(where: {
            trimmedString($0["sourceModelPath"] ?? $0["source_model_path"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt source model path evidence"
        }
        if let sourceModelPath {
            let expectedSourcePath = normalizedPath(sourceModelPath)
            if rows.contains(where: {
                normalizedPath(trimmedString($0["sourceModelPath"] ?? $0["source_model_path"]) ?? "") != expectedSourcePath
            }) {
                return "eval.jsonl source model path does not match reviewed source"
            }
        }
        if rows.contains(where: {
            jsonBool($0["maskApplied"] ?? $0["mask_applied"]) != true
        }) {
            return "eval.jsonl did not record an applied BF16/vMLX mask"
        }
        if rows.contains(where: {
            jsonBool($0["maskApplied"] ?? $0["mask_applied"]) == true
                && (intValue($0["disabledExpertCount"] ?? $0["disabled_expert_count"]) ?? 0) <= 0
        }) {
            return "eval.jsonl is missing per-prompt disabled expert evidence; top-k-only comparisons cannot authorize hard pruning"
        }
        if rows.contains(where: {
            trimmedString($0["risk"]) == nil
                || trimmedString($0["regressionSeverity"] ?? $0["regression_severity"]) == nil
        }) {
            return "eval.jsonl is missing per-prompt regression flag evidence"
        }
        if let issue = evalRowValidatorClassificationIssue(rows: rows) {
            return issue
        }
        return nil
    }

    private static func evalRowValidatorClassificationIssue(rows: [[String: Any]]) -> String? {
        for row in rows {
            guard trimmedString(row["validatorKind"] ?? row["validator_kind"]) != nil,
                  jsonBool(row["validatorAvailable"] ?? row["validator_available"]) != nil,
                  trimmedString(row["promptClassification"] ?? row["prompt_classification"]) != nil,
                  jsonBool(row["baselineQualified"] ?? row["baseline_qualified"]) != nil,
                  jsonBool(row["safeDropEvidenceEligible"] ?? row["safe_drop_evidence_eligible"]) != nil else {
                return "eval.jsonl is missing per-prompt validator classification evidence"
            }
            let classification = promptClassification(row)
            let baselinePassed = jsonBool(row["baselinePassed"] ?? row["baseline_passed"])
            let maskedPassed = jsonBool(row["maskedPassed"] ?? row["masked_passed"])
            switch classification {
            case "baseline_invalid":
                if baselinePassed != false {
                    return "eval.jsonl baseline_invalid row has inconsistent baseline validator result"
                }
            case "preserved":
                if baselinePassed != true || maskedPassed != true {
                    return "eval.jsonl preserved row has inconsistent validator results"
                }
            case "degraded":
                if baselinePassed != true || maskedPassed != false {
                    return "eval.jsonl degraded row has inconsistent validator results"
                }
            case "inconclusive":
                break
            default:
                return "eval.jsonl has unknown prompt classification \(classification)"
            }
        }
        return nil
    }

    private static func evalComparisonConsistencyIssue(
        comparison: [String: Any]?,
        index: [String: Any],
        evalRows: [[String: Any]]?
    ) -> String? {
        guard let comparison else { return nil }
        let comparisonPromptCount = intValue(comparison["promptCount"] ?? comparison["prompt_count"]) ?? 0
        if let indexPromptCount = intValue(index["prompt_count"] ?? index["promptCount"]),
           comparisonPromptCount != indexPromptCount {
            return "comparison summary prompt count does not match eval_index.json"
        }

        let comparisonHighRisk = stringArrayValue(comparison["highRiskDomains"] ?? comparison["high_risk_domains"]) ?? []
        let indexHighRisk = stringArrayValue(index["high_risk_domains"] ?? index["highRiskDomains"]) ?? []
        if Set(comparisonHighRisk) != Set(indexHighRisk) {
            return "comparison summary high-risk domains do not match eval_index.json"
        }

        if let comparisonSeverity = trimmedString(comparison["regressionSeverity"] ?? comparison["regression_severity"]),
           let indexSeverity = trimmedString(index["regression_severity"] ?? index["regressionSeverity"]),
           comparisonSeverity != indexSeverity {
            return "comparison summary regression severity does not match eval_index.json"
        }

        guard let evalRows else { return nil }
        if evalRows.count != comparisonPromptCount {
            return "comparison summary covers \(comparisonPromptCount) of \(evalRows.count) eval.jsonl rows"
        }
        let rowHighRisk = evalRowsHighRiskDomains(evalRows)
        if Set(comparisonHighRisk) != Set(rowHighRisk) {
            return "comparison summary high-risk domains do not match eval.jsonl"
        }
        let rowSeverity = evalRowsRegressionSeverity(evalRows)
        if rowSeverity != "none" {
            guard let comparisonSeverity = trimmedString(comparison["regressionSeverity"] ?? comparison["regression_severity"]) else {
                return "comparison summary is missing regression severity evidence"
            }
            if comparisonSeverity != rowSeverity {
                return "comparison summary regression severity does not match eval.jsonl"
            }
        }
        let rowTextDeltas = evalRows.compactMap {
            doubleValue($0["textDelta"] ?? $0["text_delta"])
        }
        if rowTextDeltas.count == evalRows.count,
           let summaryMean = doubleValue(comparison["meanTextDelta"] ?? comparison["mean_text_delta"]),
           !doubleEqual(summaryMean, mean(rowTextDeltas)) {
            return "comparison summary mean text delta does not match eval.jsonl"
        }
        let rowLatencyDeltas = evalRows.compactMap {
            doubleValue($0["latencyDeltaPct"] ?? $0["latency_delta_pct"])
        }
        if rowLatencyDeltas.count == evalRows.count,
           let summaryMean = doubleValue(comparison["meanLatencyDeltaPct"] ?? comparison["mean_latency_delta_pct"]),
           !doubleEqual(summaryMean, mean(rowLatencyDeltas)) {
            return "comparison summary mean latency delta does not match eval.jsonl"
        }
        let baselinePassRate = passRate(evalRows.map {
            jsonBool($0["baselinePassed"] ?? $0["baseline_passed"])
        })
        if let baselinePassRate,
           let summaryPassRate = doubleValue(comparison["passRateBaseline"] ?? comparison["pass_rate_baseline"]),
           !doubleEqual(summaryPassRate, baselinePassRate) {
            return "comparison summary baseline pass rate does not match eval.jsonl"
        }
        let maskedPassRate = passRate(evalRows.map {
            jsonBool($0["maskedPassed"] ?? $0["masked_passed"])
        })
        if let maskedPassRate,
           let summaryPassRate = doubleValue(comparison["passRateMasked"] ?? comparison["pass_rate_masked"]),
           !doubleEqual(summaryPassRate, maskedPassRate) {
            return "comparison summary masked pass rate does not match eval.jsonl"
        }
        let rowClassifications = evalRowsPromptClassifications(evalRows)
        if let summaryCounts = intDictionaryValue(
            comparison["classificationCounts"] ?? comparison["prompt_classification_counts"]
        ),
            summaryCounts != classificationCounts(rowClassifications) {
            return "comparison summary prompt classification counts do not match eval.jsonl"
        }
        let rowBaselineQualified = evalRows.filter { jsonBool($0["baselineQualified"] ?? $0["baseline_qualified"]) == true }
        if let summaryBaselineQualified = intValue(
            comparison["baselineQualifiedPromptCount"] ?? comparison["baseline_qualified_prompt_count"]
        ),
            summaryBaselineQualified != rowBaselineQualified.count {
            return "comparison summary baseline-qualified prompt count does not match eval.jsonl"
        }
        let rowDegradedIDs = evalRows.compactMap { row -> String? in
            promptClassification(row) == "degraded" ? promptID(in: row) : nil
        }
        if let summaryDegradedIDs = stringArrayValue(comparison["degradedPromptIDs"] ?? comparison["degraded_prompt_ids"]),
           summaryDegradedIDs != rowDegradedIDs {
            return "comparison summary degraded prompt IDs do not match eval.jsonl"
        }
        let rowBaselineQualifiedCoverage = evalRowsBaselineQualifiedSemanticCoverage(evalRows)
        if let summaryCoverage = stringArrayValue(
            comparison["baselineQualifiedSemanticCoverage"] ?? comparison["baseline_qualified_semantic_coverage"]
        ),
            Set(summaryCoverage) != Set(rowBaselineQualifiedCoverage) {
            return "comparison summary baseline-qualified semantic coverage does not match eval.jsonl"
        }
        let rowMissingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(Set(rowBaselineQualifiedCoverage))
            .sorted()
        if let summaryMissingCoverage = stringArrayValue(
            comparison["missingBaselineQualifiedSemanticCoverage"]
                ?? comparison["missing_baseline_qualified_semantic_coverage"]
        ),
            Set(summaryMissingCoverage) != Set(rowMissingCoverage) {
            return "comparison summary missing baseline-qualified semantic coverage does not match eval.jsonl"
        }
        return nil
    }

    private static func evalRowsHighRiskDomains(_ rows: [[String: Any]]) -> [String] {
        Array(Set(rows.filter(rowIsBlockingRegression).flatMap { row -> [String] in
            if let domains = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]),
               !domains.isEmpty {
                return domains
            }
            return [trimmedString(row["domain"]) ?? "unknown"]
        })).sorted()
    }

    private static func evalRowsPromptClassifications(_ rows: [[String: Any]]) -> [String] {
        rows.map(promptClassification)
    }

    private static func classificationCounts(_ classifications: [String]) -> [String: Int] {
        var counts = [
            "baseline_invalid": 0,
            "preserved": 0,
            "degraded": 0,
            "inconclusive": 0
        ]
        for classification in classifications {
            counts[classification, default: 0] += 1
        }
        return counts
    }

    private static func evalRowsBaselineQualifiedSemanticCoverage(_ rows: [[String: Any]]) -> [String] {
        let domains = rows.filter {
            jsonBool($0["baselineQualified"] ?? $0["baseline_qualified"]) == true
        }.flatMap { row -> [String] in
            if let semantic = stringArrayValue(row["semanticDomains"] ?? row["semantic_domains"]),
               !semantic.isEmpty {
                return semantic.map(ExpertDomainTaxonomy.canonicalSemanticDomain)
            }
            if let domain = trimmedString(row["domain"]) {
                return [ExpertDomainTaxonomy.canonicalSemanticDomain(domain)]
            }
            return []
        }
        return Array(Set(domains)).filter { $0 != "general" }.sorted()
    }

    private static func rowIsBlockingRegression(_ row: [String: Any]) -> Bool {
        if promptClassification(row) == "degraded" {
            return true
        }
        if ["baseline_invalid", "inconclusive", "preserved"].contains(promptClassification(row)) {
            return false
        }
        let severity = rowRegressionSeverity(row)
        return severity == "high" || severity == "critical"
    }

    private static func evalRowsRegressionSeverity(_ rows: [[String: Any]]) -> String {
        rows.map(rowRegressionSeverity).max {
            severityRank($0) < severityRank($1)
        } ?? "none"
    }

    private static func rowRegressionSeverity(_ row: [String: Any]) -> String {
        let classification = promptClassification(row)
        if classification == "degraded" {
            return "critical"
        }
        if ["baseline_invalid", "inconclusive", "preserved"].contains(classification) {
            if let severity = trimmedString(row["regressionSeverity"] ?? row["regression_severity"]),
               severity == "watch" {
                return severity
            }
            return "none"
        }
        if let severity = trimmedString(row["regressionSeverity"] ?? row["regression_severity"]) {
            return severity
        }
        if trimmedString(row["risk"]) == "regression" {
            return "critical"
        }
        if jsonBool(row["maskedPassed"] ?? row["masked_passed"]) == false {
            return "high"
        }
        if let textDelta = doubleValue(row["textDelta"] ?? row["text_delta"]) {
            if textDelta > 0.50 { return "high" }
            if textDelta > 0.20 { return "watch" }
        }
        return "none"
    }

    private static func promptClassification(_ row: [String: Any]) -> String {
        if let classification = trimmedString(row["promptClassification"] ?? row["prompt_classification"]) {
            return classification
        }
        guard let baselinePassed = jsonBool(row["baselinePassed"] ?? row["baseline_passed"]),
              let maskedPassed = jsonBool(row["maskedPassed"] ?? row["masked_passed"]) else {
            return "inconclusive"
        }
        if !baselinePassed { return "baseline_invalid" }
        return maskedPassed ? "preserved" : "degraded"
    }

    private static func severityRank(_ severity: String) -> Int {
        switch severity {
        case "critical":
            return 3
        case "high":
            return 2
        case "watch":
            return 1
        default:
            return 0
        }
    }

    private static func passRate(_ values: [Bool?]) -> Double? {
        let scored = values.compactMap { $0 }
        guard !scored.isEmpty else { return nil }
        return Double(scored.filter { $0 }.count) / Double(scored.count)
    }

    private static func mean(_ values: [Double]) -> Double {
        guard !values.isEmpty else { return 0 }
        return values.reduce(0, +) / Double(values.count)
    }

    private static func doubleEqual(_ lhs: Double, _ rhs: Double) -> Bool {
        abs(lhs - rhs) <= 0.000_001
    }

    private struct DecodeSettings {
        let maxTokens: Int
        let temperature: Double
        let topP: Double
        let topK: Int
    }

    private static func evalDecodeSettingsIssue(index: [String: Any], rows: [[String: Any]]) -> String? {
        guard jsonBool(index["generation_settings_checked"] ?? index["generationSettingsChecked"]) == true else {
            return "eval_index.json is missing decode settings evidence"
        }
        for row in rows {
            let baselineValue = row["baselineGenerationSettings"] ?? row["baseline_generation_settings"]
            let maskedValue = row["maskedGenerationSettings"] ?? row["masked_generation_settings"]
            guard baselineValue != nil, maskedValue != nil else {
                return "eval.jsonl is missing baseline/masked decode settings evidence"
            }
            guard let baseline = decodeSettings(baselineValue),
                  let masked = decodeSettings(maskedValue) else {
                return "eval.jsonl has unreadable baseline/masked decode settings evidence"
            }
            if baseline.maxTokens != masked.maxTokens
                || abs(baseline.temperature - masked.temperature) > 0.000_001
                || abs(baseline.topP - masked.topP) > 0.000_001
                || baseline.topK != masked.topK {
                return "eval.jsonl baseline/masked decode settings do not match"
            }
        }
        return nil
    }

    private static func decodeSettings(_ value: Any?) -> DecodeSettings? {
        guard let object = value as? [String: Any],
              let maxTokens = intValue(object["max_tokens"] ?? object["maxTokens"]),
              maxTokens > 0,
              let temperature = doubleValue(object["temperature"]),
              temperature.isFinite,
              let topP = doubleValue(object["top_p"] ?? object["topP"]),
              topP.isFinite,
              let topK = intValue(object["top_k"] ?? object["topK"]),
              topK >= 0 else {
            return nil
        }
        return DecodeSettings(maxTokens: maxTokens, temperature: temperature, topP: topP, topK: topK)
    }

    private static func evalIndexLayerStatsCoverageIssue(index: [String: Any], promptCount: Int) -> String? {
        let baselineCount = intValue(index["baseline_layer_stats_prompt_count"] ?? index["baselineLayerStatsPromptCount"])
        let maskedCount = intValue(index["masked_layer_stats_prompt_count"] ?? index["maskedLayerStatsPromptCount"])
        guard baselineCount != nil || maskedCount != nil else { return nil }
        guard baselineCount == promptCount, maskedCount == promptCount else {
            return "eval_index.json layer-stat coverage is incomplete for indexed prompts"
        }
        return nil
    }

    private static func evalRowLayerStatsCoverageIssue(rows: [[String: Any]]) -> String? {
        let baselineRows = rows.filter { nonEmptyJSONArray($0["baselineLayerStats"] ?? $0["baseline_layer_stats"]) }.count
        let maskedRows = rows.filter { nonEmptyJSONArray($0["maskedLayerStats"] ?? $0["masked_layer_stats"]) }.count
        guard baselineRows > 0 || maskedRows > 0 else { return nil }
        guard baselineRows == rows.count, maskedRows == rows.count else {
            return "eval.jsonl layer-stat evidence is incomplete for baseline/masked prompts"
        }
        return nil
    }

    private static func evalTraceVariantIssue(
        rows: [[String: Any]],
        expectedPromptIDs: [String],
        disabledExpertCount: Int?,
        topKOverride: Int?,
        expectedDisabledByLayer: [Int: Set<Int>]? = nil,
        expectedBaselineRouteRecordCount: Int? = nil,
        expectedMaskedRouteRecordCount: Int? = nil
    ) -> String? {
        let expected = Set(expectedPromptIDs)
        let expectedMaskLayers = Set((expectedDisabledByLayer ?? [:]).filter { !$0.value.isEmpty }.keys)
        var baselinePromptIDs = Set<String>()
        var maskedPromptIDs = Set<String>()
        var maskedPromptIDsWithMaskEvidence = Set<String>()
        var maskedPromptExpectedMaskLayers: [String: Set<Int>] = [:]
        var baselineTraceCount = 0
        var maskedTraceCount = 0

        for row in rows {
            guard let id = promptID(in: row) else {
                return "eval_trace.jsonl prompt IDs are unreadable"
            }
            let variant = trimmedString(row["variant"])?.lowercased()
            switch variant {
            case "baseline":
                baselineTraceCount += 1
                baselinePromptIDs.insert(id)
            case "masked":
                maskedTraceCount += 1
                maskedPromptIDs.insert(id)
                if let issue = traceRowDisabledSelectionIssue(row, promptID: id) {
                    return issue
                }
                if let layer = traceRowExpectedMaskEvidenceLayer(
                    row,
                    expectedDisabledByLayer: expectedDisabledByLayer
                ) {
                    maskedPromptExpectedMaskLayers[id, default: []].insert(layer)
                }
                if traceRowHasMaskEvidence(
                    row,
                    disabledExpertCount: disabledExpertCount,
                    topKOverride: topKOverride
                ) {
                    maskedPromptIDsWithMaskEvidence.insert(id)
                }
            default:
                continue
            }
        }

        let missingBaseline = expected.subtracting(baselinePromptIDs)
        if !missingBaseline.isEmpty {
            return "eval_trace.jsonl missing baseline routing records for prompt IDs: \(previewIDs(missingBaseline))"
        }
        let missingMasked = expected.subtracting(maskedPromptIDs)
        if !missingMasked.isEmpty {
            return "eval_trace.jsonl missing masked routing records for prompt IDs: \(previewIDs(missingMasked))"
        }
        if let expectedBaselineRouteRecordCount,
           baselineTraceCount != expectedBaselineRouteRecordCount {
            return "eval_trace.jsonl has \(baselineTraceCount) baseline routing records for \(expectedBaselineRouteRecordCount) indexed baseline route records"
        }
        if let expectedMaskedRouteRecordCount,
           maskedTraceCount != expectedMaskedRouteRecordCount {
            return "eval_trace.jsonl has \(maskedTraceCount) masked routing records for \(expectedMaskedRouteRecordCount) indexed masked route records"
        }
        if (disabledExpertCount ?? 0) > 0 || topKOverride != nil {
            let missingMaskEvidence = expected.subtracting(maskedPromptIDsWithMaskEvidence)
            if !missingMaskEvidence.isEmpty {
                return "eval_trace.jsonl masked routing records are missing mask evidence for prompt IDs: \(previewIDs(missingMaskEvidence))"
            }
        }
        if !expectedMaskLayers.isEmpty {
            for id in expectedPromptIDs {
                let missingLayers = expectedMaskLayers.subtracting(maskedPromptExpectedMaskLayers[id] ?? [])
                if !missingLayers.isEmpty {
                    return "eval_trace.jsonl masked routing records are missing mask.json evidence for prompt \(id) layers: \(previewInts(missingLayers))"
                }
            }
        }
        return nil
    }

    private static func traceRowExpectedMaskEvidenceLayer(
        _ row: [String: Any],
        expectedDisabledByLayer: [Int: Set<Int>]?
    ) -> Int? {
        guard let expectedDisabledByLayer,
              let record = row["record"] as? [String: Any],
              let layer = intValue(record["layer"] ?? record["layerIndex"] ?? record["layer_index"]),
              let expectedDisabled = expectedDisabledByLayer[layer],
              !expectedDisabled.isEmpty else {
            return nil
        }
        let disabled = Set(arrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap { intValue($0) })
        return expectedDisabled.isSubset(of: disabled) ? layer : nil
    }

    private static func traceRowDisabledSelectionIssue(_ row: [String: Any], promptID: String) -> String? {
        guard let record = row["record"] as? [String: Any] else { return nil }
        let disabled = Set(arrayValue(record["disabledExperts"] ?? record["disabled_experts"]).compactMap { intValue($0) })
        guard !disabled.isEmpty else { return nil }
        let selected = Set(arrayValue(record["selectedExperts"] ?? record["selected_experts"]).compactMap { intValue($0) })
        let leaked = selected.intersection(disabled)
        guard !leaked.isEmpty else { return nil }
        let sorted = leaked.sorted()
        let head = sorted.prefix(5).map { String($0) }.joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        let preview = remaining == 0 ? head : "\(head), +\(remaining) more"
        return "eval_trace.jsonl masked routing records selected disabled experts for prompt \(promptID): \(preview)"
    }

    private static func traceRowHasMaskEvidence(
        _ row: [String: Any],
        disabledExpertCount: Int?,
        topKOverride: Int?
    ) -> Bool {
        guard let record = row["record"] as? [String: Any] else { return false }
        if (disabledExpertCount ?? 0) > 0 {
            let disabledExperts = arrayValue(record["disabledExperts"] ?? record["disabled_experts"])
            if !disabledExperts.isEmpty { return true }
            if (intValue(record["disabledExpertCount"] ?? record["disabled_expert_count"]) ?? 0) > 0 {
                return true
            }
            return false
        }
        if topKOverride != nil {
            return intValue(record["effectiveTopK"] ?? record["effective_top_k"] ?? record["topK"] ?? record["top_k"]) != nil
        }
        return true
    }

    private static func disabledExpertsByLayer(fromMaskURL url: URL) -> [Int: Set<Int>]? {
        guard let mask = readJSONObject(url) else { return nil }
        for key in ["disabled_by_layer", "layers", "disabledExpertsByLayer"] {
            if let map = intSetMap(mask[key]) {
                return map.filter { !$0.value.isEmpty }
            }
        }
        return [:]
    }

    private static func intSetMap(_ value: Any?) -> [Int: Set<Int>]? {
        guard let dictionary = value as? [String: Any] else { return nil }
        var result: [Int: Set<Int>] = [:]
        for (key, rawExperts) in dictionary {
            guard let layer = intValue(key) else { return nil }
            result[layer] = Set(arrayValue(rawExperts).compactMap { intValue($0) })
        }
        return result
    }

    private static func prunedSourceSuiteVerificationIssue(
        summary: [String: Any],
        prunedSource: URL,
        tracedPromptCount: Int?,
        suiteURL: URL?,
        expectedLayerCount: Int? = nil
    ) -> String? {
        guard jsonBool(summary["pruned_suite_verification_ready"]) == true else {
            if let issue = summary["pruned_suite_verification_issue"] as? String,
               !issue.isEmpty {
                return issue
            }
            return "missing pruned-source same-suite vMLX generation evidence"
        }
        guard let summaryURL = embeddedSidecarURL(
            summary["pruned_suite_summary"],
            prunedSource: prunedSource,
            fallbackName: "expert_lab_pruned_generation_summary.json"
        ),
            let generationURL = embeddedSidecarURL(
                summary["pruned_suite_generations"],
                prunedSource: prunedSource,
                fallbackName: "expert_lab_pruned_generations.jsonl"
            ) else {
            return "pruned-source generation sidecar paths must be embedded in the pruned BF16/F16 source"
        }
        guard FileManager.default.isReadableFile(atPath: summaryURL.path),
              FileManager.default.isReadableFile(atPath: generationURL.path) else {
            return "missing pruned-source generation sidecar paths"
        }
        guard let prunedSummary = readJSONObject(summaryURL) else {
            return "pruned-source generation summary is unreadable"
        }
        guard jsonBool(prunedSummary["ready"]) == true else {
            if let issue = prunedSummary["issue"] as? String, !issue.isEmpty {
                return issue
            }
            return "pruned-source generation summary did not pass"
        }
        let promptCount = intValue(prunedSummary["prompt_count"]) ?? 0
        if let tracedPromptCount, tracedPromptCount > 0, promptCount != tracedPromptCount {
            return "pruned-source generation covers \(promptCount) of \(tracedPromptCount) reviewed prompts"
        }
        if let suiteURL {
            let suiteCount = lineCount(suiteURL)
            if suiteCount != promptCount {
                return "pruned-source generation covers \(promptCount) of \(suiteCount) suite prompts"
            }
            guard let expectedSuiteSHA256 = fileSHA256(suiteURL) else {
                return "pruned-source reviewed prompt suite fingerprint could not be computed"
            }
            guard let recordedSuiteSHA256 = stringValue(prunedSummary["suite_sha256"] ?? prunedSummary["suiteSHA256"]),
                  !recordedSuiteSHA256.isEmpty else {
                return "pruned-source reviewed prompt suite fingerprint was not recorded"
            }
            if recordedSuiteSHA256 != expectedSuiteSHA256 {
                return "pruned-source reviewed prompt suite fingerprint does not match reviewed suite"
            }
        }
        if let recordedGenerationCount = intValue(prunedSummary["generation_count"]),
           recordedGenerationCount != promptCount {
            return "pruned-source generation summary records \(recordedGenerationCount) rows for \(promptCount) prompts"
        }
        let generationCount = lineCount(generationURL)
        if generationCount != promptCount {
            return "pruned-source generation JSONL has \(generationCount) rows for \(promptCount) prompts"
        }
        if let suiteURL,
           let issue = sameSuitePromptIDIssue(suiteURL: suiteURL, generationURL: generationURL) {
            return issue
        }
        guard let generationRows = jsonlObjects(generationURL) else {
            return "pruned-source generation JSONL is unreadable"
        }
        if let issue = prunedGenerationSettingsIssue(
            rows: generationRows,
            suiteURL: suiteURL,
            generationDefaults: prunedSummary["generation_defaults"] as? [String: Any]
        ) {
            return issue
        }
        guard let prunedSourcePath = stringValue(prunedSummary["pruned_source"] ?? summary["pruned_source"]) else {
            return "pruned-source generation summary is missing pruned source path evidence"
        }
        if canonicalPath(prunedSourcePath) != canonicalPath(prunedSource.path) {
            return "pruned-source generation summary path does not match the selected pruned BF16/F16 source"
        }
        if let issue = prunedGenerationRowEvidenceIssue(
            rows: generationRows,
            prunedSourcePath: prunedSourcePath,
            expectedLayerCount: expectedLayerCount
        ) {
            return issue
        }
        let requiredComparisonCount = max(promptCount, tracedPromptCount ?? 0, suiteURL.map(lineCount) ?? 0)
        guard let comparedCount = intValue(
            prunedSummary["reviewed_ab_comparison_count"] ?? prunedSummary["reviewed_masked_comparison_count"]
        ),
              comparedCount >= requiredComparisonCount else {
            let compared = intValue(
                prunedSummary["reviewed_ab_comparison_count"] ?? prunedSummary["reviewed_masked_comparison_count"]
            ) ?? 0
            return "pruned-source reviewed-output comparison covers \(compared) of \(requiredComparisonCount) prompts"
        }
        guard jsonBool(prunedSummary["pruned_validator_outcomes_checked"]) == true else {
            return "pruned-source generation is missing validator outcome evidence"
        }
        guard let baselineQualified = intValue(
            prunedSummary["baseline_qualified_prompt_count"] ?? prunedSummary["baselineQualifiedPromptCount"]
        ),
              baselineQualified > 0 else {
            return "pruned-source generation has no baseline-qualified validator prompts"
        }
        if let degraded = stringArrayValue(
            prunedSummary["pruned_degraded_prompt_ids"] ?? prunedSummary["prunedDegradedPromptIDs"]
        ),
            !degraded.isEmpty {
            return "pruned-source generation failed validators for baseline-qualified prompts: \(previewIDs(Set(degraded)))"
        }
        if let passRate = doubleValue(
            prunedSummary["pruned_baseline_qualified_pass_rate"] ?? prunedSummary["prunedBaselineQualifiedPassRate"]
        ),
            passRate < 1.0 {
            return "pruned-source generation validator pass rate is below 100% on baseline-qualified prompts"
        }
        let missingCoverage = stringArrayValue(
            prunedSummary["missing_baseline_qualified_semantic_coverage"]
                ?? prunedSummary["missingBaselineQualifiedSemanticCoverage"]
        ) ?? []
        if !missingCoverage.isEmpty {
            return "pruned-source generation baseline-qualified semantic coverage is missing: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        guard let classificationCounts = intDictionaryValue(
            prunedSummary["pruned_classification_counts"] ?? prunedSummary["prunedClassificationCounts"]
        ),
              classificationCounts.values.reduce(0, +) >= baselineQualified else {
            return "pruned-source generation is missing prompt classification counts"
        }
        guard let runtimeMode = prunedSummary["runtime_mode"] as? String,
              !runtimeMode.isEmpty,
              let runtimeDevice = prunedSummary["runtime_device"] as? String,
              !runtimeDevice.isEmpty,
              jsonBool(prunedSummary["runtime_metal_enabled"]) == true else {
            return "pruned-source generation is missing vMLX Metal runtime evidence"
        }
        if runtimeMode != "bf16_vmlx" {
            return "pruned-source generation did not record BF16/vMLX runtime evidence"
        }
        if stringValue(prunedSummary["runtime_backend"] ?? prunedSummary["runtimeBackend"]) != "vmlx" {
            return "pruned-source generation did not record vMLX backend evidence"
        }
        if jsonBool(prunedSummary["hook_coverage_complete"] ?? prunedSummary["hookCoverageComplete"]) == false {
            return "pruned-source generation recorded incomplete vMLX routed-layer hook coverage"
        }
        if let expectedLayerCount {
            guard let hookedLayers = intValue(prunedSummary["hooked_moe_layers"] ?? prunedSummary["hookedMOELayers"]) else {
                return "pruned-source generation is missing vMLX routed-layer hook evidence"
            }
            if hookedLayers < expectedLayerCount {
                return "pruned-source generation vMLX hook coverage \(hookedLayers) of \(expectedLayerCount) routed layers"
            }
        }
        if let expectedMOELayers = intValue(prunedSummary["expected_moe_layers"] ?? prunedSummary["expectedMOELayers"]),
           let hookedLayers = intValue(prunedSummary["hooked_moe_layers"] ?? prunedSummary["hookedMOELayers"]),
           hookedLayers < expectedMOELayers {
            return "pruned-source generation vMLX hook coverage \(hookedLayers) of \(expectedMOELayers) config-routed layers"
        }
        guard let jangToolsVersion = stringValue(prunedSummary["jang_tools_version"] ?? prunedSummary["jangToolsVersion"]),
              !jangToolsVersion.isEmpty,
              let mlxVersion = stringValue(prunedSummary["mlx_version"] ?? prunedSummary["mlxVersion"]),
              !mlxVersion.isEmpty,
              let mlxLMVersion = stringValue(prunedSummary["mlx_lm_version"] ?? prunedSummary["mlxLMVersion"]),
              !mlxLMVersion.isEmpty else {
            return "pruned-source generation is missing vMLX package version evidence"
        }
        guard let runtimeSourcePath = stringValue(prunedSummary["runtime_source_model_path"]) else {
            return "pruned-source generation is missing runtime source path evidence"
        }
        if canonicalPath(runtimeSourcePath) != canonicalPath(prunedSource.path) {
            return "pruned-source generation source path does not match the pruned BF16/F16 source"
        }
        return nil
    }

    private static func prunedGenerationRowEvidenceIssue(
        rows: [[String: Any]],
        prunedSourcePath: String?,
        expectedLayerCount: Int? = nil
    ) -> String? {
        for row in rows {
            guard let result = row["result"] as? [String: Any] else {
                return "pruned-source generation row is missing result"
            }
            let text = trimmedString(result["text"]) ?? ""
            let tokens = intValue(result["tokens"]) ?? 0
            if text.isEmpty || tokens <= 0 {
                return "pruned-source generation produced an empty prompt output"
            }
            guard let runtime = result["runtime_info"] as? [String: Any] else {
                return "pruned-source generation is missing per-prompt runtime evidence"
            }
            guard let runtimeMode = trimmedString(runtime["runtime_mode"] ?? runtime["runtimeMode"]),
                  !runtimeMode.isEmpty,
                  let runtimeDevice = trimmedString(runtime["device_name"] ?? runtime["runtime_device"] ?? runtime["runtimeDevice"]),
                  !runtimeDevice.isEmpty,
                  jsonBool(runtime["runtime_metal_enabled"] ?? runtime["runtimeMetalEnabled"]) == true else {
                return "pruned-source generation is missing per-prompt vMLX Metal runtime evidence"
            }
            if runtimeMode != "bf16_vmlx" {
                return "pruned-source generation did not record per-prompt BF16/vMLX runtime evidence"
            }
            if trimmedString(runtime["backend"] ?? runtime["runtime_backend"] ?? runtime["runtimeBackend"]) != "vmlx" {
                return "pruned-source generation did not record per-prompt vMLX backend evidence"
            }
            if trimmedString(runtime["jang_tools_version"] ?? runtime["jangToolsVersion"]) == nil
                || trimmedString(runtime["mlx_version"] ?? runtime["mlxVersion"]) == nil
                || trimmedString(runtime["mlx_lm_version"] ?? runtime["mlxLMVersion"]) == nil {
                return "pruned-source generation is missing per-prompt vMLX package version evidence"
            }
            guard let runtimeSourcePath = trimmedString(runtime["source_model_path"] ?? runtime["sourceModelPath"]) else {
                return "pruned-source generation is missing per-prompt source path evidence"
            }
            if let prunedSourcePath,
               normalizedPath(runtimeSourcePath) != normalizedPath(prunedSourcePath) {
                return "pruned-source generation per-prompt source path does not match the pruned BF16/F16 source"
            }
            if let issue = prunedGenerationLayerStatsIssue(
                result: result,
                expectedLayerCount: expectedLayerCount
            ) {
                return issue
            }
            if let issue = prunedGenerationTokenTraceIssue(
                result: result,
                expectedLayerCount: expectedLayerCount
            ) {
                return issue
            }
        }
        return nil
    }

    private static func prunedGenerationLayerStatsIssue(
        result: [String: Any],
        expectedLayerCount: Int?
    ) -> String? {
        guard expectedLayerCount != nil else { return nil }
        guard let rows = result["layer_stats"] as? [[String: Any]], !rows.isEmpty else {
            return "pruned-source generation is missing per-prompt routed-layer stats"
        }
        let layerIDs = rows.compactMap { intValue($0["layer"] ?? $0["layer_id"] ?? $0["layerID"]) }
        if layerIDs.count != rows.count {
            return "pruned-source generation routed-layer stats have unreadable layer IDs"
        }
        if Set(layerIDs).count < layerIDs.count {
            return "pruned-source generation routed-layer stats contain duplicate layers"
        }
        if let expectedLayerCount {
            if rows.count < expectedLayerCount {
                return "pruned-source generation routed-layer stats cover \(rows.count) of \(expectedLayerCount) layers"
            }
        }
        if rows.contains(where: { (intValue($0["token_count"] ?? $0["tokenCount"]) ?? 0) <= 0 }) {
            return "pruned-source generation routed-layer stats are missing token-position depth"
        }
        if rows.contains(where: { layerStatMap($0["hit_counts"] ?? $0["hitCounts"]).isEmpty }) {
            return "pruned-source generation routed-layer stats are missing expert hit counts"
        }
        if rows.contains(where: { layerStatMap($0["probability_mass"] ?? $0["probabilityMass"]).isEmpty }) {
            return "pruned-source generation routed-layer stats are missing expert gate-mass evidence"
        }
        return nil
    }

    private static func prunedGenerationTokenTraceIssue(
        result: [String: Any],
        expectedLayerCount: Int?
    ) -> String? {
        guard expectedLayerCount != nil else { return nil }
        guard let layerStats = result["layer_stats"] as? [[String: Any]], !layerStats.isEmpty else {
            return nil
        }
        let expectedRoutes = layerStats.reduce(0) {
            $0 + (intValue($1["token_count"] ?? $1["tokenCount"]) ?? 0)
        }
        guard expectedRoutes > 0 else {
            return "pruned-source generation is missing per-prompt routed layer-token records"
        }
        guard let trace = result["token_trace"] as? [[String: Any]], !trace.isEmpty else {
            return "pruned-source generation is missing per-prompt token_trace routing evidence"
        }
        if trace.count != expectedRoutes {
            return "pruned-source generation token_trace has \(trace.count) rows for \(expectedRoutes) routed layer-token records"
        }
        for row in trace {
            guard intValue(row["layer"]) != nil,
                  intValue(row["token_index"] ?? row["tokenIndex"]) != nil else {
                return "pruned-source generation token_trace is missing layer/token evidence"
            }
            if arrayValue(row["selected_experts"] ?? row["selectedExperts"]).isEmpty {
                return "pruned-source generation token_trace is missing selected expert evidence"
            }
        }
        return nil
    }

    private static func layerStatMap(_ value: Any?) -> [String: Any] {
        if let value = value as? [String: Any] { return value }
        return [:]
    }

    private static func sameSuitePromptIDIssue(suiteURL: URL, generationURL: URL) -> String? {
        guard let suiteIDs = jsonlStringIDs(suiteURL, keys: ["id", "prompt_id", "promptID"]) else {
            return "suite.jsonl prompt IDs are unreadable"
        }
        guard let generationIDs = jsonlPromptIDs(generationURL) else {
            return "pruned-source generation prompt IDs are unreadable"
        }
        let suiteSet = Set(suiteIDs)
        if suiteSet.count != suiteIDs.count {
            return "suite.jsonl contains duplicate prompt IDs"
        }
        let generationSet = Set(generationIDs)
        if generationSet.count != generationIDs.count {
            return "pruned-source generation JSONL contains duplicate prompt IDs"
        }
        let missing = suiteSet.subtracting(generationSet)
        if !missing.isEmpty {
            return "pruned-source generation missing suite prompt IDs: \(previewIDs(missing))"
        }
        let unexpected = generationSet.subtracting(suiteSet)
        if !unexpected.isEmpty {
            return "pruned-source generation has prompt IDs outside reviewed suite: \(previewIDs(unexpected))"
        }
        if generationIDs != suiteIDs {
            return "pruned-source generation prompt order does not match reviewed suite"
        }
        return nil
    }

    private static func prunedGenerationSettingsIssue(
        rows: [[String: Any]],
        suiteURL: URL?,
        generationDefaults: [String: Any]?
    ) -> String? {
        guard let generationDefaults else { return nil }
        let defaultMaxTokens = intValue(generationDefaults["max_tokens"] ?? generationDefaults["maxTokens"])
        let defaultTemperature = doubleValue(generationDefaults["temperature"])
        guard defaultMaxTokens != nil || defaultTemperature != nil else {
            return "pruned-source generation defaults are missing decode settings"
        }
        let suiteSettings = suiteURL.flatMap(suiteGenerationSettings)
        for row in rows {
            guard let prompt = row["prompt"] as? [String: Any],
                  let promptID = promptID(in: prompt) else {
                return "pruned-source generation row is missing prompt settings evidence"
            }
            guard let result = row["result"] as? [String: Any],
                  let settings = result["generation_settings"] as? [String: Any] else {
                return "pruned-source generation row is missing decode settings evidence"
            }
            guard let recordedMaxTokens = intValue(settings["max_tokens"] ?? settings["maxTokens"]),
                  recordedMaxTokens > 0,
                  let recordedTemperature = doubleValue(settings["temperature"]) else {
                return "pruned-source generation row has unreadable decode settings"
            }
            let expected = suiteSettings?[promptID]
            if let expectedMaxTokens = expected?.maxTokens ?? defaultMaxTokens,
               recordedMaxTokens != expectedMaxTokens {
                return "pruned-source generation max_tokens for \(promptID) does not match reviewed suite"
            }
            if let expectedTemperature = expected?.temperature ?? defaultTemperature,
               abs(recordedTemperature - expectedTemperature) > 0.000_001 {
                return "pruned-source generation temperature for \(promptID) does not match reviewed suite"
            }
        }
        return nil
    }

    private static func suiteGenerationSettings(_ url: URL) -> [String: (maxTokens: Int?, temperature: Double?)]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var settings: [String: (maxTokens: Int?, temperature: Double?)] = [:]
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                  let promptID = promptID(in: object) else {
                return nil
            }
            settings[promptID] = (
                intValue(object["max_new_tokens"] ?? object["maxTokens"]),
                doubleValue(object["temperature"])
            )
        }
        return settings
    }

    private static func stringArrayValue(_ value: Any?) -> [String]? {
        switch value {
        case let values as [String]:
            return values
        case let values as [Any]:
            return values.compactMap { $0 as? String }
        default:
            return nil
        }
    }

    private static func arrayValue(_ value: Any?) -> [Any] {
        switch value {
        case let values as [Any]:
            return values
        default:
            return []
        }
    }

    private static func urlValue(_ value: Any?) -> URL? {
        guard let path = value as? String, !path.isEmpty else { return nil }
        return URL(fileURLWithPath: path)
    }

    private static func canonicalPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }

    private static func jsonlStringIDs(_ url: URL, keys: [String]) -> [String]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var ids: [String] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            if let id = keys.lazy.compactMap({ object[$0] as? String }).first {
                ids.append(id)
            }
        }
        return ids
    }

    private static func jsonlObjects(_ url: URL) -> [[String: Any]]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var rows: [[String: Any]] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            rows.append(object)
        }
        return rows
    }

    private static func jsonlPromptIDs(_ url: URL) -> [String]? {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return nil }
        var ids: [String] = []
        for rawLine in text.split(whereSeparator: \.isNewline) {
            guard let data = rawLine.data(using: .utf8),
                  let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
                return nil
            }
            if let id = ["promptID", "prompt_id", "id"].lazy.compactMap({ object[$0] as? String }).first {
                ids.append(id)
                continue
            }
            if let prompt = object["prompt"] as? [String: Any],
               let id = ["promptID", "prompt_id", "id"].lazy.compactMap({ prompt[$0] as? String }).first {
                ids.append(id)
                continue
            }
            return nil
        }
        return ids
    }

    private static func lineCount(_ url: URL) -> Int {
        guard let text = try? String(contentsOf: url, encoding: .utf8) else { return 0 }
        return text.split(whereSeparator: \.isNewline).count
    }

    private static func expectedReviewedLayerCount(summary: [String: Any]) -> Int? {
        if let layerCount = intValue(summary["layer_count"] ?? summary["layerCount"] ?? summary["num_layers"] ?? summary["numLayers"]),
           layerCount > 0 {
            return layerCount
        }
        if let layers = summary["layers"] as? [String: Any], !layers.isEmpty {
            return layers.count
        }
        let candidatePaths = [
            stringValue(summary["source_model_path"] ?? summary["sourceModelPath"]),
            stringValue(summary["source_model"] ?? summary["sourceModel"]),
            stringValue(summary["pruned_source"] ?? summary["prunedSource"])
        ].compactMap { $0 }
        for path in candidatePaths {
            if let layerCount = configLayerCount(modelPath: path) {
                return layerCount
            }
        }
        return nil
    }

    private static func configLayerCount(modelPath: String) -> Int? {
        let configURL = URL(fileURLWithPath: modelPath).appendingPathComponent("config.json")
        guard let config = readJSONObject(configURL) else { return nil }
        let textConfig = config["text_config"] as? [String: Any] ?? config
        let layerCount = intValue(
            textConfig["num_hidden_layers"] ?? textConfig["n_layer"] ?? textConfig["num_layers"]
        )
        guard let layerCount, layerCount > 0 else { return nil }
        return layerCount
    }

    private static func suiteSemanticCoverageIssue(_ suiteURL: URL) -> String? {
        guard let suite = try? ExpertPromptSuite.loadJSONL(
            name: suiteURL.deletingPathExtension().lastPathComponent,
            from: suiteURL
        ) else {
            return "suite.jsonl semantic prompt coverage is unreadable"
        }
        let semanticDomains = Set(suite.prompts.flatMap { ExpertDomainTaxonomy.semanticDomains(for: $0) })
        let missing = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains.subtracting(semanticDomains).sorted()
        return missing.isEmpty
            ? nil
            : "suite.jsonl is missing required semantic prompt probes: \(missing.joined(separator: ", "))"
    }

    private static func evalIndexSemanticCoverageIssue(_ index: [String: Any]) -> String? {
        guard let semanticCoverage = stringArrayValue(index["semantic_coverage"] ?? index["semanticCoverage"]),
              !semanticCoverage.isEmpty else {
            return "eval_index.json is missing semantic coverage evidence"
        }
        let coverage = Set(
            semanticCoverage
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        let missingCoverage = ExpertDomainTaxonomy.requiredReviewedPruneSemanticDomains
            .subtracting(coverage)
            .sorted()
        if !missingCoverage.isEmpty {
            return "eval_index.json semantic coverage is missing required probes: \(missingCoverage.joined(separator: ", "))"
        }
        guard let recordedMissing = stringArrayValue(index["missing_semantic_coverage"] ?? index["missingSemanticCoverage"]) else {
            return "eval_index.json is missing missing-semantic-coverage evidence"
        }
        let missing = Set(
            recordedMissing
                .map(ExpertDomainTaxonomy.canonicalSemanticDomain)
                .filter { $0 != "general" }
        )
        if !missing.isEmpty {
            return "eval_index.json records missing semantic prompt probes: \(missing.sorted().joined(separator: ", "))"
        }
        return nil
    }

    private static func evalIndexValidatorEvidenceIssue(_ index: [String: Any]) -> String? {
        guard trimmedString(index["validator_schema"] ?? index["validatorSchema"]) != nil,
              intValue(index["validator_available_prompt_count"] ?? index["validatorAvailablePromptCount"]) != nil,
              dictionaryValue(index["prompt_classification_counts"] ?? index["promptClassificationCounts"]) != nil else {
            return "eval_index.json is missing validator classification evidence"
        }
        guard let promptCount = intValue(index["prompt_count"] ?? index["promptCount"]) else {
            return "eval_index.json is missing prompt count evidence"
        }
        guard let baselineQualified = intValue(
            index["baseline_qualified_prompt_count"] ?? index["baselineQualifiedPromptCount"]
        ),
              baselineQualified > 0 else {
            return "eval_index.json has no baseline-qualified validator prompts"
        }
        guard let baselineQualifiedIDs = stringArrayValue(
            index["baseline_qualified_prompt_ids"] ?? index["baselineQualifiedPromptIDs"]
        ),
              let baselineInvalidIDs = stringArrayValue(
                index["baseline_invalid_prompt_ids"] ?? index["baselineInvalidPromptIDs"]
              ),
              let inconclusiveIDs = stringArrayValue(
                index["inconclusive_prompt_ids"] ?? index["inconclusivePromptIDs"]
              ),
              let preservedIDs = stringArrayValue(
                index["preserved_prompt_ids"] ?? index["preservedPromptIDs"]
              ),
              let degradedIDs = stringArrayValue(
                index["degraded_prompt_ids"] ?? index["degradedPromptIDs"]
              ) else {
            return "eval_index.json is missing prompt classification ID lists"
        }
        if baselineQualifiedIDs.count != baselineQualified {
            return "eval_index.json baseline-qualified prompt IDs do not match the baseline-qualified count"
        }
        let classified = baselineInvalidIDs.count + inconclusiveIDs.count + preservedIDs.count + degradedIDs.count
        if classified != promptCount {
            return "eval_index.json prompt classifications cover \(classified) of \(promptCount) prompts"
        }
        if !degradedIDs.isEmpty {
            return "eval_index.json has baseline-qualified prompt regressions: \(previewIDs(Set(degradedIDs)))"
        }
        let missingCoverage = stringArrayValue(
            index["missing_baseline_qualified_semantic_coverage"]
                ?? index["missingBaselineQualifiedSemanticCoverage"]
        ) ?? []
        if !missingCoverage.isEmpty {
            return "eval_index.json baseline-qualified semantic coverage is missing: \(missingCoverage.sorted().joined(separator: ", "))"
        }
        guard let coverage = stringArrayValue(
            index["baseline_qualified_semantic_coverage"]
                ?? index["baselineQualifiedSemanticCoverage"]
        ),
              !coverage.isEmpty else {
            return "eval_index.json is missing baseline-qualified semantic coverage evidence"
        }
        if let passRate = doubleValue(
            index["baseline_qualified_masked_pass_rate"] ?? index["baselineQualifiedMaskedPassRate"]
        ),
            passRate < 1.0 {
            return "eval_index.json masked validator pass rate is below 100% on baseline-qualified prompts"
        }
        return nil
    }

    private static func previewIDs(_ ids: Set<String>) -> String {
        let sorted = ids.sorted()
        let head = sorted.prefix(5).joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func previewInts(_ values: Set<Int>) -> String {
        let sorted = values.sorted()
        let head = sorted.prefix(5).map { String($0) }.joined(separator: ", ")
        let remaining = max(0, sorted.count - 5)
        return remaining == 0 ? head : "\(head), +\(remaining) more"
    }

    private static func readJSONObject(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return nil
        }
        return json
    }

    private static func fileSHA256(_ url: URL) -> String? {
        guard let data = try? Data(contentsOf: url) else { return nil }
        return SHA256.hash(data: data)
            .map { String(format: "%02x", $0) }
            .joined()
    }

    private static func intValue(_ value: Any?) -> Int? {
        switch value {
        case let value as Int:
            return value
        case let value as Double where value.isFinite:
            return Int(value)
        case let value as NSNumber:
            return value.intValue
        case let value as String:
            return Int(value)
        default:
            return nil
        }
    }

    private static func doubleValue(_ value: Any?) -> Double? {
        switch value {
        case let value as Double where value.isFinite:
            return value
        case let value as Float where value.isFinite:
            return Double(value)
        case let value as Int:
            return Double(value)
        case let value as NSNumber:
            return value.doubleValue
        case let value as String:
            return Double(value)
        default:
            return nil
        }
    }

    private static func isBlockingRegressionSeverity(_ value: Any?) -> Bool {
        guard let severity = value as? String else { return false }
        return severity == "high" || severity == "critical"
    }

    private static func stringValue(_ value: Any?) -> String? {
        guard let value else { return nil }
        if value is NSNull { return nil }
        if let string = value as? String {
            return string.isEmpty ? nil : string
        }
        return String(describing: value)
    }

    private static func trimmedString(_ value: Any?) -> String? {
        guard let trimmed = stringValue(value)?.trimmingCharacters(in: .whitespacesAndNewlines),
              !trimmed.isEmpty else {
            return nil
        }
        return trimmed
    }

    private static func nonEmptyJSONArray(_ value: Any?) -> Bool {
        guard let array = value as? [Any] else { return false }
        return !array.isEmpty
    }

    private static func dictionaryValue(_ value: Any?) -> [String: Any]? {
        value as? [String: Any]
    }

    private static func intDictionaryValue(_ value: Any?) -> [String: Int]? {
        guard let dictionary = dictionaryValue(value) else { return nil }
        var result: [String: Int] = [:]
        for (key, raw) in dictionary {
            guard let value = intValue(raw) else { return nil }
            result[key] = value
        }
        return result
    }

    private static func promptID(in row: [String: Any]) -> String? {
        ["promptID", "prompt_id", "id"].lazy.compactMap { key in
            trimmedString(row[key])
        }.first
    }

    private static func normalizedPath(_ path: String) -> String {
        URL(fileURLWithPath: path)
            .resolvingSymlinksInPath()
            .standardizedFileURL
            .path
    }

    private static let requiredPrunedSourceVerificationChecks = [
        "config_parses",
        "index_parses",
        "index_covers_tensors",
        "router_rows_match",
        "expert_rows_match"
    ]

    private static func readPrunedSourceVerification(from obj: [String: Any]) -> (ok: Bool, hint: String) {
        guard jsonBool(obj["ok"]) == true else {
            return (false, "Pruned source verification did not pass")
        }
        let checks = (obj["checks"] as? [String: Any])?.compactMapValues(jsonBool) ?? [:]
        let missing = requiredPrunedSourceVerificationChecks.filter { checks[$0] == nil }
        let failed = checks
            .filter { !$0.value }
            .map(\.key)
            .sorted()
        if !missing.isEmpty {
            return (false, "Missing required verification checks: \(missing.joined(separator: ", "))")
        }
        if !failed.isEmpty {
            return (false, "Failed verification checks: \(failed.joined(separator: ", "))")
        }
        return (true, "verification.json passed all structural checks")
    }

    private static func jsonBool(_ value: Any?) -> Bool? {
        if let value = value as? Bool {
            return value
        }
        if let value = value as? NSNumber {
            return value.boolValue
        }
        return nil
    }

    private static func bundledPythonHealthy() -> PreflightCheck {
        let ok = BundleResolver.healthCheck()
        return .init(id: .bundledPythonHealthy, title: "Bundled Python runtime healthy",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "Bundled python3 missing — reinstall JANG Studio")
    }
}
