// JANGStudio/JANGStudio/Verify/PreflightRunner.swift
import Foundation

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
        out.append(Self.jangtqProfileSupported(plan: plan, profiles: profiles))
        out.append(Self.jangtqSourceDtype(plan: plan))
        out.append(Self.bf16For512Experts(plan: plan, types: capabilities.knownExpert512Types))
        out.append(Self.hadamardVsLowBits(plan: plan, profiles: profiles))
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
        if let p = profiles.jangtq.first(where: { $0.name == profile }) { return p.avgBits ?? Double(p.bits) }
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

    private static func jangtqArchSupported(plan: ConversionPlan, whitelist: [String]) -> PreflightCheck {
        if plan.family != .jangtq {
            return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported", status: .pass, hint: nil)
        }
        let mt = plan.detected?.modelType ?? ""
        let ok = whitelist.contains(mt)
        return .init(id: .jangtqArchSupported, title: "JANGTQ arch supported",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ supports \(whitelist.joined(separator: ", ")); detected \(mt)")
    }

    private static func jangtqProfileSupported(plan: ConversionPlan, profiles: Profiles) -> PreflightCheck {
        if plan.family != .jangtq {
            return .init(id: .jangtqProfileSupported, title: "JANGTQ profile supported", status: .pass, hint: nil)
        }
        let mt = plan.detected?.modelType ?? ""
        let allowed = profiles.jangtqProfileNames(for: mt)
        let ok = allowed.contains(plan.profile)
        return .init(id: .jangtqProfileSupported, title: "JANGTQ profile supported",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "\(mt) supports \(allowed.joined(separator: ", ")); selected \(plan.profile)")
    }

    private static func jangtqSourceDtype(plan: ConversionPlan) -> PreflightCheck {
        if plan.family != .jangtq { return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype", status: .pass, hint: nil) }
        let d = plan.detected?.dtype ?? .unknown
        let ok = (d == .bf16 || d == .fp8)
        return .init(id: .jangtqSourceDtype, title: "JANGTQ source dtype",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "JANGTQ expects BF16 or FP8 source; detected \(d.rawValue)")
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
            return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil)
        }
        if plan.overrides.forceDtype == .fp16 {
            let expertStr = dynamic512 ? "\(plan.detected?.numExperts ?? 0) experts" : mt
            return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .warn,
                         hint: "\(expertStr) — bfloat16 strongly recommended over float16 to avoid overflow")
        }
        return .init(id: .bf16For512Experts, title: "BF16 forced for 512+ expert model", status: .pass, hint: nil)
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

    private static func bundledPythonHealthy() -> PreflightCheck {
        let ok = BundleResolver.healthCheck()
        return .init(id: .bundledPythonHealthy, title: "Bundled Python runtime healthy",
                     status: ok ? .pass : .fail,
                     hint: ok ? nil : "Bundled python3 missing — reinstall JANG Studio")
    }
}
