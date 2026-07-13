// JANGStudio/JANGStudio/Wizard/QuantizationGuide.swift
import SwiftUI

struct QuantProfileGuide: Equatable {
    let name: String
    let title: String
    let body: String
    let detail: String?
    let badges: [String]
}

struct QuantGlossaryItem: Equatable, Identifiable {
    let term: String
    let explanation: String

    var id: String { term }
}

enum QuantizationGuidance {
    static func familyTitle(_ family: Family) -> String {
        switch family {
        case .jang: "JANG"
        case .jangtq: "JANGTQ"
        }
    }

    static func familyBody(_ family: Family, isMoE: Bool) -> String {
        switch family {
        case .jang:
            return "General-purpose JANG mixed precision. It works across dense, MoE, vision-language, and hybrid models by giving sensitive tensors more bits and cheaper tensors fewer bits."
        case .jangtq:
            if isMoE {
                return "TurboQuant path for supported MoE models. It codebook-compresses routed experts for smaller Expert Lab bundles, trace collection, and temporary expert masks."
            }
            return "TurboQuant is only useful for supported routed-expert MoE models. Dense models should stay on JANG."
        }
    }

    static func profileGuide(name: String, profiles: Profiles) -> QuantProfileGuide {
        if let p = profiles.jang.first(where: { $0.name == name }) {
            return jangGuide(p)
        }
        if let p = profiles.jangtq.first(where: { $0.name == name }) {
            return jangtqGuide(p)
        }
        if name.hasPrefix("JANGTQ") {
            let bits = Int(name.filter(\.isNumber).prefix(1)) ?? 0
            return QuantProfileGuide(
                name: name,
                title: bits > 0 ? "\(bits)-bit TurboQuant" : "TurboQuant profile",
                body: "A JANGTQ profile for routed-expert MoE conversion. JANG Studio will use the dedicated architecture converter when this profile is supported.",
                detail: "Method and Hadamard controls do not apply to JANGTQ converters.",
                badges: ["JANGTQ", bits > 0 ? "\(bits)-bit" : "codebook"]
            )
        }
        return QuantProfileGuide(
            name: name,
            title: "Custom profile",
            body: "This profile is not in the bundled metadata. JANG Studio will pass it through to jang-tools, but size and quality guidance may be incomplete.",
            detail: nil,
            badges: ["custom"]
        )
    }

    static func methodBody(_ method: QuantMethod, family: Family) -> String {
        if family == .jangtq {
            return "JANGTQ converters use their own codebook quantization path. The Method picker is ignored for JANGTQ runs."
        }
        switch method {
        case .mse:
            return "MSE searches for per-group scales that minimize reconstruction error. It is the quality default and the right choice unless you are optimizing conversion time."
        case .rtn:
            return "RTN is round-to-nearest. It converts faster, but usually has more quantization error than MSE."
        case .mseAll:
            return "MSE (all) applies the slower MSE search more broadly. Use it when you want the most careful JANG conversion and can spend extra conversion time."
        }
    }

    static func hadamardBody(isEnabled: Bool, profile: String, family: Family) -> String {
        if family == .jangtq {
            return "Hadamard rotation is not passed to JANGTQ converters. JANGTQ handles expert compression with its own codebook layout."
        }
        let state = isEnabled ? "On" : "Off"
        if profile.contains("_1") || profile.contains("_2") {
            return "\(state). Hadamard rotation can hurt very low-bit profiles, so 1-bit and 2-bit JANG profiles usually leave it off."
        }
        return "\(state). For 3-bit and higher JANG profiles, Hadamard rotation often reduces quantization error before packing weights."
    }

    static func forceDtypeBody(_ dtype: SourceDtype?) -> String {
        switch dtype {
        case .bf16:
            return "Forces bfloat16 handling. This is safest for very large MoE sources and avoids float16 overflow on huge expert routers."
        case .fp16:
            return "Forces float16 handling. Use only when you know the source weights are true FP16 and BF16 auto-detection is wrong."
        case .fp8:
            return "Forces FP8 source handling. Use for sources stored as FP8 E4M3/E5M2 when auto-detection misses the quantization config."
        case .jangV2:
            return "JANG v2 is an output format, not a normal source dtype override. Leave this on Auto for ordinary conversions."
        case .unknown:
            return "Auto lets JANG Studio infer the source dtype from config and safetensors metadata."
        case nil:
            return "Auto lets JANG Studio infer the source dtype from config and safetensors metadata."
        }
    }

    static func blockSizeBody(_ blockSize: Int?) -> String {
        guard let blockSize, blockSize > 0 else {
            return "Auto chooses the converter's default group size. Smaller groups can preserve more detail; larger groups can reduce sidecar overhead."
        }
        return "\(blockSize) weights share each quantization scale/bias group. Change this only when matching a known runtime or experiment."
    }

    static let prequantPruneBody =
        "Hard pruning physically removes routed-expert tensors from the BF16/F16 source before quantization. It writes a new HuggingFace-style source folder, then you convert that smaller folder."

    static let traceMaskBody =
        "Trace masks are temporary runtime controls on a converted Expert Lab bundle. They help you test experts without deleting weights from the original or pruned source."

    static let tierGlossary: [QuantGlossaryItem] = [
        QuantGlossaryItem(term: "K-quant", explanation: "Budget-neutral target-bit profile. The average size stays near the named bit width while important tensors get extra precision."),
        QuantGlossaryItem(term: "S / M / L", explanation: "Small, medium, large protection levels for tiered profiles. Larger letters spend more bits on critical or important tensors."),
        QuantGlossaryItem(term: "Critical", explanation: "Usually attention and other tensors where quantization errors are most visible."),
        QuantGlossaryItem(term: "Important", explanation: "Middle tier, often embeddings or architecture-specific sensitive weights."),
        QuantGlossaryItem(term: "Compress", explanation: "The base bit width used for most tensors in a tiered JANG profile.")
    ]

    private static func jangGuide(_ profile: ProfileInfo) -> QuantProfileGuide {
        if profile.isKquant {
            let bits = bitsText(profile.avgBits)
            return QuantProfileGuide(
                name: profile.name,
                title: "\(bits)-bit K-quant target",
                body: "\(profile.name) aims for about \(bits) bits per weight overall. K-quant keeps the output close to a uniform \(bits)-bit size, but reallocates precision toward sensitive tensors.",
                detail: profile.description,
                badges: ["JANG", "K-quant", "\(bits)-bit avg"]
            )
        }

        let critical = profile.criticalBits ?? 0
        let important = profile.importantBits ?? 0
        let compress = profile.compressBits ?? 0
        let sizeLabel = profile.name.split(separator: "_").last.map(String.init) ?? profile.name
        return QuantProfileGuide(
            name: profile.name,
            title: "\(sizeLabel) tiered protection",
            body: "Most tensors use \(compress)-bit compression, important tensors use \(important)-bit, and critical tensors use \(critical)-bit. This costs more than pure \(compress)-bit, but protects the parts most likely to break quality.",
            detail: profile.description,
            badges: ["JANG", "critical \(critical)", "important \(important)", "base \(compress)"]
        )
    }

    private static func jangtqGuide(_ profile: JANGTQProfileInfo) -> QuantProfileGuide {
        QuantProfileGuide(
            name: profile.name,
            title: "\(profile.bits)-bit TurboQuant",
            body: "\(profile.name) uses a \(profile.bits)-bit codebook path for supported MoE routed experts. It is the path that enables smaller Expert Lab bundles and interactive expert masking.",
            detail: "\(profile.description). Requires \(profile.minSourceDtype.joined(separator: " or ")) source weights.",
            badges: ["JANGTQ", "\(profile.bits)-bit", "MoE"]
        )
    }

    private static func bitsText(_ value: Double) -> String {
        let rounded = value.rounded()
        if abs(value - rounded) < 0.01 {
            return "\(Int(rounded))"
        }
        return String(format: "%.1f", value)
    }
}

struct GuidanceCard: View {
    let title: String
    let systemImage: String
    let bodyText: String

    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: systemImage)
                .foregroundStyle(.secondary)
                .frame(width: 18)
            VStack(alignment: .leading, spacing: 4) {
                Text(title)
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(bodyText)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(10)
        .background(Color.secondary.opacity(0.09))
        .clipShape(RoundedRectangle(cornerRadius: 8))
    }
}

struct QuantProfileGuideView: View {
    let profileName: String
    let profiles: Profiles

    private var guide: QuantProfileGuide {
        QuantizationGuidance.profileGuide(name: profileName, profiles: profiles)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(alignment: .firstTextBaseline) {
                Text(guide.name)
                    .font(.headline)
                Text(guide.title)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Text(guide.body)
                .font(.caption)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
            if let detail = guide.detail, !detail.isEmpty {
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.tertiary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 6) {
                ForEach(guide.badges, id: \.self) { badge in
                    Text(badge)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(RoundedRectangle(cornerRadius: 5))
                }
            }
        }
        .padding(.top, 4)
    }
}

struct QuantProfileCatalog: View {
    let family: Family
    let profiles: Profiles

    var body: some View {
        DisclosureGroup("What the profile names mean") {
            VStack(alignment: .leading, spacing: 10) {
                ForEach(QuantizationGuidance.tierGlossary) { item in
                    VStack(alignment: .leading, spacing: 2) {
                        Text(item.term)
                            .font(.caption)
                            .fontWeight(.semibold)
                        Text(item.explanation)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Divider()
                if family == .jang {
                    ForEach(profiles.jang) { profile in
                        ProfileCatalogRow(guide: QuantizationGuidance.profileGuide(name: profile.name, profiles: profiles))
                    }
                } else {
                    ForEach(profiles.jangtq) { profile in
                        ProfileCatalogRow(guide: QuantizationGuidance.profileGuide(name: profile.name, profiles: profiles))
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }
}

private struct ProfileCatalogRow: View {
    let guide: QuantProfileGuide

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text(guide.name)
                    .font(.caption)
                    .fontWeight(.semibold)
                Text(guide.title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(guide.body)
                .font(.caption2)
                .foregroundStyle(.secondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}
