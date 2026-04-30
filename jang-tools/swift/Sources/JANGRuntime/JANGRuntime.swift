//
//  JANGRuntime — Swift-side bundle loader. Pure metadata / config wiring;
//  weight materialization delegates to vmlx-swift's MLX binding once we
//  expose the JANGTQLinear / JANGLinear shims there.
//

import Foundation
import JANGQuant

public struct BundleHandle: Sendable {
    public let url: URL
    public let meta: QuantMeta
    public let modelType: String
    public let architectures: [String]
}

public enum BundleLoader {
    public static func open(at url: URL) throws -> BundleHandle {
        let cfgURL = url.appendingPathComponent("config.json")
        let data = try Data(contentsOf: cfgURL)
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        let arch = (json["architectures"] as? [String]) ?? []
        let mt = (json["text_config"] as? [String: Any])?["model_type"] as? String
              ?? json["model_type"] as? String ?? ""
        let meta = try BundleProbe.detect(at: url)
        return BundleHandle(url: url, meta: meta, modelType: mt, architectures: arch)
    }
}
