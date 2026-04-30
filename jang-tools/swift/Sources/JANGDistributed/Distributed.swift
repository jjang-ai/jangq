//
//  JANGDistributed — distributed inference primitives for the two-node rig
//  (M3 Ultra Studio + M4 Max MacBook over Thunderbolt 5 RDMA).
//
//  Mirrors the Python jang_tools.distributed package. Once vmlx-swift exposes
//  its mlx-core distributed Group binding, the bodies below switch from the
//  TCP fallback to jaccl in one place (Backend.preferred).
//

import Foundation

public enum Backend: String, Sendable {
    case jaccl
    case ring
}

public struct Hostfile: Codable, Sendable {
    public var entries: [Entry]
    public struct Entry: Codable, Sendable {
        public var ssh: String
        public var ips: [String]
    }
    public init(entries: [Entry]) { self.entries = entries }
}

public actor World {
    public let rank: Int
    public let size: Int
    public let backend: Backend
    public init(rank: Int, size: Int, backend: Backend) {
        self.rank = rank; self.size = size; self.backend = backend
    }
    public var isMain: Bool { rank == 0 }
}

public enum DistributedInit {
    /// Initialize the distributed group, preferring jaccl over ring.
    /// Reads MLX env vars set by `mlx.launch`.
    public static func initWorld(prefer: [Backend] = [.jaccl, .ring]) throws -> World {
        let env = ProcessInfo.processInfo.environment
        let rank = Int(env["MLX_RANK"] ?? env["RANK"] ?? "0") ?? 0
        let size = Int(env["MLX_WORLD_SIZE"] ?? env["WORLD_SIZE"] ?? "1") ?? 1
        let backend = Backend(rawValue: env["MLX_DISTRIBUTED_BACKEND"] ?? "")
            ?? prefer.first ?? .ring
        return World(rank: rank, size: size, backend: backend)
    }
}

public enum HostfileLoader {
    public static func load(at url: URL) throws -> Hostfile {
        let data = try Data(contentsOf: url)
        let arr = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] ?? []
        let entries: [Hostfile.Entry] = try arr.map { d in
            guard let ssh = d["ssh"] as? String,
                  let ips = d["ips"] as? [String] else {
                throw NSError(domain: "JANGDistributed", code: 1,
                              userInfo: [NSLocalizedDescriptionKey: "bad hostfile entry"])
            }
            return Hostfile.Entry(ssh: ssh, ips: ips)
        }
        return Hostfile(entries: entries)
    }
}

/// EP / PP plan mirroring jang_tools.distributed.sharding.ShardPlan.
public struct ShardPlan: Sendable {
    public let rank: Int
    public let worldSize: Int
    public let expertOwner: [Int]    // length = nExperts
    public let layerOwner: [Int]     // -1 = replicated
    public init(rank: Int, worldSize: Int, expertOwner: [Int], layerOwner: [Int]) {
        self.rank = rank; self.worldSize = worldSize
        self.expertOwner = expertOwner; self.layerOwner = layerOwner
    }
    public var myExperts: [Int] {
        zip(expertOwner.indices, expertOwner).compactMap { $1 == rank ? $0 : nil }
    }
}

public enum ShardPlanner {
    /// Even split weighted by per-rank RAM ratio.
    public static func evenExperts(nExperts: Int, ramWeights: [Double]) -> [Int] {
        let total = ramWeights.reduce(0, +)
        var cuts = [0]; var acc = 0.0
        for w in ramWeights.dropLast() {
            acc += w
            cuts.append(Int((acc / total * Double(nExperts)).rounded()))
        }
        cuts.append(nExperts)
        var owner = Array(repeating: 0, count: nExperts)
        for (r, (a, b)) in zip(cuts.dropLast(), cuts.dropFirst()).enumerated() {
            for e in a..<b { owner[e] = r }
        }
        return owner
    }
}
