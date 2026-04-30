//
//  jang-probe — CLI front-end for the TB5/RDMA probe.
//
//  Usage:  jang-probe --hostfile /path/to/hostfile.json [--rounds 5]
//

import Foundation
import JANGDistributed

@main
struct ProbeMain {
    static func main() throws {
        var args = CommandLine.arguments.dropFirst()
        var hostfile = URL(fileURLWithPath:
            ProcessInfo.processInfo.environment["JANG_HOSTFILE"] ?? "hostfile.json")
        var rounds = 5
        while let a = args.first {
            args = args.dropFirst()
            switch a {
            case "--hostfile":
                if let v = args.first { hostfile = URL(fileURLWithPath: v); args = args.dropFirst() }
            case "--rounds":
                if let v = args.first, let n = Int(v) { rounds = n; args = args.dropFirst() }
            default:
                FileHandle.standardError.write(Data("unknown arg \(a)\n".utf8))
                exit(2)
            }
        }
        try TB5Probe.runViaPython(hostfile: hostfile, rounds: rounds)
    }
}
