//
//  TB5Probe — bandwidth + latency probe stub. Mirrors tb5_probe.py.
//  Real comms wire-up plugs into mlx-swift's distributed binding when
//  available; until then this drives the Python probe via a subprocess.
//

import Foundation

public enum TB5Probe {
    public static func runViaPython(hostfile: URL, rounds: Int = 5) throws {
        let p = Process()
        p.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        p.arguments = [
            "mlx.launch", "--verbose",
            "--hostfile", hostfile.path,
            "-m", "jang_tools.distributed.tb5_probe",
            "--rounds", String(rounds),
        ]
        // Pipe drain pattern (per feedback_pipe_drain_pattern):
        // detach a Task that drains stdout/stderr to console.
        let stdout = Pipe(); p.standardOutput = stdout
        let stderr = Pipe(); p.standardError = stderr
        try p.run()
        Task.detached {
            for try await line in stdout.fileHandleForReading.bytes.lines { print(line) }
        }
        Task.detached {
            for try await line in stderr.fileHandleForReading.bytes.lines {
                FileHandle.standardError.write(Data((line + "\n").utf8))
            }
        }
        p.waitUntilExit()
    }
}
