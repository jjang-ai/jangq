/*
 * MXQ Metal Device Management
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Manages the Metal device, command queue, compute pipelines,
 * and the compiled metallib containing all MXQ kernels.
 */

import Metal
import Foundation

/// Manages all Metal resources for MXQ inference.
public final class MXQMetalDevice: @unchecked Sendable {
    public let device: MTLDevice
    public let commandQueue: MTLCommandQueue
    private let library: MTLLibrary

    // Cached compute pipeline states
    private var pipelines: [String: MTLComputePipelineState] = [:]
    private let pipelineLock = NSLock()

    public init() throws {
        guard let device = MTLCreateSystemDefaultDevice() else {
            throw MXQError.metalNotAvailable
        }
        self.device = device

        guard let queue = device.makeCommandQueue() else {
            throw MXQError.metalNotAvailable
        }
        self.commandQueue = queue

        // Load the pre-compiled metallib
        self.library = try Self.loadMetalLibrary(device: device)

        // Pre-compile all pipeline states
        try self.precompilePipelines()
    }

    /// Get a cached compute pipeline state by kernel name.
    public func pipeline(for kernelName: String) throws -> MTLComputePipelineState {
        pipelineLock.lock()
        defer { pipelineLock.unlock() }

        if let existing = pipelines[kernelName] {
            return existing
        }

        guard let function = library.makeFunction(name: kernelName) else {
            throw MXQError.kernelNotFound(kernelName)
        }

        let pipeline = try device.makeComputePipelineState(function: function)
        pipelines[kernelName] = pipeline
        return pipeline
    }

    /// Create a buffer from existing memory (zero-copy for mmap'd data).
    public func makeBuffer(bytesNoCopy pointer: UnsafeMutableRawPointer,
                           length: Int) -> MTLBuffer? {
        return device.makeBuffer(bytesNoCopy: pointer,
                                 length: length,
                                 options: .storageModeShared,
                                 deallocator: nil)
    }

    /// Create a new buffer with data.
    public func makeBuffer<T>(from array: [T]) -> MTLBuffer? {
        let length = array.count * MemoryLayout<T>.stride
        return device.makeBuffer(bytes: array,
                                 length: length,
                                 options: .storageModeShared)
    }

    /// Create an empty buffer of given size.
    public func makeBuffer(length: Int) -> MTLBuffer? {
        return device.makeBuffer(length: length, options: .storageModeShared)
    }

    /// Device info for logging.
    public var deviceInfo: String {
        let name = device.name
        let ram = device.recommendedMaxWorkingSetSize / (1024 * 1024 * 1024)
        return "\(name) (\(ram) GB)"
    }

    // MARK: - Private

    private static func loadMetalLibrary(device: MTLDevice) throws -> MTLLibrary {
        // Search paths for the metallib
        let searchPaths: [String] = [
            // Next to the executable
            URL(fileURLWithPath: CommandLine.arguments[0])
                .deletingLastPathComponent()
                .appendingPathComponent("mxq.metallib").path,
            // In the build directory
            URL(fileURLWithPath: CommandLine.arguments[0])
                .deletingLastPathComponent()
                .deletingLastPathComponent()
                .appendingPathComponent("Metal/mxq.metallib").path,
            // Current working directory
            FileManager.default.currentDirectoryPath + "/mxq.metallib",
            // Project Metal directory
            FileManager.default.currentDirectoryPath + "/Metal/mxq.metallib",
        ]

        for path in searchPaths {
            if FileManager.default.fileExists(atPath: path) {
                return try device.makeLibrary(URL: URL(fileURLWithPath: path))
            }
        }

        // Last resort: compile from default Metal library
        if let defaultLib = device.makeDefaultLibrary() {
            return defaultLib
        }

        throw MXQError.metallibNotFound
    }

    private func precompilePipelines() throws {
        let kernelNames = [
            "mxq_dequantize",
            "mxq_dequant_gemv",
            "mxq_dequant_gemm",
            "mxq_rms_norm",
            "mxq_rope",
            "mxq_softmax",
            "mxq_silu",
            "mxq_silu_mul",
            "mxq_add",
            "mxq_embedding",
        ]

        for name in kernelNames {
            _ = try pipeline(for: name)
        }
    }
}
