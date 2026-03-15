/*
 * Safetensors File Reader for Swift
 * Created by Eric Jang (eric@vmlx.net)
 *
 * Reads safetensors files via memory mapping for zero-copy loading.
 * On Apple Silicon with unified memory, mmap'd data is directly
 * accessible by the GPU — no CPU→GPU copy needed.
 *
 * Format:
 *   [8 bytes: header_size (uint64 LE)]
 *   [header_size bytes: JSON header]
 *   [remaining: tensor data]
 *
 * The JSON header maps tensor names to {dtype, shape, data_offsets}.
 */

import Foundation
import Metal

/// Metadata for a single tensor in a safetensors file.
public struct TensorInfo: Sendable {
    public let name: String
    public let dtype: String        // "F16", "BF16", "F32", "U8", "I32", etc.
    public let shape: [Int]
    public let dataOffset: Int      // byte offset from start of data section
    public let dataLength: Int      // byte length of tensor data
}

/// A memory-mapped safetensors file.
public final class SafetensorsFile: @unchecked Sendable {
    public let url: URL
    public let tensors: [String: TensorInfo]
    private let fileHandle: FileHandle
    private let mappedData: Data
    private let dataStartOffset: Int  // offset where tensor data begins

    public init(url: URL) throws {
        self.url = url

        // Memory-map the entire file
        self.fileHandle = try FileHandle(forReadingFrom: url)
        guard let data = try fileHandle.availableData as Data?,
              data.count > 8 else {
            throw MXQError.safetensorsError("File too small: \(url.lastPathComponent)")
        }

        // Read header size (first 8 bytes, little-endian uint64)
        let headerSize = data.withUnsafeBytes { ptr in
            ptr.load(as: UInt64.self)
        }

        let headerEnd = 8 + Int(headerSize)
        guard data.count >= headerEnd else {
            throw MXQError.safetensorsError("File truncated: header says \(headerSize) bytes")
        }

        // Parse JSON header
        let headerData = data[8..<headerEnd]
        guard let headerDict = try JSONSerialization.jsonObject(with: headerData) as? [String: Any] else {
            throw MXQError.safetensorsError("Invalid JSON header")
        }

        // Build tensor info map
        var tensorMap: [String: TensorInfo] = [:]
        for (name, value) in headerDict {
            // Skip __metadata__ key
            if name == "__metadata__" { continue }

            guard let info = value as? [String: Any],
                  let dtype = info["dtype"] as? String,
                  let shape = info["shape"] as? [Int],
                  let offsets = info["data_offsets"] as? [Int],
                  offsets.count == 2 else {
                continue
            }

            let dataOffset = offsets[0]
            let dataLength = offsets[1] - offsets[0]

            tensorMap[name] = TensorInfo(
                name: name,
                dtype: dtype,
                shape: shape,
                dataOffset: dataOffset,
                dataLength: dataLength
            )
        }

        self.tensors = tensorMap
        self.dataStartOffset = headerEnd

        // Re-map the file properly for GPU access
        // Use mmap for zero-copy — the data stays on disk until accessed
        self.mappedData = try Data(contentsOf: url, options: .mappedIfSafe)
    }

    /// Get raw bytes for a tensor (zero-copy pointer into mmap'd file).
    public func tensorData(name: String) throws -> Data {
        guard let info = tensors[name] else {
            throw MXQError.tensorNotFound(name)
        }

        let start = dataStartOffset + info.dataOffset
        let end = start + info.dataLength

        guard end <= mappedData.count else {
            throw MXQError.safetensorsError("Tensor \(name) data extends beyond file")
        }

        return mappedData[start..<end]
    }

    /// Get a pointer to tensor data for creating Metal buffers.
    public func tensorPointer(name: String) throws -> (UnsafeRawPointer, Int) {
        guard let info = tensors[name] else {
            throw MXQError.tensorNotFound(name)
        }

        let start = dataStartOffset + info.dataOffset
        let length = info.dataLength

        return mappedData.withUnsafeBytes { basePtr in
            let ptr = basePtr.baseAddress!.advanced(by: start)
            return (ptr, length)
        }
    }

    /// Create a Metal buffer directly from mmap'd tensor data (zero-copy on Apple Silicon).
    public func makeMetalBuffer(name: String, device: MTLDevice) throws -> MTLBuffer {
        guard let info = tensors[name] else {
            throw MXQError.tensorNotFound(name)
        }

        let start = dataStartOffset + info.dataOffset
        let length = info.dataLength

        // We need the data to stay alive, so copy into a buffer
        // (True zero-copy via bytesNoCopy requires page-aligned data
        // which safetensors doesn't guarantee)
        let data = try tensorData(name: name)
        guard let buffer = data.withUnsafeBytes({ ptr in
            device.makeBuffer(bytes: ptr.baseAddress!, length: length, options: .storageModeShared)
        }) else {
            throw MXQError.bufferAllocationFailed(length)
        }

        return buffer
    }

    /// List all tensor names.
    public var tensorNames: [String] {
        return Array(tensors.keys).sorted()
    }

    /// Total number of tensors.
    public var count: Int {
        return tensors.count
    }

    deinit {
        try? fileHandle.close()
    }
}

/// Load all safetensors shards from a model directory.
public func loadSafetensorsShards(from directory: URL) throws -> [SafetensorsFile] {
    let fm = FileManager.default
    let contents = try fm.contentsOfDirectory(at: directory,
                                               includingPropertiesForKeys: nil)
    let safetensorFiles = contents
        .filter { $0.pathExtension == "safetensors" }
        .sorted { $0.lastPathComponent < $1.lastPathComponent }

    guard !safetensorFiles.isEmpty else {
        throw MXQError.modelNotFound("No .safetensors files in \(directory.path)")
    }

    return try safetensorFiles.map { try SafetensorsFile(url: $0) }
}
