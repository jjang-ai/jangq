/*
 * MXQ Error Types
 * Created by Eric Jang (eric@vmlx.net)
 */

import Foundation

public enum JANGError: Error, CustomStringConvertible {
    case metalNotAvailable
    case metallibNotFound
    case kernelNotFound(String)
    case modelNotFound(String)
    case invalidFormat(String)
    case configMissing(String)
    case tensorNotFound(String)
    case safetensorsError(String)
    case tokenizerError(String)
    case bufferAllocationFailed(Int)
    case shapeError(String)
    case inferenceError(String)

    public var description: String {
        switch self {
        case .metalNotAvailable:
            return "Metal GPU not available"
        case .metallibNotFound:
            return "jang.metallib not found — ensure Metal shaders are compiled"
        case .kernelNotFound(let name):
            return "Metal kernel '\(name)' not found in metallib"
        case .modelNotFound(let path):
            return "Model not found: \(path)"
        case .invalidFormat(let msg):
            return "Invalid MXQ format: \(msg)"
        case .configMissing(let file):
            return "Config file missing: \(file)"
        case .tensorNotFound(let name):
            return "Tensor not found: \(name)"
        case .safetensorsError(let msg):
            return "Safetensors error: \(msg)"
        case .tokenizerError(let msg):
            return "Tokenizer error: \(msg)"
        case .bufferAllocationFailed(let bytes):
            return "Failed to allocate Metal buffer: \(bytes) bytes"
        case .shapeError(let msg):
            return "Shape error: \(msg)"
        case .inferenceError(let msg):
            return "Inference error: \(msg)"
        }
    }
}
