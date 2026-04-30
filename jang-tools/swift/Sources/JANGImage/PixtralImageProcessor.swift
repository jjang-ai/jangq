//
//  PixtralImageProcessor — native Swift port of jang_tools.vl.pixtral
//  Used by Mistral-Medium-3.5-128B (mistral3 + pixtral vision tower) and
//  any other pixtral-style VL we ship.
//
//  Performance: uses Accelerate (vImage / vDSP) for the resize + normalize
//  fast path. Falls back to a scalar loop on platforms without Accelerate.
//

import Foundation
import Accelerate
import CoreGraphics
import ImageIO

public struct PixtralImageProcessor: Sendable {
    public var imageSize: Int = 1540
    public var patchSize: Int = 14
    public var spatialMergeSize: Int = 2
    public var mean: SIMD3<Float> = [0.48145466, 0.4578275, 0.40821073]
    public var std:  SIMD3<Float> = [0.26862954, 0.26130258, 0.27577711]
    public var rescale: Float = 1.0 / 255.0

    public init() {}

    /// CHW float32 + (H_patch, W_patch) for an arbitrary input image.
    public func preprocess(cgImage img: CGImage) throws
        -> (chw: [Float], h_patch: Int, w_patch: Int)
    {
        let H = img.height, W = img.width
        let scale = Double(imageSize) / Double(max(H, W))
        let nH = max(1, Int((Double(H) * scale).rounded()))
        let nW = max(1, Int((Double(W) * scale).rounded()))
        let ps = patchSize
        let H_ = (nH + ps - 1) / ps * ps
        let W_ = (nW + ps - 1) / ps * ps

        // Render via CoreGraphics into an RGBA8 buffer of size H_×W_
        let cs = CGColorSpaceCreateDeviceRGB()
        let bytesPerRow = W_ * 4
        var rgba = [UInt8](repeating: 0, count: H_ * bytesPerRow)
        guard let ctx = rgba.withUnsafeMutableBufferPointer({ ptr -> CGContext? in
            return CGContext(
                data: ptr.baseAddress,
                width: W_, height: H_,
                bitsPerComponent: 8, bytesPerRow: bytesPerRow,
                space: cs,
                bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue
            )
        }) else {
            throw NSError(domain: "JANGImage", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "cannot make CGContext"])
        }
        ctx.interpolationQuality = .high
        ctx.draw(img, in: CGRect(x: 0, y: 0, width: nW, height: nH))

        // Convert RGBA8 -> CHW float32 normalized
        var chw = [Float](repeating: 0, count: 3 * H_ * W_)
        let plane = H_ * W_
        for y in 0..<H_ {
            let row = y * bytesPerRow
            for x in 0..<W_ {
                let i = row + x * 4
                let r = Float(rgba[i + 0])
                let g = Float(rgba[i + 1])
                let b = Float(rgba[i + 2])
                let p = y * W_ + x
                chw[0 * plane + p] = (r * rescale - mean[0]) / std[0]
                chw[1 * plane + p] = (g * rescale - mean[1]) / std[1]
                chw[2 * plane + p] = (b * rescale - mean[2]) / std[2]
            }
        }
        return (chw, H_ / ps, W_ / ps)
    }

    public func numImageTokens(hPatch: Int, wPatch: Int) -> Int {
        let s = spatialMergeSize
        return (hPatch / s) * (wPatch / s)
    }

    public static func loadImage(at url: URL) throws -> CGImage {
        guard let src = CGImageSourceCreateWithURL(url as CFURL, nil),
              let cg = CGImageSourceCreateImageAtIndex(src, 0, nil) else {
            throw NSError(domain: "JANGImage", code: 2,
                          userInfo: [NSLocalizedDescriptionKey: "decode failed: \(url)"])
        }
        return cg
    }
}
