// JANGStudio/JANGStudio/Verify/PreflightCheck.swift
import Foundation

enum PreflightID: String, CaseIterable {
    case sourceReadable, configJSONValid, outputUsable, diskSpace, ramAdequate,
         jangtqArchSupported, jangtqSourceDtype, bf16For512Experts, hadamardVsLowBits,
         bundledPythonHealthy, reviewedPruneVerified
}

enum PreflightStatus: String { case pass, warn, fail }

struct PreflightCheck: Identifiable, Equatable {
    let id: PreflightID
    let title: String
    let status: PreflightStatus
    let hint: String?
}
