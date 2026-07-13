// JANGStudio/JANGStudio/Verify/VerifyCheck.swift
import Foundation

enum VerifyID: String, CaseIterable {
    case jangConfigExists, jangConfigFormat, schemaValid, capabilitiesPresent,
         chatTemplate, tokenizerFiles, shardsMatchIndex, vlPreprocessors,
         videoPreprocessors, generationConfig, layerCountSane,
         miniMaxCustomPy, tokenizerClassConcrete,
         // M116 (iter 40): feedback_model_checklist.md rule 2 — disk size
         // ≈ GPU RAM, no bloat. Warns when actual disk size differs ≥2×
         // from what the target bit-width + source size would predict.
         diskSizeSanity,
         reviewedPruneSource, reviewedPruneVerification, reviewedPruneSameSuite,
         reviewedPrunePlan, expertLabFinalReport, expertLabNativeSmoke,
         expertLabFinalComparison
}

struct VerifyCheck: Identifiable, Equatable {
    let id: VerifyID
    let title: String
    let status: PreflightStatus
    let required: Bool
    let hint: String?
}
