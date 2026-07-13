import XCTest
@testable import JANGStudio

final class RecommendationServiceTests: XCTestCase {
    func testRecommendationDecodesSnakeCaseMoEAcronyms() throws {
        let json = """
        {
          "detected": {
            "model_type": "qwen3_5_moe",
            "family_class": "moe_hybrid_ssm",
            "param_count_billions": 35.0,
            "expert_count": 256,
            "is_moe": true,
            "is_vl": true,
            "is_video_vl": true,
            "source_dtype": "bfloat16",
            "has_tool_parser": true,
            "has_reasoning_parser": true,
            "is_gated_model": false,
            "name_or_path": "Qwen/Qwen3.6-35B-A3B"
          },
          "recommended": {
            "family": "jang",
            "profile": "JANG_4K",
            "method": "mse",
            "hadamard": true,
            "block_size": 64,
            "force_dtype": null,
            "alternatives": [
              {
                "family": "jangtq",
                "profile": "JANGTQ3",
                "use_when": "You want 30% smaller output."
              }
            ]
          },
          "beginner_summary": "This is a hybrid MoE model.",
          "warnings": [],
          "why_each_choice": {
            "family": "JANG is safe.",
            "profile": "JANG_4K is balanced.",
            "method": "MSE is high quality.",
            "hadamard": "Hadamard is enabled.",
            "block_size": "Block size 64 is default.",
            "force_dtype": "Use source dtype."
          }
        }
        """

        let rec = try JSONDecoder().decode(Recommendation.self, from: Data(json.utf8))
        XCTAssertEqual(rec.detected.modelType, "qwen3_5_moe")
        XCTAssertTrue(rec.detected.isMoE)
        XCTAssertTrue(rec.detected.isVL)
        XCTAssertTrue(rec.detected.isVideoVL)
        XCTAssertEqual(rec.recommended.blockSize, 64)
        XCTAssertNil(rec.recommended.forceDtype)
        XCTAssertEqual(rec.recommended.alternatives.first?.useWhen, "You want 30% smaller output.")
    }
}
