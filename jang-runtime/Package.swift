// swift-tools-version: 6.0
// JANG Runtime — Jang Adaptive N-bit Grading
// Created by Eric Jang (eric@vmlx.net)

import PackageDescription

let package = Package(
    name: "JANGRuntime",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(name: "jang", targets: ["JANGCLI"]),
        .executable(name: "jang-spec-iobench", targets: ["JangSpecIOBench"]),
        .executable(name: "jang-core", targets: ["JangCoreCLI"]),
        .library(name: "JANG", targets: ["JANG"]),
        .library(name: "JANGKit", targets: ["JANGKit"]),
        .library(name: "JANGExpertLab", targets: ["JANGExpertLab"]),
        .library(name: "JANGCore", targets: ["JANGCore"]),
        .library(name: "JANGCoreMetal", targets: ["JANGCoreMetal"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(name: "JANGMetal", dependencies: [], path: "Sources/JANGMetal"),
        .target(name: "JANG", dependencies: ["JANGMetal", "JANGCoreMetal"], path: "Sources/JANG"),
        .target(name: "JANGKit", dependencies: ["JANG"], path: "Sources/JANGKit"),
        .target(name: "JANGExpertLab", dependencies: ["JANGKit", "JANG"], path: "Sources/JANGExpertLab"),
        .target(name: "JANGCore", dependencies: [], path: "Sources/JANGCore"),
        .target(
            name: "JANGCoreMetal",
            dependencies: ["JANGCore"],
            path: "Sources/JANGCoreMetal",
            resources: [
                .copy("JangV2QuantMatmul.metal"),
                .copy("JANGTQMatmul.metal"),
                .copy("JANGTQAffine8Matmul.metal"),
                .copy("JANGTQDecodeOps.metal"),
            ]
        ),
        .executableTarget(
            name: "JANGCLI",
            dependencies: ["JANG", "JANGCoreMetal", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/JANGCLI"
        ),
        .executableTarget(
            name: "JangSpecIOBench",
            dependencies: [],
            path: "Sources/jang-spec-iobench"
        ),
        .executableTarget(
            name: "JangCoreCLI",
            dependencies: ["JANGCore", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/jang-core"
        ),
        .testTarget(name: "JANGTests", dependencies: ["JANG", "JANGCoreMetal"], path: "Tests/JANGTests"),
        .testTarget(name: "JANGKitTests", dependencies: ["JANGKit"], path: "Tests/JANGKitTests"),
        .testTarget(name: "JANGExpertLabTests", dependencies: ["JANGExpertLab"], path: "Tests/JANGExpertLabTests"),
        .testTarget(name: "JANGCoreTests", dependencies: ["JANGCore"], path: "Tests/JANGCoreTests"),
        .testTarget(
            name: "JANGCoreMetalTests",
            dependencies: ["JANGCoreMetal"],
            path: "Tests/JANGCoreMetalTests",
            resources: [
                .copy("fixtures")
            ]
        ),
    ]
)
