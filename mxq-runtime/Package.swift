// swift-tools-version: 6.0
// MLXQ Runtime — Swift + Metal Inference Engine for Apple Silicon
// Created by Eric Jang (eric@vmlx.net)

import PackageDescription

let package = Package(
    name: "MLXQRuntime",
    platforms: [
        .macOS(.v15),
    ],
    products: [
        .executable(name: "mlxq", targets: ["MXQCLI"]),
        .library(name: "MXQ", targets: ["MXQ"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MXQMetal",
            dependencies: [],
            path: "Sources/MXQMetal"
        ),
        .target(
            name: "MXQ",
            dependencies: ["MXQMetal"],
            path: "Sources/MXQ"
        ),
        .executableTarget(
            name: "MXQCLI",
            dependencies: [
                "MXQ",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MXQCLI"
        ),
        .testTarget(
            name: "MXQTests",
            dependencies: ["MXQ"],
            path: "Tests/MXQTests"
        ),
    ]
)
