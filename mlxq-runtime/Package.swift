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
        .executable(name: "mlxq", targets: ["MLXQCLI"]),
        .library(name: "MLXQ", targets: ["MLXQ"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "MLXQMetal",
            dependencies: [],
            path: "Sources/MLXQMetal"
        ),
        .target(
            name: "MLXQ",
            dependencies: ["MLXQMetal"],
            path: "Sources/MLXQ"
        ),
        .executableTarget(
            name: "MLXQCLI",
            dependencies: [
                "MLXQ",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            path: "Sources/MLXQCLI"
        ),
        .testTarget(
            name: "MLXQTests",
            dependencies: ["MLXQ"],
            path: "Tests/MLXQTests"
        ),
    ]
)
