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
        .library(name: "JANG", targets: ["JANG"]),
    ],
    dependencies: [
        .package(url: "https://github.com/apple/swift-argument-parser.git", from: "1.3.0"),
    ],
    targets: [
        .target(name: "JANGMetal", dependencies: [], path: "Sources/JANGMetal"),
        .target(name: "JANG", dependencies: ["JANGMetal"], path: "Sources/JANG"),
        .executableTarget(
            name: "JANGCLI",
            dependencies: ["JANG", .product(name: "ArgumentParser", package: "swift-argument-parser")],
            path: "Sources/JANGCLI"
        ),
        .testTarget(name: "JANGTests", dependencies: ["JANG"], path: "Tests/JANGTests"),
    ]
)
