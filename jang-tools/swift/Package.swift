// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "JANG",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "JANGRuntime", targets: ["JANGRuntime"]),
        .library(name: "JANGQuant", targets: ["JANGQuant"]),
        .library(name: "JANGImage", targets: ["JANGImage"]),
        .library(name: "JANGDistributed", targets: ["JANGDistributed"]),
        .executable(name: "jang-probe", targets: ["JANGProbe"]),
    ],
    dependencies: [],
    targets: [
        .target(name: "JANGCxx",
                publicHeadersPath: "include",
                cxxSettings: [.unsafeFlags(["-std=c++17"])]),
        .target(name: "JANGQuant"),
        .target(name: "JANGRuntime", dependencies: ["JANGQuant", "JANGCxx"]),
        .target(name: "JANGImage"),
        .target(name: "JANGDistributed", dependencies: ["JANGRuntime"]),
        .executableTarget(name: "JANGProbe", dependencies: ["JANGDistributed"]),
        .testTarget(name: "JANGTests",
                    dependencies: ["JANGRuntime", "JANGQuant", "JANGImage", "JANGDistributed"],
                    path: "Tests/JANGTests"),
    ]
)
