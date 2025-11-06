// swift-tools-version: 5.9
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MLXR",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "MLXR",
            targets: ["MLXR"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "MLXR",
            dependencies: [],
            path: "Sources/MLXR"
        ),
        .testTarget(
            name: "MLXRTests",
            dependencies: ["MLXR"],
            path: "Tests/MLXRTests"
        ),
    ]
)
