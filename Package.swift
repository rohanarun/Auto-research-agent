// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AutoResearch",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "AutoResearch", targets: ["AutoResearch"])],
    targets: [
        .target(name: "ResearchCore"),
        .executableTarget(name: "AutoResearch", dependencies: ["ResearchCore"]),
        .testTarget(name: "ResearchCoreTests", dependencies: ["ResearchCore"])
    ],
    swiftLanguageModes: [.v5]
)
