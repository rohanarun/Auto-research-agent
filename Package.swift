// swift-tools-version: 6.0
import PackageDescription

let package = Package(
    name: "AutoResearch",
    platforms: [.macOS(.v14)],
    products: [.executable(name: "AutoResearch", targets: ["AutoResearch"])],
    targets: [
        .systemLibrary(name: "CArchive"),
        .target(name: "ResearchCore", dependencies: ["CArchive"]),
        .executableTarget(name: "AutoResearch", dependencies: ["ResearchCore"]),
        .testTarget(name: "ResearchCoreTests", dependencies: ["ResearchCore"]),
        .testTarget(name: "AutoResearchTests", dependencies: ["AutoResearch"])
    ],
    swiftLanguageModes: [.v5]
)
