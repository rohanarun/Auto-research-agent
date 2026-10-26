import AppKit
import SwiftUI
import XCTest
import ResearchCore
@testable import AutoResearch

private final class EmptyGatewayKeychain: OpenRouterCredentialStore {
    func containsKey() -> Bool { false }
    func read() throws -> String { throw ResearchError("No test credential") }
    func save(_ value: String) throws { XCTFail("UI render must not save credentials") }
    func remove() throws { XCTFail("UI render must not remove credentials") }
}

final class GatewaySettingsViewTests: XCTestCase {
    @MainActor func testGatewaySettingsRenderWithoutAccountOrNetwork() throws {
        _ = NSApplication.shared
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = try Workspace.directory(root.path, "tmp/gateway-ui-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: folder) }
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.example.json")))
        settings.workspaceRoot = folder.path
        settings.codexPath = "/usr/bin/false"; settings.claudePath = "/usr/bin/false"
        let store = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: settings, gatewayCredentials: EmptyGatewayKeychain())
        let host = NSHostingView(rootView: GatewaySettingsView(store: store).padding(24).background(Color(NSColor.windowBackgroundColor)).preferredColorScheme(.light))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 660), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        XCTAssertFalse(store.gateway.enabled)
        XCTAssertFalse(store.gatewayKeyConfigured)
        XCTAssertTrue(store.runners.isEmpty)
        let output = try Workspace.directory(root.path, "evidence/gateway-settings")
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: output.appendingPathComponent("gateway-settings.png"))
    }
}
