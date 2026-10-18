import Foundation
import XCTest
@testable import ResearchCore

final class ResearchMethodsTests: XCTestCase {
    let resources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources")
    func testOldSettingsRemainOffAndNewFlagsRoundTrip() throws {
        var old = try JSONDecoder().decode(SharedContextOptions.self, from: Data("{\"enabled\":true,\"agentCount\":5,\"intervalMinutes\":5}".utf8))
        XCTAssertFalse(old.usesMethods)
        var methods = try JSONDecoder().decode(ResearchMethodOptions.self, from: Data(contentsOf: resources.appendingPathComponent("ResearchMethodsDefaults.json")))
        XCTAssertFalse(methods.enabled)
        methods.rsiTop = true; methods.forkAndFlush = true; old.methods = methods
        let decoded = try JSONDecoder().decode(SharedContextOptions.self, from: JSONEncoder().encode(old))
        XCTAssertEqual(decoded, old)
        XCTAssertTrue(decoded.usesMethods)
        methods.branches = 1
        XCTAssertThrowsError(try methods.validate())
    }
    func testBranchCommandStartsFreshAndPreservesConfiguredModelAndMCP() throws {
        var options = try JSONDecoder().decode(ResearchMethodOptions.self, from: Data(contentsOf: resources.appendingPathComponent("ResearchMethodsDefaults.json")))
        options.forkAndFlush = true
        var settings = try JSONDecoder().decode(Settings.self, from: Data(contentsOf: resources.appendingPathComponent("Defaults.json")))
        settings.codexModel = "selected-model-fixture"
        let files = try ResearchMethods.files(options: options, provider: .codex, settings: settings, policy: "base policy", resources: resources, disabledMCPServers: ["private"], enabledMCPServers: ["chosen"])
        let config = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(files["config.json"])) as? [String: Any])
        let args = try XCTUnwrap(config["arguments"] as? [String])
        XCTAssertFalse(args.contains("resume"))
        XCTAssertTrue(args.contains("selected-model-fixture"))
        XCTAssertTrue(args.contains("mcp_servers.private.enabled=false"))
        XCTAssertTrue(args.contains("mcp_servers.chosen.enabled=true"))
        XCTAssertNotNil(files["ResearchMethods.py"])
        XCTAssertNotNil(files["ResearchMethodsPolicy.md"])
    }
    func testMethodsReceiveExplicitSharingConfigurationWithoutChangingFlagsOrCount() throws {
        var options = try JSONDecoder().decode(ResearchMethodOptions.self, from: Data(contentsOf: resources.appendingPathComponent("ResearchMethodsDefaults.json")))
        options.rsiTop = true; options.forkAndFlush = true; options.branches = 4
        let settings = try JSONDecoder().decode(Settings.self, from: Data(contentsOf: resources.appendingPathComponent("Defaults.json")))
        let sharing = SharedContextOptions(enabled: true, agentCount: 5, intervalMinutes: 5)
        let files = try ResearchMethods.files(options: options, provider: .codex, settings: settings, policy: "fixture", resources: resources, sharedContext: sharing)
        let config = try XCTUnwrap(JSONSerialization.jsonObject(with: XCTUnwrap(files["config.json"])) as? [String: Any])
        XCTAssertEqual(config["shareContext"] as? Bool, true)
        XCTAssertEqual(config["sharingIntervalMinutes"] as? Double, 5)
        let methods = try XCTUnwrap(config["options"] as? [String: Any])
        XCTAssertEqual(methods["branches"] as? Int, 4)
        XCTAssertEqual(methods["rsiTop"] as? Bool, true)
        XCTAssertEqual(methods["forkAndFlush"] as? Bool, true)
    }

}
