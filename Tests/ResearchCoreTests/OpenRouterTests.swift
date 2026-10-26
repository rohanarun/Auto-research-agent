import XCTest
@testable import ResearchCore

final class OpenRouterTests: XCTestCase {
    private var resources: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("Resources") }
    private func settings() throws -> Settings { try JSONDecoder().decode(Settings.self, from: Data(contentsOf: resources.appendingPathComponent("Defaults.example.json"))) }

    func testGatewayOffPreservesDirectCommandAndOpenCodeNeverReceivesKey() throws {
        let config = OpenRouterConfiguration(enabled: true, codexModel: "fixture/model")
        let direct = CLI.chat(provider: .codex, settings: try settings(), sessionID: nil, prompt: "x", policy: "p")
        XCTAssertFalse(direct.arguments.contains { $0.contains("openrouter") })
        let openCode = CLI.chat(provider: .opencode, settings: try settings(), sessionID: nil, prompt: "x", policy: "p", gateway: config)
        XCTAssertFalse(openCode.arguments.contains { $0.contains("openrouter") })
        XCTAssertTrue(config.environment(provider: .opencode, key: "fixture-secret", claudeConfigDirectory: "/unused").isEmpty)
    }
    func testCodexUsesPerRunCommandAuthAndEditableModelWithoutSecretInArguments() throws {
        let config = OpenRouterConfiguration(enabled: true, codexModel: "fixture/model")
        let command = CLI.chat(provider: .codex, settings: try settings(), sessionID: nil, prompt: "x", policy: "p", gateway: config)
        XCTAssertEqual(command.arguments.first, "--no-daemon")
        XCTAssertTrue(command.arguments.contains("fixture/model"))
        let flags = command.arguments.joined(separator: " ")
        XCTAssertTrue(flags.contains("/usr/bin/printenv"))
        XCTAssertTrue(flags.contains("OPENROUTER_API_KEY"))
        XCTAssertFalse(flags.contains("env_key"))
        XCTAssertFalse(flags.contains("fixture-secret"))
        XCTAssertEqual(config.environment(provider: .codex, key: "fixture-secret", claudeConfigDirectory: ""), ["OPENROUTER_API_KEY": "fixture-secret"])
        try config.validate(for: .codex)
        XCTAssertThrowsError(try config.validate(for: .claude))
    }
    func testClaudeEnvironmentIsIsolatedAndOverridesDirectAuthentication() {
        let config = OpenRouterConfiguration(enabled: true, claudeModel: "fixture/claude")
        let env = config.environment(provider: .claude, key: "fixture-secret", claudeConfigDirectory: "/Volumes/fixture/gateway/claude-profile")
        XCTAssertEqual(env["ANTHROPIC_AUTH_TOKEN"], "fixture-secret")
        XCTAssertEqual(env["ANTHROPIC_API_KEY"], "")
        XCTAssertEqual(env["CLAUDE_CODE_OAUTH_TOKEN"], "")
        XCTAssertEqual(env["CLAUDE_CONFIG_DIR"], "/Volumes/fixture/gateway/claude-profile")
        XCTAssertEqual(env["CLAUDE_CODE_SUBAGENT_MODEL"], "fixture/claude")
        XCTAssertNil(env["OPENROUTER_API_KEY"])
    }
    func testLegacyChatsStayDirectAndRouteChangesRequireNewChat() throws {
        let config = OpenRouterConfiguration(enabled: true, codexModel: "fixture/model")
        XCTAssertThrowsError(try config.validateResume(provider: .codex, savedRoute: nil, hasProviderSession: true))
        XCTAssertNoThrow(try config.validateResume(provider: .codex, savedRoute: nil, hasProviderSession: false))
        XCTAssertNoThrow(try config.validateResume(provider: .codex, savedRoute: "openrouter:codex", hasProviderSession: true))
        XCTAssertThrowsError(try OpenRouterConfiguration().validateResume(provider: .codex, savedRoute: "openrouter:codex", hasProviderSession: true))
    }
    func testMethodBranchesKeepGatewayArgumentsButNoCredentialsInConfig() throws {
        var options = try JSONDecoder().decode(ResearchMethodOptions.self, from: Data(contentsOf: resources.appendingPathComponent("ResearchMethodsDefaults.json")))
        options.forkAndFlush = true
        let files = try ResearchMethods.files(options: options, provider: .codex, settings: settings(), policy: "p", resources: resources, gateway: .init(enabled: true, codexModel: "fixture/model"))
        let value = try XCTUnwrap(files["config.json"])
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: value) as? [String: Any])
        let args = try XCTUnwrap(json["arguments"] as? [String])
        XCTAssertEqual(args.filter { $0 == "--no-daemon" }.count, 1)
        XCTAssertTrue(args.contains("fixture/model"))
        XCTAssertFalse(String(decoding: value, as: UTF8.self).contains("fixture-secret"))
        XCTAssertNil(json["environment"])
    }
}
