import XCTest
@testable import ResearchCore

final class ResearchCoreTests: XCTestCase {
    var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func settings() throws -> Settings { try JSONDecoder().decode(Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.json"))) }

    func testLiveCatalogSnapshotIncludesCardsAndEveryTrack() throws {
        let html = try String(contentsOf: root.appendingPathComponent("Resources/catalog-source.html"), encoding: .utf8)
        let projects = try Catalog.parse(html)
        let repos = Set(projects.map(\.repository))
        for match in Catalog.matches("\\brepo[\"']?\\s*:\\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']", html) { XCTAssertTrue(repos.contains(match[1]), match[1]) }
        for repo in ["CrocSwap/integer-mult-bounds", "shea256/fourier-transform-below-nlogn", "karpathy/autoresearch", "Layr-Labs/heesch", "sjtu-sai-agents/PhysMaster", "Argonaut-Math/argonaut-math-quasi-riemann-boundary", "rohanarun/sidorenko-template-refinements"] { XCTAssertTrue(repos.contains(repo), repo) }
        XCTAssertEqual(projects.count, repos.count)
        XCTAssertFalse(repos.contains("user-attachments/assets"))
        XCTAssertFalse(projects.first(where: { $0.repository == "Layr-Labs/heesch" })!.summary.isEmpty)
        print("Catalog verified: \(projects.count) repositories\n" + projects.map(\.repository).joined(separator: "\n"))
    }
    func testCatalogFailureDoesNotBecomeAnEmptySuccess() { XCTAssertThrowsError(try Catalog.parse("<html>Service unavailable</html>")) }
    func testCodexResumeKeepsPromptOnStdinAndSessionExplicit() throws {
        let command = CLI.chat(provider: .codex, settings: try settings(), sessionID: "specific-session", prompt: "literal $(echo surprise) `text`", policy: "policy")
        XCTAssertTrue(command.arguments.contains("resume")); XCTAssertTrue(command.arguments.contains("specific-session"))
        XCTAssertEqual(command.arguments.last, "-")
        XCTAssertFalse(command.arguments.contains("--last"))
        XCTAssertFalse(command.arguments.contains("--dangerously-bypass-approvals-and-sandbox"))
        XCTAssertTrue(command.input!.contains("$(echo surprise)"))
    }
    func testClaudeResumeDoesNotSwitchModelsOrBypassPermissions() throws {
        let command = CLI.chat(provider: .claude, settings: try settings(), sessionID: "specific-session", prompt: "request", policy: "policy")
        XCTAssertTrue(command.arguments.contains("--resume")); XCTAssertTrue(command.arguments.contains("specific-session"))
        XCTAssertFalse(command.arguments.contains("--model")); XCTAssertFalse(command.arguments.contains("--dangerously-skip-permissions"))
        XCTAssertEqual(command.input, "request")
    }
    func testOpenCodeResumeKeepsPromptOnStdinAndDoesNotAutoApproveByDefault() throws {
        var config = try settings(); config.opencodePath = "/usr/local/bin/opencode"
        let command = CLI.chat(provider: .opencode, settings: config, sessionID: "ses_specific", prompt: "literal $(echo surprise)", policy: "policy")
        XCTAssertEqual(command.executable, "/usr/local/bin/opencode")
        XCTAssertEqual(Array(command.arguments.prefix(3)), ["run", "--format", "json"])
        XCTAssertEqual(command.arguments.firstIndex(of: "--session").map { command.arguments[$0 + 1] }, "ses_specific")
        XCTAssertFalse(command.arguments.contains("--auto")); XCTAssertFalse(command.arguments.contains("--model"))
        XCTAssertTrue(command.input!.hasPrefix("Harness instructions:\npolicy")); XCTAssertTrue(command.input!.contains("$(echo surprise)"))
        config.opencodePermission = "auto"; config.opencodeModel = "provider/model-id"
        let opted = CLI.chat(provider: .opencode, settings: config, sessionID: nil, prompt: "research", policy: "policy")
        XCTAssertTrue(opted.arguments.contains("--auto")); XCTAssertFalse(opted.arguments.contains("--session"))
        XCTAssertEqual(opted.arguments.firstIndex(of: "--model").map { opted.arguments[$0 + 1] }, "provider/model-id")
        XCTAssertEqual(CLI.resumeArguments(.opencode, sessionID: "ses_x"), ["--session", "ses_x"])
    }
    func testOpenCodeDecoderTracksSessionToolsAndTurnEnd() {
        let text = EventDecoder.decode(#"{"type":"text","timestamp":1,"sessionID":"ses_a","part":{"id":"prt_1","type":"text","text":"Done.\nRESEARCH_STATUS: continue"}}"#, provider: .opencode)
        XCTAssertEqual(text, [.session("ses_a"), .message(id: "prt_1", role: "assistant", text: "Done.\nRESEARCH_STATUS: continue")])
        let tool = EventDecoder.decode(#"{"type":"tool_use","timestamp":1,"sessionID":"ses_a","part":{"id":"prt_2","type":"tool","tool":"bash","state":{"status":"completed","input":{"command":"ls"},"output":"README.md"}}}"#, provider: .opencode)
        XCTAssertEqual(tool, [.session("ses_a"), .message(id: "tool-prt_2", role: "tool", text: "bash\ncompleted\n{\n  \"command\" : \"ls\"\n}\nREADME.md")])
        XCTAssertFalse(EventDecoder.decode(#"{"type":"step_finish","sessionID":"ses_a","part":{"type":"step-finish","reason":"tool-calls"}}"#, provider: .opencode).contains(.completed))
        XCTAssertTrue(EventDecoder.decode(#"{"type":"step_finish","sessionID":"ses_a","part":{"type":"step-finish","reason":"stop"}}"#, provider: .opencode).contains(.completed))
        // Shape observed from opencode 1.19 when the provider rejects a request.
        let failure = EventDecoder.decode(#"{"type":"error","timestamp":1,"sessionID":"ses_a","error":{"name":"APIError","data":{"message":"Forbidden","statusCode":403}}}"#, provider: .opencode)
        XCTAssertTrue(failure.contains(.failure("Forbidden"))); XCTAssertFalse(failure.contains(.completed))
        XCTAssertEqual(EventDecoder.decode(#"{"type":"error","sessionID":"","error":{"type":"unknown","message":"You must provide a message"}}"#, provider: .opencode), [.failure("You must provide a message")])
    }
    func testOpenCodeSignInStatusReadsCredentialList() {
        XCTAssertFalse(CLI.opencodeSignedIn("No authenticated integrations\n"))
        XCTAssertFalse(CLI.opencodeSignedIn("┌  Credentials ~/.local/share/opencode/auth.json\n│\n└  0 credentials\n"))
        XCTAssertTrue(CLI.opencodeSignedIn("┌  Credentials ~/.local/share/opencode/auth.json\n│\n└  0 credentials\n\n┌  Environment\n│\n●  Anthropic ANTHROPIC_API_KEY\n└  1 environment variable\n"))
        XCTAssertTrue(CLI.opencodeSignedIn("Anthropic   Claude Pro/Max            stored\n"))
    }
    func testAntigravityCommandCarriesPolicyInPrintArgumentAndResumesExplicitly() throws {
        var config = try settings()
        config.antigravityPath = "/usr/local/bin/agy"
        config.antigravityModel = ""
        config.antigravityPermission = "review"
        let command = CLI.chat(provider: .antigravity, settings: config, sessionID: "conv_specific", prompt: "literal $(echo surprise)", policy: "policy")
        XCTAssertEqual(command.executable, "/usr/local/bin/agy")
        XCTAssertEqual(Array(command.arguments.prefix(2)), ["--output-format", "stream-json"])
        XCTAssertEqual(command.arguments.firstIndex(of: "--mode").map { command.arguments[$0 + 1] }, "accept-edits")
        XCTAssertEqual(command.arguments.firstIndex(of: "--conversation").map { command.arguments[$0 + 1] }, "conv_specific")
        XCTAssertFalse(command.arguments.contains("--dangerously-skip-permissions"))
        XCTAssertFalse(command.arguments.contains("--model"))
        XCTAssertNil(command.input)
        let request = try XCTUnwrap(command.arguments.last)
        XCTAssertTrue(request.hasPrefix("--print=Harness instructions:\npolicy"))
        XCTAssertTrue(request.contains("$(echo surprise)"))
        config.antigravityPermission = "auto"; config.antigravityModel = "fixture-model"; config.antigravityMode = "plan"
        let opted = CLI.chat(provider: .antigravity, settings: config, sessionID: nil, prompt: "research", policy: "policy")
        XCTAssertTrue(opted.arguments.contains("--dangerously-skip-permissions"))
        XCTAssertEqual(opted.arguments.firstIndex(of: "--mode").map { opted.arguments[$0 + 1] }, "plan")
        XCTAssertEqual(opted.arguments.firstIndex(of: "--model").map { opted.arguments[$0 + 1] }, "fixture-model")
        XCTAssertFalse(opted.arguments.contains("--conversation"))
        XCTAssertEqual(CLI.resumeArguments(.antigravity, sessionID: "conv_specific"), ["--conversation", "conv_specific"])
        XCTAssertEqual(CLI.auth(.antigravity, settings: config, login: false).arguments, ["models"])
    }
    func testAntigravityExampleUsesConfiguredCLIModelByDefault() throws {
        let example = try JSONDecoder().decode(Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.example.json")))
        XCTAssertEqual(example.antigravityModel, "")
        XCTAssertEqual(example.antigravityMode, "accept-edits")
        XCTAssertEqual(example.antigravityPermission, "review")
    }
    func testAntigravityDecoderTracksSessionToolsAndResult() {
        XCTAssertEqual(EventDecoder.decode(#"{"event":"init","conversation_id":"conv_1","init":{"model":"fixture-model"}}"#, provider: .antigravity), [.session("conv_1")])
        let active = EventDecoder.decode(#"{"event":"step_update","step_update":{"conversation_id":"conv_1","step_index":2,"state":"ACTIVE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"ls"}}}}"#, provider: .antigravity)
        XCTAssertEqual(active, [.session("conv_1"), .message(id: "antigravity-tool-2", role: "tool", text: "run_command\n{\n  \"CommandLine\" : \"ls\"\n}")])
        let done = EventDecoder.decode(#"{"event":"step_update","step_update":{"conversation_id":"conv_1","step_index":2,"state":"DONE","step_type":"tool","tool_name":"run_command","tool_info":{"name":"run_command","parameters":{"CommandLine":"ls"},"output":"README.md"}}}"#, provider: .antigravity)
        XCTAssertEqual(done, [.session("conv_1"), .message(id: "antigravity-tool-2", role: "tool", text: "run_command\n{\n  \"CommandLine\" : \"ls\"\n}\nREADME.md")])
        XCTAssertEqual(EventDecoder.decode(#"{"event":"step_update","step_update":{"conversation_id":"conv_1","step_index":3,"state":"ACTIVE","step_type":"agent_response","text_delta":"partial"}}"#, provider: .antigravity), [.session("conv_1")])
        let success = EventDecoder.decode(#"{"event":"result","result":{"conversation_id":"conv_1","status":"SUCCESS","response":"Done.\nRESEARCH_STATUS: continue","num_turns":1}}"#, provider: .antigravity)
        XCTAssertTrue(success.contains(.completed))
        XCTAssertTrue(success.contains(.message(id: "antigravity-result-conv_1-1", role: "assistant", text: "Done.\nRESEARCH_STATUS: continue")))
        let failure = EventDecoder.decode(#"{"event":"result","result":{"conversation_id":"","status":"ERROR","response":"","error":"invalid model selection","num_turns":0}}"#, provider: .antigravity)
        XCTAssertEqual(failure, [.failure("invalid model selection")]); XCTAssertFalse(failure.contains(.completed))
    }
    func testSettingsSavedBeforeOpenCodeStillLoad() throws {
        let legacy = #"{"workspaceRoot":"/Volumes/x","catalogURL":"https://example.com","codexPath":"","claudePath":"","codexModel":"","claudeModel":"","codexSandbox":"workspace-write","claudePermissionMode":"auto"}"#
        let decoded = try JSONDecoder().decode(Settings.self, from: Data(legacy.utf8))
        XCTAssertEqual(decoded.opencodePath, ""); XCTAssertEqual(decoded.opencodePermission, "reject")
        XCTAssertEqual(decoded.antigravityPath, ""); XCTAssertEqual(decoded.antigravityMode, "accept-edits"); XCTAssertEqual(decoded.antigravityPermission, "review")
        let roundTrip = try JSONDecoder().decode(Settings.self, from: JSONEncoder().encode(decoded))
        XCTAssertEqual(roundTrip.opencodePermission, "reject"); XCTAssertEqual(roundTrip.workspaceRoot, "/Volumes/x")
        XCTAssertEqual(roundTrip.antigravityPermission, "review")
    }
    func testMCPSelectionIsGenericAndScopedToThisRun() throws {
        let names = try CLI.configuredMCPServers(Data(#"[{"name":"unrelated_service","enabled":true},{"name":"docs"}]"#.utf8))
        XCTAssertEqual(names, ["docs", "unrelated_service"])
        let command = CLI.chat(provider: .codex, settings: try settings(), sessionID: nil, prompt: "research", policy: "policy", disabledMCPServers: names)
        XCTAssertTrue(command.arguments.contains("mcp_servers.unrelated_service.enabled=false"))
        XCTAssertTrue(command.arguments.contains("mcp_servers.docs.enabled=false"))
        XCTAssertFalse(command.arguments.contains("--ignore-user-config"))
        let optedIn = CLI.chat(provider: .codex, settings: try settings(), sessionID: nil, prompt: "research", policy: "policy", disabledMCPServers: ["unrelated_service"], enabledMCPServers: ["docs"])
        XCTAssertTrue(optedIn.arguments.contains("mcp_servers.docs.enabled=true"))
        XCTAssertTrue(optedIn.arguments.contains("mcp_servers.unrelated_service.enabled=false"))
    }
    func testDecodersDistinguishFailureAndCompletion() {
        XCTAssertEqual(EventDecoder.decode(#"{"type":"thread.started","thread_id":"abc"}"#, provider: .codex), [.session("abc")])
        XCTAssertEqual(EventDecoder.decode(#"{"type":"turn.failed","error":{"message":"quota"}}"#, provider: .codex), [.failure("quota")])
        XCTAssertEqual(EventDecoder.decode(#"{"type":"turn.completed"}"#, provider: .codex), [.completed])
        let claude = EventDecoder.decode(#"{"type":"result","session_id":"xyz","is_error":true,"errors":["Sign in"]}"#, provider: .claude)
        XCTAssertTrue(claude.contains(.failure("Sign in"))); XCTAssertFalse(claude.contains(.completed))
        let denied = EventDecoder.decode(#"{"type":"result","is_error":false,"permission_denials":[{"tool_name":"Bash"}]}"#, provider: .claude)
        XCTAssertTrue(denied.contains { if case .failure = $0 { return true }; return false })
    }
    func testDecoderKeepsToolOutputSeparateFromAssistant() {
        let events = EventDecoder.decode(#"{"type":"item.completed","item":{"id":"tool1","type":"command_execution","command":"git status","aggregated_output":"clean","status":"completed"}}"#, provider: .codex)
        XCTAssertEqual(events, [.message(id: "tool1", role: "tool", text: "git status\ncompleted\nclean")])
        XCTAssertEqual(EventDecoder.decode("non-JSON startup message", provider: .codex), [])
    }
    func testContinuationRequiresAnExplicitModelDecision() {
        XCTAssertEqual(EventDecoder.continuationDecision("Next experiment is justified.\nRESEARCH_STATUS: continue"), "continue")
        XCTAssertEqual(EventDecoder.continuationDecision("I need a GPU.\nRESEARCH_STATUS: needs_input"), "needs_input")
        XCTAssertNil(EventDecoder.continuationDecision("I could continue, but need your answer."))
        XCTAssertNil(EventDecoder.continuationDecision("RESEARCH_STATUS: something unexpected"))
    }
    func testStorageRejectsICloudFakeMountAndTraversal() throws {
        XCTAssertThrowsError(try Workspace.validate(NSHomeDirectory() + "/Documents/test"))
        XCTAssertThrowsError(try Workspace.validate("/Volumes/DefinitelyNotMounted/research"))
        XCTAssertThrowsError(try Workspace.directory(root.path, "../../../../tmp/escape"))
        XCTAssertEqual(try Workspace.validate(root.path), root)
    }
    func testInstructionLookupPrefersRootAndRejectsSymlinkEscape() throws {
        let directory = try Workspace.directory(root.path, "tmp/test-instructions-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        try FileManager.default.createDirectory(at: directory.appendingPathComponent("docs"), withIntermediateDirectories: true)
        try "root policy".write(to: directory.appendingPathComponent("auto-research.md"), atomically: true, encoding: .utf8)
        try "other".write(to: directory.appendingPathComponent("docs/AUTO-RESEARCH.md"), atomically: true, encoding: .utf8)
        XCTAssertEqual(Workspace.instructionFile(in: directory, trackedFiles: ["docs/AUTO-RESEARCH.md", "auto-research.md"])?.lastPathComponent, "auto-research.md")
        try FileManager.default.removeItem(at: directory.appendingPathComponent("auto-research.md"))
        try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent("auto-research.md"), withDestinationURL: URL(fileURLWithPath: "/etc/hosts"))
        XCTAssertNil(Workspace.instructionFile(in: directory, trackedFiles: ["auto-research.md"]))
    }
    func testRunnerPreservesUTF8StdinAndBothOutputStreams() async throws {
        let result = try await ProcessRunner().run(Command("/bin/zsh", ["-c", "cat; print -u2 'diagnostic'"], input: "λ $(unchanged)\n"), directory: root)
        XCTAssertEqual(result.code, 0); XCTAssertEqual(result.stdout, "λ $(unchanged)\n"); XCTAssertTrue(result.stderr.contains("diagnostic"))
    }
    func testStopCancelsOnlyOwnedProcess() async throws {
        let runner = ProcessRunner()
        let task = Task { try await runner.run(Command("/bin/sleep", ["30"]), directory: root) }
        try await Task.sleep(nanoseconds: 150_000_000)
        runner.stop()
        let result = try await task.value
        XCTAssertNotEqual(result.code, 0)
    }
    func testLiveCodexTurnAndResumeWhenExplicitlyEnabled() async throws {
        guard ProcessInfo.processInfo.environment["RESEARCH_LIVE_CODEX_TEST"] == "1" else { throw XCTSkip("Live provider smoke test is opt-in.") }
        var config = try settings(); config.codexSandbox = "read-only"
        let directory = try Workspace.directory(root.path, "tmp/live-codex-smoke")
        let discovery = try await ProcessRunner().run(CLI.mcpDiscovery(config.codexPath), directory: directory)
        let disabledServers = try CLI.configuredMCPServers(Data(discovery.stdout.utf8))
        let policy = "This is a read-only CLI bridge test. Do not use tools, access files, or perform research. Reply only to the requested text."
        let first = try await ProcessRunner().run(CLI.chat(provider: .codex, settings: config, sessionID: nil, prompt: "Remember the word heliotrope. Reply BRIDGE_READY.", policy: policy, disabledMCPServers: disabledServers), directory: directory)
        try first.stdout.write(to: root.appendingPathComponent("evidence/codex-first.jsonl"), atomically: true, encoding: .utf8)
        try first.stderr.write(to: root.appendingPathComponent("evidence/codex-first.stderr.log"), atomically: true, encoding: .utf8)
        XCTAssertEqual(first.code, 0, first.stderr)
        XCTAssertFalse(first.stderr.contains("MCP server startup failed"), first.stderr)
        let events = first.stdout.split(separator: "\n").flatMap { EventDecoder.decode(String($0), provider: .codex) }
        XCTAssertTrue(events.contains(.completed))
        let id = events.compactMap { event -> String? in if case .session(let id) = event { return id }; return nil }.first
        XCTAssertNotNil(id)
        let second = try await ProcessRunner().run(CLI.chat(provider: .codex, settings: config, sessionID: id, prompt: "What word did I ask you to remember? Reply with that word only.", policy: policy, disabledMCPServers: disabledServers), directory: directory)
        try second.stdout.write(to: root.appendingPathComponent("evidence/codex-resume.jsonl"), atomically: true, encoding: .utf8)
        XCTAssertEqual(second.code, 0, second.stderr)
        XCTAssertFalse(second.stderr.contains("MCP server startup failed"), second.stderr)
        let messages = second.stdout.split(separator: "\n").flatMap { EventDecoder.decode(String($0), provider: .codex) }.compactMap { event -> String? in if case .message(_, "assistant", let text) = event { return text }; return nil }
        XCTAssertTrue(messages.joined().lowercased().contains("heliotrope"), messages.joined())
    }
}
