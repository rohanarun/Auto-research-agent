import XCTest
@testable import ResearchCore

final class SharedContextTests: XCTestCase {
    var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testLiveCodexSharedContextWhenExplicitlyEnabled() async throws {
        guard ProcessInfo.processInfo.environment["RESEARCH_LIVE_SHARED_TEST"] == "1" else { throw XCTSkip("Live shared-context verification is opt-in.") }
        var settings = try JSONDecoder().decode(Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.json")))
        settings.codexSandbox = "read-only"
        let config = settings
        let directory = try Workspace.directory(root.path, "tmp/live-shared-\(UUID().uuidString)")
        let discovery = try await ProcessRunner().run(CLI.mcpDiscovery(settings.codexPath), directory: directory)
        let disabled = try CLI.configuredMCPServers(Data(discovery.stdout.utf8))
        @Sendable func run(_ name: String, prompt: String, policy: String) async throws -> (String, String) {
            let folder = try Workspace.directory(directory.path, name)
            let result = try await ProcessRunner().run(CLI.chat(provider: .codex, settings: config, sessionID: nil, prompt: prompt, policy: policy, disabledMCPServers: disabled), directory: folder)
            try Workspace.write(result.stdout, to: folder.appendingPathComponent("events.json"))
            let events = result.stdout.split(separator: "\n").flatMap { EventDecoder.decode(String($0), provider: .codex) }
            guard result.code == 0, events.contains(.completed), !events.contains(where: { if case .failure = $0 { return true }; return false }) else { throw ResearchError("Live CLI verification failed: \(result.stderr)") }
            let id = events.compactMap { if case .session(let id) = $0 { return id }; return nil }.first!
            let text = events.compactMap { if case .message(_, "assistant", let text) = $0 { return text }; return nil }.joined(separator: "\n")
            return (id, text)
        }
        let simple = "This is a harness verification fixture. Do not run experiments, edit files, or use external services. Follow only the test request."
        async let a = run("worker-1", prompt: "Report this test finding verbatim: Agent one observed amber-cedar-17.", policy: simple)
        async let b = run("worker-2", prompt: "Report this test finding verbatim: Agent two observed violet-birch-29.", policy: simple)
        let initial = try await [a, b]
        var agents: [[String: String]] = []
        for (index, result) in initial.enumerated() {
            let transcript = directory.appendingPathComponent("agent-\(index + 1).json")
            try Workspace.write([ChatMessage(role: "assistant", text: result.1)], to: transcript)
            agents.append(["number": String(index + 1), "transcript": transcript.path, "checkout": directory.appendingPathComponent("worker-\(index + 1)").path])
        }
        let manifest = directory.appendingPathComponent("manifest.json")
        try Workspace.write(agents, to: manifest)
        let policy = try String(contentsOf: root.appendingPathComponent("Resources/SharedContextPolicy.md"), encoding: .utf8)
        let combined = try await run("synthesis", prompt: "Combine every transcript referenced by \(manifest.path). This is a test fixture; preserve both exact observation markers and assign each agent to verify the other's marker.", policy: policy)
        XCTAssertTrue(EventDecoder.sharedContextReady(combined.1), combined.1)
        XCTAssertTrue(combined.1.contains("amber-cedar-17")); XCTAssertTrue(combined.1.contains("violet-birch-29"))
        async let c = run("successor-1", prompt: "Shared handoff from both previous agents:\n\(combined.1)\nReply with both exact observation markers only.", policy: simple)
        async let d = run("successor-2", prompt: "Shared handoff from both previous agents:\n\(combined.1)\nReply with both exact observation markers only.", policy: simple)
        let successors = try await [c, d]
        for result in successors { XCTAssertTrue(result.1.contains("amber-cedar-17")); XCTAssertTrue(result.1.contains("violet-birch-29")) }
        XCTAssertEqual(Set((initial + [combined] + successors).map { $0.0 }).count, 5)
        print("Verified live Codex shared handoff with two parallel workers, synthesis, and two fresh successors. Evidence: \(directory.path)")
    }

    @MainActor func testReplacementWaitsForDrainAndDurableMergeAndUsesNewIDs() async throws {
        var order: [String] = []; var launched: [[UUID]] = []; var cycle: SharedContextCycle!
        let context = URL(fileURLWithPath: "/saved/context.md")
        let hooks = SharedContextCycle.Hooks(prepare: { order.append("prepare") }, launch: { generation, shared, previous in
            if generation == 2 { XCTAssertEqual(shared, context); XCTAssertEqual(previous, launched[0]) }
            let ids = [UUID(), UUID()]; launched.append(ids); order.append("launch\(generation)"); return ids
        }, drain: { _ in order.append("drain"); await Task.yield() }, capture: { _, _, _ in order.append("capture"); return URL(fileURLWithPath: "/saved/manifest.json") }, merge: { _ in order.append("merge"); await Task.yield(); order.append("merged"); return context }, retire: { _ in order.append("retire") }, cancelOperation: {}, update: { phase, generation, _, _ in
            if phase == .researching { Task { @MainActor in if generation == 1 { cycle.synchronize() } else { cycle.stop() } } }
        })
        cycle = SharedContextCycle(options: .init(enabled: true, agentCount: 2, intervalMinutes: 10), hooks: hooks)
        cycle.start(); await cycle.waitUntilStopped()
        XCTAssertEqual(order, ["prepare", "launch1", "drain", "capture", "merge", "merged", "retire", "launch2", "drain"])
        XCTAssertTrue(Set(launched[0]).isDisjoint(with: launched[1])); XCTAssertFalse(cycle.active)
    }

    @MainActor func testFailedOrCancelledSynthesisCannotLaunchReplacements() async {
        for cancel in [false, true] {
            var launches = 0; var retires = 0; var last: SharedContextCycle.Phase?; var cycle: SharedContextCycle!
            let hooks = SharedContextCycle.Hooks(prepare: {}, launch: { _, _, _ in launches += 1; return [UUID(), UUID()] }, drain: { _ in }, capture: { _, _, _ in URL(fileURLWithPath: "/saved/snapshot") }, merge: { _ in
                if cancel { cycle.stop(); return URL(fileURLWithPath: "/late/result") }
                throw ResearchError("Provider quota reached")
            }, retire: { _ in retires += 1 }, cancelOperation: {}, update: { phase, _, _, _ in
                last = phase
                if phase == .researching { Task { @MainActor in cycle.synchronize() } }
            })
            cycle = SharedContextCycle(options: .init(enabled: true, agentCount: 2, intervalMinutes: 10), hooks: hooks)
            cycle.start(); await cycle.waitUntilStopped()
            XCTAssertEqual(launches, 1); XCTAssertEqual(retires, 0); XCTAssertEqual(last, cancel ? .stopped : .failed)
        }
    }

    @MainActor func testStopDuringPreparationCannotStartAgents() async {
        var launches = 0; var cycle: SharedContextCycle!
        let hooks = SharedContextCycle.Hooks(prepare: { cycle.stop() }, launch: { _, _, _ in launches += 1; return [] }, drain: { _ in }, capture: { _, _, _ in URL(fileURLWithPath: "/snapshot") }, merge: { $0 }, retire: { _ in }, cancelOperation: {}, update: { _, _, _, _ in })
        cycle = SharedContextCycle(options: .init(enabled: true, agentCount: 2, intervalMinutes: 1), hooks: hooks)
        cycle.start(); await cycle.waitUntilStopped(); XCTAssertEqual(launches, 0)
    }

    func testLegacyHistoryAndInvalidSettings() throws {
        let session = ResearchSession(projectID: "a/b", title: "Old chat", provider: .codex)
        let state = SavedState(projects: [], sessions: [session], selectedProjectID: nil, selectedSessionID: nil, provider: .codex)
        let data = try JSONEncoder().encode(state)
        let decoded = try JSONDecoder().decode(SavedState.self, from: data)
        XCTAssertNil(decoded.sharedOptions); XCTAssertNil(decoded.sessions[0].sharedAgent)
        XCTAssertThrowsError(try SharedContextOptions(enabled: true, agentCount: 1, intervalMinutes: 10).validate())
        XCTAssertThrowsError(try SharedContextOptions(enabled: true, agentCount: 2, intervalMinutes: .nan).validate())
        XCTAssertThrowsError(try SharedContextOptions(enabled: true, agentCount: 2, intervalMinutes: 0).validate())
        XCTAssertFalse(EventDecoder.sharedContextReady("I could not read agent two."))
        XCTAssertFalse(EventDecoder.sharedContextReady("SHARED_CONTEXT_STATUS: needs_input"))
        XCTAssertTrue(EventDecoder.sharedContextReady("Combined findings.\nSHARED_CONTEXT_STATUS: ready\n"))
    }

    func testIndependentCheckoutsPreserveDirtyFilesAndDoNotWriteSource() async throws {
        let folder = try Workspace.directory(root.path, "tmp/shared-workspace-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: folder) }
        let source = try Workspace.directory(folder.path, "source")
        let runner = ProcessRunner()
        func git(_ args: [String], at path: URL) async throws {
            let result = try await runner.run(Command("/usr/bin/git", args), directory: path)
            XCTAssertEqual(result.code, 0, result.stderr)
        }
        try await git(["init"], at: source)
        try "baseline".write(to: source.appendingPathComponent("tracked.txt"), atomically: true, encoding: .utf8)
        try await git(["add", "."], at: source)
        try await git(["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-m", "fixture"], at: source)
        try "dirty work".write(to: source.appendingPathComponent("tracked.txt"), atomically: true, encoding: .utf8)
        try "untracked work".write(to: source.appendingPathComponent("notes.txt"), atomically: true, encoding: .utf8)
        let first = folder.appendingPathComponent("agent-1"), second = folder.appendingPathComponent("agent-2")
        for destination in [first, second] {
            try await SharedWorkspace.prepare(source: source, destination: destination, runner: runner, environment: [:])
            XCTAssertEqual(try String(contentsOf: destination.appendingPathComponent("tracked.txt"), encoding: .utf8), "dirty work")
            XCTAssertEqual(try String(contentsOf: destination.appendingPathComponent("notes.txt"), encoding: .utf8), "untracked work")
            XCTAssertTrue(FileManager.default.fileExists(atPath: destination.appendingPathComponent(".git/HEAD").path))
        }
        try "agent one changed this".write(to: first.appendingPathComponent("tracked.txt"), atomically: true, encoding: .utf8)
        XCTAssertEqual(try String(contentsOf: second.appendingPathComponent("tracked.txt"), encoding: .utf8), "dirty work")
        XCTAssertEqual(try String(contentsOf: source.appendingPathComponent("tracked.txt"), encoding: .utf8), "dirty work")
    }
}
