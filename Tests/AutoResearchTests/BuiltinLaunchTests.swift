import AppKit
import SwiftUI
import XCTest
import ResearchCore
@testable import AutoResearch

/// Any accidental cloud path fails locally; these fixtures never contact a
/// provider, the official judge, or a GPU service.
private final class BuiltinNoNetworkProtocol: URLProtocol, @unchecked Sendable {
    private static let lock = NSLock()
    private static var count = 0
    static func reset() { lock.lock(); defer { lock.unlock() }; count = 0 }
    static var requests: Int { lock.lock(); defer { lock.unlock() }; return count }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        Self.lock.lock(); Self.count += 1; Self.lock.unlock()
        client?.urlProtocol(self, didFailWithError: ResearchError("Network use is forbidden in the built-in launch fixture."))
    }
    override func stopLoading() {}
}

final class BuiltinLaunchTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var resources: URL { root.appendingPathComponent("Resources") }

    @MainActor private func fixture() throws -> (AppStore, URL, ResearchCore.Settings) {
        let directory = try Workspace.directory(root.path, "tmp/builtin-launch-" + UUID().uuidString)
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: resources.appendingPathComponent("Defaults.example.json")))
        settings.workspaceRoot = try Workspace.directory(directory.path, "runtime").path
        let cli = directory.appendingPathComponent("fixture-cli")
        try #"""
        #!/usr/bin/python3
        import json,sys,uuid
        from pathlib import Path
        if 'mcp' in sys.argv:
            print('[]');sys.exit(0)
        Path('fixture-input.txt').write_text(sys.stdin.read())
        print(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}))
        print(json.dumps({'type':'item.completed','item':{'type':'agent_message','id':'answer','text':'Local fixture completed.\nRESEARCH_STATUS: finished'}}))
        print(json.dumps({'type':'turn.completed'}))
        """#.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cli.path)
        settings.codexPath = cli.path
        let store = AppStore(resourceDirectory: resources, defaults: settings)
        store.provider = .codex
        store.builtinAccountVerifier = { "fixture-account" }
        return (store, directory, settings)
    }

    @MainActor private func configurePlay(_ store: AppStore, project: ResearchProject) {
        XCTAssertTrue(store.setBuiltinDraft(KernelResearchDraft(sourcePath: "my-original/kernel.py", checkBudget: "0",
            researchPrompt: "Inspect my selected kernel without spending quota.", expectedAccount: "fixture-account"), for: project))
    }

    @MainActor private func enableOtherExecutionModes(_ store: AppStore, project: ResearchProject) throws {
        store.daytona.enabled = true
        BuiltinNoNetworkProtocol.reset()
        let configuration = URLSessionConfiguration.ephemeral
        configuration.protocolClasses = [BuiltinNoNetworkProtocol.self]
        store.daytonaClient = try DaytonaClient(apiKey: "fixture-only", installID: store.daytona.installID,
            apiURL: URL(string: "https://app.daytona.io/api")!, session: URLSession(configuration: configuration))
        var options = SharedContextOptions(enabled: true, agentCount: 3, intervalMinutes: 1, parallelEnabled: true)
        var methods = try store.methodConfiguration(project.id)
        methods.rsiTop = true; methods.forkAndFlush = true
        options.methods = methods
        store.setSharedConfiguration(options, for: project.id)
    }

    @MainActor private func finish(_ store: AppStore) async throws -> ResearchSession {
        for task in Array(store.builtinPlayTasks.values) { await task.value }
        let session = try XCTUnwrap(store.sessions.last)
        let task = try XCTUnwrap(store.workerTasks[session.id])
        await task.value
        let completed = try XCTUnwrap(store.sessions.first { $0.id == session.id })
        XCTAssertEqual(completed.state, "Completed", completed.messages.map(\.text).joined(separator: "\n"))
        XCTAssertFalse(store.hasActiveRuns)
        return completed
    }

    @MainActor func testBuiltinsAppearForExistingSavedCatalogWithoutReplacingHistory() throws {
        let (initial, directory, settings) = try fixture()
        defer { initial.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let existing = ResearchProject(repository: "example/existing-project")
        var session = ResearchSession(projectID: existing.id, title: "Preserved research", provider: .codex)
        session.state = "Completed"
        session.providerSessionID = "saved-provider-session"
        session.messages = [ChatMessage(role: "assistant", text: "Keep the original evidence.")]
        let saved = SavedState(projects: [existing], sessions: [session], selectedProjectID: existing.id,
            selectedSessionID: session.id, provider: .codex)
        try Workspace.write(saved, to: URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("state.json"))
        let restored = AppStore(resourceDirectory: resources, defaults: settings)
        defer { restored.shutdown() }
        let definitions = try BuiltinProjects.load(resources: resources)
        XCTAssertEqual(definitions.count, 5)
        XCTAssertEqual(Set(restored.projects.compactMap(\.builtinID)), Set(definitions.map(\.id)))
        XCTAssertEqual(restored.projects.filter { $0.id == existing.id }.count, 1)
        XCTAssertEqual(restored.selectedProjectID, existing.id)
        XCTAssertEqual(restored.selectedSessionID, session.id)
        XCTAssertEqual(restored.sessions.first?.providerSessionID, "saved-provider-session")
        XCTAssertEqual(restored.sessions.first?.messages.first?.text, "Keep the original evidence.")
        XCTAssertTrue(restored.persist())
        let again = AppStore(resourceDirectory: resources, defaults: settings)
        defer { again.shutdown() }
        XCTAssertEqual(again.projects.count, definitions.count + 1)
        XCTAssertEqual(Set(again.projects.map(\.id)).count, again.projects.count)
    }

    @MainActor private func verifyOfflineCatalog(cachedHTML: String?) throws {
        let (initial, directory, settings) = try fixture()
        defer { initial.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let offlineResources = try Workspace.directory(directory.path, "offline-resources")
        for file in ["CatalogPolicy.json", "SharedContextDefaults.json", "DaytonaDefaults.json", "BuiltinProjects.json", "StarterPrompts"] {
            try FileManager.default.copyItem(at: resources.appendingPathComponent(file), to: offlineResources.appendingPathComponent(file))
        }
        if let cachedHTML {
            try cachedHTML.write(to: offlineResources.appendingPathComponent("catalog-source.html"), atomically: true, encoding: .utf8)
        }
        let offline = AppStore(resourceDirectory: offlineResources, defaults: settings)
        defer { offline.shutdown() }
        let expected = try BuiltinProjects.load(resources: resources)
        XCTAssertEqual(Set(offline.projects.compactMap(\.builtinID)), Set(expected.map(\.id)))
        XCTAssertNotNil(offline.selectedProjectID)
        XCTAssertTrue(offline.sessions.isEmpty)
    }

    @MainActor func testBuiltinsAvailableWithoutCachedRemoteCatalog() throws {
        try verifyOfflineCatalog(cachedHTML: nil)
    }

    @MainActor func testBuiltinsAvailableWhenCachedRemoteCatalogIsMalformed() throws {
        try verifyOfflineCatalog(cachedHTML: "<html><p>Unavailable catalog fixture</p></html>")
    }

    @MainActor func testPlayUsesOneLocalAgentDespiteCloudParallelAndMethodPreferences() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let definition = try XCTUnwrap(store.builtinProjects.first)
        let project = definition.project
        try enableOtherExecutionModes(store, project: project)
        configurePlay(store, project: project)
        store.play(project)
        let session = try await finish(store)
        XCTAssertEqual(store.sessions.count, 1)
        XCTAssertNil(session.cloud); XCTAssertNil(session.sharedAgent)
        XCTAssertTrue(store.sharedRuns.isEmpty)
        XCTAssertEqual(BuiltinNoNetworkProtocol.requests, 0)
        XCTAssertTrue(store.daytona.enabled, "Project routing must not erase the user's other preferences.")
        XCTAssertTrue(store.sharedConfiguration(project.id).parallelEnabled)
        XCTAssertTrue(store.sharedConfiguration(project.id).usesMethods)
        let checkout = URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory))
        XCTAssertEqual(checkout.path, URL(fileURLWithPath: store.settings.workspaceRoot).appendingPathComponent(project.managedWorkspaceComponent).path)
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkout.appendingPathComponent(".git").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: checkout.appendingPathComponent(".auto-research-methods").path))
        let prompt = try String(contentsOf: checkout.appendingPathComponent("fixture-input.txt"), encoding: .utf8)
        XCTAssertTrue(prompt.contains(definition.id))
        XCTAssertTrue(prompt.contains(try definition.instructions(resources: resources)))
        XCTAssertTrue(prompt.contains("my-original/kernel.py"))
        XCTAssertTrue(prompt.contains("fixture-account"))
        XCTAssertFalse(store.showingProjectDetails)
        XCTAssertEqual(store.selectedSessionID, session.id)
        XCTAssertFalse(prompt.contains("Optional research methods are enabled"))
        XCTAssertFalse(prompt.contains("there are no coordinated roles"))
    }

    @MainActor func testComposerIncludesProjectGuideAndManualRequestOnLocalPath() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let definition = try XCTUnwrap(store.builtinProjects.last)
        let project = definition.project
        try enableOtherExecutionModes(store, project: project)
        store.select(project)
        let request = "Explain this public starter. Do not spend any official quota."
        store.composer = request
        store.send()
        let session = try await finish(store)
        XCTAssertEqual(store.sessions.count, 1)
        XCTAssertEqual(store.composer, "")
        XCTAssertNil(session.cloud); XCTAssertNil(session.sharedAgent)
        XCTAssertEqual(BuiltinNoNetworkProtocol.requests, 0)
        let checkout = URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory))
        let prompt = try String(contentsOf: checkout.appendingPathComponent("fixture-input.txt"), encoding: .utf8)
        XCTAssertTrue(prompt.contains(request))
        XCTAssertTrue(prompt.contains(definition.id))
        XCTAssertTrue(prompt.contains(try definition.instructions(resources: resources)))
    }

    @MainActor func testRelaunchPreservesWorkspaceResultsAndEditedStarter() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = try XCTUnwrap(store.builtinProjects.first).project
        configurePlay(store, project: project)
        let checkout = try store.prepareBuiltinWorkspace(project)
        let editedGuide = "Public starter review only. Keep our saved negative result; do not rerun it."
        let evidence = Data("Unchanged evidence".utf8)
        try editedGuide.write(to: checkout.appendingPathComponent("auto-research.md"), atomically: true, encoding: .utf8)
        try evidence.write(to: checkout.appendingPathComponent("negative-result.txt"))
        store.play(project)
        let first = try await finish(store)
        store.play(project)
        let second = try await finish(store)
        XCTAssertNotEqual(first.id, second.id)
        XCTAssertEqual(first.workingDirectory, checkout.path)
        XCTAssertEqual(second.workingDirectory, checkout.path)
        XCTAssertEqual(try store.prepareBuiltinWorkspace(project), checkout)
        XCTAssertEqual(try String(contentsOf: checkout.appendingPathComponent("auto-research.md"), encoding: .utf8), editedGuide)
        XCTAssertEqual(try Data(contentsOf: checkout.appendingPathComponent("negative-result.txt")), evidence)
        let prompt = try String(contentsOf: checkout.appendingPathComponent("fixture-input.txt"), encoding: .utf8)
        XCTAssertTrue(prompt.contains(editedGuide))
    }

    @MainActor func testUnavailableBuiltinFailsBeforeCloudOrProviderLaunch() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = ResearchProject(repository: "", name: "Unavailable starter", builtinID: "missing-fixture")
        store.projects.append(project)
        try enableOtherExecutionModes(store, project: project)
        store.play(project)
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.error?.contains("resources are unavailable") == true)
        XCTAssertTrue(store.showingProjectDetails)
        XCTAssertTrue(store.sharedRuns.isEmpty)
        XCTAssertEqual(BuiltinNoNetworkProtocol.requests, 0)
        XCTAssertFalse(FileManager.default.fileExists(atPath: URL(fileURLWithPath: store.settings.workspaceRoot).appendingPathComponent(project.managedWorkspaceComponent).path))
    }

    @MainActor func testMissingPlaySetupShowsFormWithoutAccountOrAgentLaunch() throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = try XCTUnwrap(store.builtinProjects.first).project
        var probes = 0
        store.builtinAccountVerifier = { probes += 1; return "fixture-account" }
        store.play(project)
        XCTAssertEqual(probes, 0)
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertTrue(store.builtinPlayTasks.isEmpty)
        XCTAssertTrue(store.showingProjectDetails)
        XCTAssertEqual(store.selectedProjectID, project.id)
        XCTAssertTrue(store.error?.contains("Complete this project's setup") == true)
    }

    @MainActor func testPlayRejectsUnexpectedAccountWithoutLaunchingAgent() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = try XCTUnwrap(store.builtinProjects.first).project
        configurePlay(store, project: project)
        store.builtinAccountVerifier = { "someone-else" }
        store.play(project)
        for task in Array(store.builtinPlayTasks.values) { await task.value }
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertTrue(store.error?.contains("does not match") == true)
        XCTAssertTrue(store.showingProjectDetails)
        XCTAssertFalse(store.hasActiveRuns)
    }

    @MainActor func testRepeatedPlayDoesNotDuplicateAccountPreparationOrUseAnEditedDraft() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = try XCTUnwrap(store.builtinProjects.first).project
        configurePlay(store, project: project)
        var probes = 0
        var release = false
        store.builtinAccountVerifier = {
            probes += 1
            while !release { try await Task.sleep(for: .milliseconds(1)) }
            return "fixture-account"
        }
        store.play(project)
        store.play(project)
        let pending = try XCTUnwrap(store.builtinPlayTasks[project.id])
        for _ in 0..<100 where probes == 0 { try await Task.sleep(for: .milliseconds(1)) }
        XCTAssertEqual(probes, 1)
        var changed = store.builtinDraft(for: project)
        changed.sourcePath = "a-different-source.py"
        XCTAssertTrue(store.setBuiltinDraft(changed, for: project))
        release = true
        await pending.value
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertTrue(store.error?.contains("setup changed") == true)
        XCTAssertFalse(store.hasActiveRuns)
    }

    @MainActor func testShutdownCancelsPendingAccountPreparationBeforeAgentLaunch() async throws {
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = try XCTUnwrap(store.builtinProjects.first).project
        configurePlay(store, project: project)
        var started = false
        var finished = false
        store.builtinAccountVerifier = {
            started = true
            defer { finished = true }
            try await Task.sleep(for: .seconds(30))
            return "fixture-account"
        }
        store.play(project)
        let pending = try XCTUnwrap(store.builtinPlayTasks[project.id])
        for _ in 0..<100 where !started { try await Task.sleep(for: .milliseconds(1)) }
        XCTAssertTrue(started)
        store.shutdown()
        await pending.value
        XCTAssertTrue(finished)
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertTrue(store.builtinPlayPending.isEmpty)
        XCTAssertFalse(store.hasActiveRuns)
    }

    @MainActor func testDraftsPersistPerProjectWithoutStartingResearch() throws {
        let (store, directory, settings) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let first = try XCTUnwrap(store.builtinProjects.first).project
        let second = try XCTUnwrap(store.builtinProjects.last).project
        let draft = KernelResearchDraft(sourcePath: "/Volumes/My SSD/original.py", checkBudget: "3",
            allowScoredSubmissions: true, scoreBudget: "1", researchPrompt: "Review this exact source.",
            scoreCriterion: "Only a verified improvement.", expectedAccount: "my-own-account")
        XCTAssertTrue(store.setBuiltinDraft(draft, for: first))
        XCTAssertTrue(store.setBuiltinDraft(KernelResearchDraft(checkBudget: "0", researchPrompt: "Analyze only.", expectedAccount: "another-account"), for: second))
        let restored = AppStore(resourceDirectory: resources, defaults: settings)
        defer { restored.shutdown() }
        XCTAssertEqual(restored.builtinDraft(for: first), draft)
        XCTAssertEqual(restored.builtinDraft(for: second).checkBudget, "0")
        XCTAssertEqual(restored.builtinDraft(for: second).expectedAccount, "another-account")
        XCTAssertTrue(restored.sessions.isEmpty)
        XCTAssertTrue(restored.runners.isEmpty)
        XCTAssertTrue(restored.builtinPlayTasks.isEmpty)
        let file = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("kernel-setups.json")
        let permissions = try FileManager.default.attributesOfItem(atPath: file.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o600)
    }

    @MainActor func testUnreadableDraftsArePreservedAndBlockPlay() throws {
        let (store, directory, settings) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let file = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("kernel-setups.json")
        let malformed = Data("preserve malformed setup".utf8)
        try malformed.write(to: file)
        let restored = AppStore(resourceDirectory: resources, defaults: settings)
        defer { restored.shutdown() }
        let project = try XCTUnwrap(restored.builtinProjects.first).project
        XCTAssertFalse(restored.setBuiltinDraft(KernelResearchDraft(), for: project))
        restored.play(project)
        XCTAssertTrue(restored.sessions.isEmpty)
        XCTAssertEqual(try Data(contentsOf: file), malformed)
        XCTAssertTrue(restored.error?.contains("unreadable") == true)
    }

    @MainActor func testBuiltinDetailRendersAtCompactAndWideWidths() async throws {
        _ = NSApplication.shared
        let (store, directory, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let project = try XCTUnwrap(store.builtinProjects.last).project
        let output = try Workspace.directory(root.path, "evidence/builtin-projects")
        for width in [620.0, 920.0] {
            let host = NSHostingView(rootView: BuiltinProjectView(store: store, project: project)
                .frame(width: width, height: 960).background(.white).preferredColorScheme(.light))
            let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: 960), styleMask: [.borderless], backing: .buffered, defer: false)
            window.contentView = host
            host.layoutSubtreeIfNeeded()
            try await Task.sleep(for: .milliseconds(50))
            host.layoutSubtreeIfNeeded()
            XCTAssertLessThanOrEqual(host.fittingSize.width, width)
            let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                .write(to: output.appendingPathComponent("prefill-\(Int(width)).png"))
        }
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertFalse(store.hasActiveRuns)
    }
}
