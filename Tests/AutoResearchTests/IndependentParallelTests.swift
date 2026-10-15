import AppKit
import SwiftUI
import XCTest
import ResearchCore
@testable import AutoResearch

final class IndependentParallelTests: XCTestCase {
    var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @MainActor func fixture(count: Int = 2) async throws -> (AppStore, ResearchProject, URL) {
        let directory = try Workspace.directory(root.path, "tmp/independent-test-\(UUID().uuidString)")
        let source = try Workspace.directory(directory.path, "source")
        try "Starting evidence".write(to: source.appendingPathComponent("baseline.txt"), atomically: true, encoding: .utf8)
        for args in [["init", "-q"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "baseline"]] {
            let result = try await ProcessRunner().run(Command("/usr/bin/git", args), directory: source)
            XCTAssertEqual(result.code, 0, result.stderr)
        }
        let cli = directory.appendingPathComponent("fixture-cli")
        try """
        #!/usr/bin/python3
        import sys,json,time,uuid
        from pathlib import Path
        if 'mcp' in sys.argv:
            print('[]');sys.exit(0)
        prompt=sys.stdin.read()
        Path('fixture-input.txt').write_text(prompt)
        Path('fixture-started').write_text('started')
        print(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}),flush=True)
        while not Path('fixture-release').exists():time.sleep(0.01)
        if Path('fixture-release').read_text()=='fail':
            print(json.dumps({'type':'turn.failed','error':{'message':'Fixture failure'}}),flush=True);sys.exit(1)
        Path('private-result.txt').write_text(Path.cwd().name)
        print(json.dumps({'type':'item.completed','item':{'type':'agent_message','id':'answer','text':'Independent fixture completed'}}),flush=True)
        print(json.dumps({'type':'turn.completed'}),flush=True)
        """.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cli.path)
        var config = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.json")))
        config.workspaceRoot = try Workspace.directory(directory.path, "runtime").path
        config.codexPath = cli.path
        let store = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: config)
        let project = ResearchProject(repository: "", name: "Same problem", localPath: source.path, isCustom: true)
        store.projects = [project]; store.select(project)
        store.setSharedConfiguration(.init(enabled: false, agentCount: count, intervalMinutes: 0, parallelEnabled: true), for: project.id)
        return (store, project, directory)
    }
    @MainActor func waitUntil(_ condition: @escaping @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(15)
        while !condition() && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertTrue(condition(), "Timed out waiting for the local fixture")
    }
    @MainActor func started(_ store: AppStore, count: Int) -> Bool {
        store.sessions.count == count && store.sessions.allSatisfy {
            $0.workingDirectory.map { FileManager.default.fileExists(atPath: URL(fileURLWithPath: $0).appendingPathComponent("fixture-started").path) } ?? false
        }
    }
    @MainActor func release(_ session: ResearchSession, fail: Bool = false) throws {
        try (fail ? "fail" : "complete").write(to: URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory)).appendingPathComponent("fixture-release"), atomically: true, encoding: .utf8)
    }

    @MainActor func testSameProblemStartsConcurrentPrivateSessionsAndPersistsOneGroup() async throws {
        let (store, project, directory) = try await fixture(count: 3)
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        store.composer = "Solve this same complete problem independently."
        store.send()
        try await waitUntil { self.started(store, count: 3) }
        XCTAssertEqual(store.runners.count, 3)
        XCTAssertNotNil(store.selectedSharedGroupID)
        XCTAssertNil(store.selectedSessionID)
        XCTAssertEqual(Set(store.sessions.compactMap(\.workingDirectory)).count, 3)
        for session in store.sessions {
            XCTAssertTrue(session.sharedAgent?.isIndependent == true)
            XCTAssertNil(session.sharedAgent?.contextFile)
            let folder = URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory))
            let prompt = try String(contentsOf: folder.appendingPathComponent("fixture-input.txt"), encoding: .utf8)
            XCTAssertTrue(prompt.contains("Solve this same complete problem independently."))
            XCTAssertTrue(prompt.contains("there are no coordinated roles"))
            XCTAssertFalse(prompt.contains("Shared context copied"))
            XCTAssertFalse(prompt.contains("Keep a separate, minimal RESEARCH_CHECKPOINT.md"))
            for other in store.sessions where other.id != session.id { XCTAssertFalse(prompt.contains(other.workingDirectory!)) }
            try release(session)
        }
        await store.sharedCycles[project.id]?.waitUntilStopped()
        XCTAssertEqual(store.sharedRuns[project.id]?.phase, .completed)
        XCTAssertFalse(store.hasActiveRuns); XCTAssertTrue(store.activePaths.isEmpty)
        XCTAssertTrue(store.sessions.allSatisfy { $0.state == "Completed" && $0.sharedAgent?.generation == 1 })
        XCTAssertEqual(Set(store.sessions.compactMap(\.providerSessionID)).count, 3)
        XCTAssertFalse(FileManager.default.fileExists(atPath: URL(fileURLWithPath: store.settings.workspaceRoot).appendingPathComponent("shared-context").path))
        XCTAssertEqual(SharedTranscript.latestAgents(in: store.sessions, projectID: project.id, groupID: store.selectedSharedGroupID!).count, 3)
        XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: store.settings)
        XCTAssertEqual(restored.selectedSharedGroupID, store.selectedSharedGroupID)
        XCTAssertTrue(restored.sessions.allSatisfy { $0.sharedAgent?.isIndependent == true })
        XCTAssertTrue(restored.sharedConfiguration(project.id).parallelEnabled)
        XCTAssertFalse(restored.sharedConfiguration(project.id).enabled)
        try await Task.sleep(for: .milliseconds(350))
    }

    @MainActor func testFailedIndependentAgentDoesNotCancelPeer() async throws {
        let (store, project, directory) = try await fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        store.composer = "Same problem"; store.send()
        try await waitUntil { self.started(store, count: 2) }
        let first = store.sessions[0], second = store.sessions[1]
        try release(first, fail: true)
        try await waitUntil { store.sessions.first { $0.id == first.id }?.state == "Needs attention" }
        XCTAssertNotNil(store.runners[second.id]); XCTAssertTrue(store.sharedActive(project.id))
        try release(second)
        await store.sharedCycles[project.id]?.waitUntilStopped()
        XCTAssertEqual(store.sessions.first { $0.id == second.id }?.state, "Completed")
        XCTAssertEqual(store.sharedRuns[project.id]?.phase, .failed)
        XCTAssertFalse(store.hasActiveRuns); XCTAssertEqual(store.sessions.count, 2)
        try await Task.sleep(for: .milliseconds(350))
    }

    @MainActor func testStopDrainsIndependentAgentsWithoutSnapshotsOrReplacements() async throws {
        let (store, project, directory) = try await fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        store.composer = "Same problem"; store.send()
        try await waitUntil { self.started(store, count: 2) }
        store.synchronizeSharedResearch(project.id)
        store.stopSharedResearch(project.id)
        await store.sharedCycles[project.id]?.waitUntilStopped()
        XCTAssertEqual(store.sharedRuns[project.id]?.phase, .stopped)
        XCTAssertFalse(store.hasActiveRuns); XCTAssertTrue(store.activePaths.isEmpty)
        XCTAssertEqual(store.sessions.count, 2)
        XCTAssertTrue(store.sessions.allSatisfy { $0.state == "Stopped" && $0.sharedAgent?.contextFile == nil })
        try await Task.sleep(for: .milliseconds(350))
    }

    func testLegacyOptionsAndIndependentModeSurviveEncoding() throws {
        for enabled in [false, true] {
            let data = Data("{\"enabled\":\(enabled),\"agentCount\":4,\"intervalMinutes\":3}".utf8)
            let old = try JSONDecoder().decode(SharedContextOptions.self, from: data)
            XCTAssertEqual(old.parallelEnabled, enabled)
        }
        let options = SharedContextOptions(enabled: false, agentCount: 4, intervalMinutes: 0, parallelEnabled: true)
        XCTAssertNoThrow(try options.validate())
        XCTAssertEqual(try JSONDecoder().decode(SharedContextOptions.self, from: JSONEncoder().encode(options)), options)
        XCTAssertThrowsError(try SharedContextOptions(enabled: true, agentCount: 4, intervalMinutes: 0).validate())
    }

    @MainActor func testIndependentAndSharingControlsFitNativeChatWidth() async throws {
        _ = NSApplication.shared
        let (store, project, directory) = try await fixture(count: 4)
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        store.composer = "Layout fixture"; store.send()
        try await waitUntil { self.started(store, count: 4) }
        let output = try Workspace.directory(root.path, "evidence/compact-toolbar-20261015")
        for sharing in [false, true] {
            // Render both toolbar modes against the same local fixture; no provider requests.
            store.sharedOptions[project.id] = .init(enabled: sharing, agentCount: 4, intervalMinutes: 3, parallelEnabled: true)
            store.sharedRuns[project.id]?.sharesContext = sharing
            for width in [697.0, 997.0] {
                let host = NSHostingView(rootView: ContentView(store: store).header.background(Color.white))
                let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: width, height: 48), styleMask: [.borderless], backing: .buffered, defer: false)
                window.contentView = host; host.layoutSubtreeIfNeeded()
                XCTAssertLessThanOrEqual(host.fittingSize.width, width)
                XCTAssertEqual(host.fittingSize.height, 48, accuracy: 1)
                let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
                host.cacheDisplay(in: host.bounds, to: bitmap)
                let data = try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                try data.write(to: output.appendingPathComponent("toolbar-\(sharing ? "sharing" : "independent")-\(Int(width)).png"))
            }
        }
        let host = NSHostingView(rootView: ContentView(store: store))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 1280, height: 860), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host; host.layoutSubtreeIfNeeded()
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:])).write(to: output.appendingPathComponent("compact-chat.png"))
        for session in store.sessions { try release(session) }
        await store.sharedCycles[project.id]?.waitUntilStopped()
        try await Task.sleep(for: .milliseconds(350))
    }
}
