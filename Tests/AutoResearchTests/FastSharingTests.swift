import XCTest
import ResearchCore
@testable import AutoResearch

final class FastSharingTests: XCTestCase {
    var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    @MainActor func testLiveWrittenCheckpointsReachFreshSuccessorsWithoutSynthesis() async throws {
        try await exercise(invalidCheckpoint: false)
    }
    @MainActor func testUnsafeCheckpointPreservesChatsAndCannotLaunchSuccessors() async throws {
        try await exercise(invalidCheckpoint: true)
    }
    @MainActor private func exercise(invalidCheckpoint: Bool) async throws {
        let folder = try Workspace.directory(root.path, "tmp/fast-sharing-\(UUID().uuidString)")
        let source = try Workspace.directory(folder.path, "source")
        try "original".write(to: source.appendingPathComponent("baseline.txt"), atomically: true, encoding: .utf8)
        for args in [["init", "-q"], ["add", "."], ["-c", "user.name=Test", "-c", "user.email=test@example.invalid", "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-qm", "fixture"]] {
            let result = try await ProcessRunner().run(Command("/usr/bin/git", args), directory: source)
            XCTAssertEqual(result.code, 0, result.stderr)
        }
        let cli = folder.appendingPathComponent("fixture-cli")
        try """
        #!/usr/bin/python3
        import json,sys,time,uuid
        from pathlib import Path
        if 'mcp' in sys.argv:print('[]');sys.exit(0)
        prompt=sys.stdin.read();cwd=Path.cwd()
        assert 'Read this context manifest and every transcript' not in prompt
        print(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}),flush=True)
        successor='Shared context copied into this fresh session' in prompt
        (cwd/('successor-input.txt' if successor else 'initial-input.txt')).write_text(prompt)
        marker=('SUCCESSOR_' if successor else 'COMPACT_')+cwd.name
        checkpoint=cwd/'RESEARCH_CHECKPOINT.md';tmp=checkpoint.with_suffix('.tmp');tmp.write_text(marker);tmp.replace(checkpoint)
        print(json.dumps({'type':'item.completed','item':{'type':'agent_message','id':'finding','text':'TRANSCRIPT_ONLY_'+cwd.name}}),flush=True)
        while True:time.sleep(.02)
        """.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cli.path)
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.json")))
        settings.workspaceRoot = try Workspace.directory(folder.path, "runtime").path;settings.codexPath = cli.path
        let store = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: settings)
        defer { store.shutdown();try? FileManager.default.removeItem(at: folder) }
        let project = ResearchProject(repository: "", name: "Fast sharing fixture", localPath: source.path, isCustom: true)
        store.projects = [project];store.select(project)
        store.setSharedConfiguration(.init(enabled: true, agentCount: 2, intervalMinutes: 10), for: project.id)
        store.composer = "Verify fast sharing using this local fixture.";store.send()
        try await wait { store.sessions.count == 2 && store.sessions.allSatisfy { s in s.workingDirectory.map { FileManager.default.fileExists(atPath: URL(fileURLWithPath: $0).appendingPathComponent("RESEARCH_CHECKPOINT.md").path) } ?? false } }
        let first = store.sessions
        if invalidCheckpoint {
            let checkpoint = URL(fileURLWithPath: first[1].workingDirectory!).appendingPathComponent("RESEARCH_CHECKPOINT.md")
            try FileManager.default.removeItem(at: checkpoint)
            try FileManager.default.createSymbolicLink(at: checkpoint, withDestinationURL: source.appendingPathComponent("baseline.txt"))
        }
        store.synchronizeSharedResearch(project.id)
        if invalidCheckpoint {
            await store.sharedCycles[project.id]?.waitUntilStopped()
            XCTAssertEqual(store.sharedRuns[project.id]?.phase, .failed)
            XCTAssertEqual(store.sessions.count, 2);XCTAssertFalse(store.hasActiveRuns)
            XCTAssertTrue(store.sessions.allSatisfy { $0.sharedAgent?.contextFile == nil })
        } else {
            try await wait { store.sessions.count == 4 && store.sessions.suffix(2).allSatisfy { s in s.workingDirectory.map { FileManager.default.fileExists(atPath: URL(fileURLWithPath: $0).appendingPathComponent("successor-input.txt").path) } ?? false } }
            let successors = Array(store.sessions.suffix(2))
            XCTAssertTrue(Set(first.compactMap(\.providerSessionID)).isDisjoint(with: successors.compactMap(\.providerSessionID)))
            XCTAssertEqual(Set(store.sessions.compactMap(\.providerSessionID)).count, 4)
            for s in successors {
                XCTAssertEqual(s.sharedAgent?.generation, 2)
                let prompt = try String(contentsOf: URL(fileURLWithPath: s.workingDirectory!).appendingPathComponent("successor-input.txt"), encoding: .utf8)
                for number in [1,2] { XCTAssertTrue(prompt.contains("COMPACT_agent-\(number)")) }
                XCTAssertFalse(prompt.contains("TRANSCRIPT_ONLY_agent-1"))
                let round = URL(fileURLWithPath: s.sharedAgent!.contextFile!).deletingLastPathComponent()
                XCTAssertFalse(FileManager.default.fileExists(atPath: round.appendingPathComponent("synthesis-events.json").path))
                XCTAssertEqual(try String(contentsOf: round.appendingPathComponent("agent-1-checkpoint.md"), encoding: .utf8), "COMPACT_agent-1")
            }
            store.stopSharedResearch(project.id);await store.sharedCycles[project.id]?.waitUntilStopped()
            XCTAssertFalse(store.hasActiveRuns)
        }
        XCTAssertEqual(try String(contentsOf: source.appendingPathComponent("baseline.txt"), encoding: .utf8), "original")
        try await Task.sleep(for: .milliseconds(350))
    }
    @MainActor private func wait(_ condition: @escaping @MainActor () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(15)
        while !condition() && Date() < deadline { try await Task.sleep(for: .milliseconds(20)) }
        XCTAssertTrue(condition(), "Local fast-sharing fixture did not reach expected state")
    }
}
