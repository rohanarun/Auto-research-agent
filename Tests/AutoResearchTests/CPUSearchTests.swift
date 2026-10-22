import XCTest
import ResearchCore
@testable import AutoResearch

final class CPUSearchTests: XCTestCase {
    @MainActor func testCPUAvailabilityRequiresInstalledFilesAndDescriptionInThisProject() async throws {
        let (store, project, source) = try await fixture()
        let task = try XCTUnwrap(store.researchSubtasks.first { $0.execution == "script" })
        XCTAssertTrue(store.availableSubtasks(project).contains { $0.id == task.id })
        XCTAssertEqual(store.cpuSearchConfiguration(project)?.summary, "Search finite candidate parameters with exact checks.")
        var other = project
        other.localPath = source.appendingPathComponent("unconfigured-project").path
        XCTAssertFalse(store.availableSubtasks(other).contains { $0.id == task.id })
        XCTAssertTrue(store.availableSubtasks(other).isEmpty)
        let verification = try XCTUnwrap(store.researchSubtasks.first { $0.execution != "script" })
        store.playSubtask(verification, project: other)
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertFalse(store.hasActiveRuns)
        let configURL = source.appendingPathComponent(".auto-research/cpu-search.json")
        let original = try Data(contentsOf: configURL)
        for summary in [nil, "", " \n"] as [String?] {
            var config = try JSONDecoder().decode(CPUSearchConfiguration.self, from: original)
            config.summary = summary
            try JSONEncoder().encode(config).write(to: configURL)
            XCTAssertFalse(store.availableSubtasks(project).contains { $0.id == task.id })
        }
        try original.write(to: configURL)
        for relative in ["search.py", ".auto-research/plan.json"] {
            let file = source.appendingPathComponent(relative)
            let data = try Data(contentsOf: file)
            try FileManager.default.removeItem(at: file)
            XCTAssertFalse(store.availableSubtasks(project).contains { $0.id == task.id })
            store.playSubtask(task, project: project)
            XCTAssertTrue(store.sessions.isEmpty)
            XCTAssertFalse(store.hasActiveRuns)
            try data.write(to: file)
        }
        XCTAssertTrue(store.availableSubtasks(project).contains { $0.id == task.id })
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.appendingPathComponent("MODEL_CALLED").path))
    }

    @MainActor func fixture() async throws -> (AppStore, ResearchProject, URL) {
        let (store, project, directory) = try await IndependentParallelTests().fixture(count: 5)
        let source = URL(fileURLWithPath: try XCTUnwrap(project.localPath))
        let setup = source.appendingPathComponent(".auto-research")
        try FileManager.default.createDirectory(at: setup, withIntermediateDirectories: true)
        try #"""
        import argparse,json,time
        from pathlib import Path
        p=argparse.ArgumentParser()
        for name in ('plan','output','overview'):p.add_argument('--'+name,required=True)
        a=p.parse_args()
        Path('script-started').write_text('started')
        while not Path('script-release').exists():time.sleep(.01)
        out=Path(a.output);out.mkdir()
        (out/'cpu-result.json').write_text(json.dumps({'status':'COMPLETED','summary':'Script fixture completed without a model'}))
        print('Exact candidate checks finished',flush=True)
        """#.write(to: source.appendingPathComponent("search.py"), atomically: true, encoding: .utf8)
        try "{}".write(to: setup.appendingPathComponent("plan.json"), atomically: true, encoding: .utf8)
        try #"{"python":"/usr/bin/python3","script":"search.py","plan":".auto-research/plan.json","summary":"Search finite candidate parameters with exact checks."}"#.write(to: setup.appendingPathComponent("cpu-search.json"), atomically: true, encoding: .utf8)
        let trap = directory.appendingPathComponent("model-trap")
        try "#!/bin/sh\ntouch '\(source.appendingPathComponent("MODEL_CALLED").path)'\nexit 99\n".write(to: trap, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: trap.path)
        store.settings.codexPath = trap.path; store.settings.claudePath = trap.path; store.settings.opencodePath = trap.path
        store.daytona.enabled = true; store.provider = .claude
        var options = store.sharedConfiguration(project.id)
        var methods = try store.methodConfiguration(project.id); methods.forkAndFlush = true; options.methods = methods
        store.setSharedConfiguration(options, for: project.id)
        return (store, project, source)
    }

    @MainActor func testCPUPlayRunsScriptWithoutAnyProviderAndBlocksPromptFallback() async throws {
        let (store, project, source) = try await fixture()
        let task = try XCTUnwrap(store.researchSubtasks.first { $0.execution == "script" })
        store.playSubtask(task, project: project)
        try await IndependentParallelTests().waitUntil { FileManager.default.fileExists(atPath: source.appendingPathComponent("script-started").path) }
        let id = try XCTUnwrap(store.session?.id)
        XCTAssertTrue(store.session?.scripted == true)
        XCTAssertNil(store.session?.providerSessionID); XCTAssertNil(store.session?.cloud)
        store.playSubtask(task, project: project)
        XCTAssertEqual(store.sessions.count, 1)
        try "done".write(to: source.appendingPathComponent("script-release"), atomically: true, encoding: .utf8)
        try await IndependentParallelTests().waitUntil { !store.hasActiveRuns }
        XCTAssertEqual(store.session?.state, "Completed")
        XCTAssertTrue(store.session!.messages.contains { $0.text.contains("Script fixture completed") })
        store.composer = "Run a model"; store.send()
        XCTAssertEqual(store.sessions.count, 1); XCTAssertFalse(store.hasActiveRuns)
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.appendingPathComponent("MODEL_CALLED").path))
        XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory: IndependentParallelTests().root.appendingPathComponent("Resources"), defaults: store.settings)
        XCTAssertEqual(restored.session?.id, id); XCTAssertTrue(restored.session?.scripted == true)
    }

    @MainActor func testStopPreservesEvidenceAndMissingSetupNeverCallsModel() async throws {
        let (store, project, source) = try await fixture()
        let task = try XCTUnwrap(store.researchSubtasks.first { $0.execution == "script" })
        store.playSubtask(task, project: project)
        try await IndependentParallelTests().waitUntil { FileManager.default.fileExists(atPath: source.appendingPathComponent("script-started").path) }
        store.stop(try XCTUnwrap(store.session?.id))
        try await IndependentParallelTests().waitUntil { !store.hasActiveRuns }
        XCTAssertEqual(store.session?.state, "Stopped")
        XCTAssertTrue(FileManager.default.fileExists(atPath: source.appendingPathComponent("script-started").path))
        try FileManager.default.removeItem(at: source.appendingPathComponent(".auto-research/cpu-search.json"))
        store.playSubtask(task, project: project)
        try await IndependentParallelTests().waitUntil { !store.hasActiveRuns }
        XCTAssertEqual(store.session?.state, "Stopped")
        XCTAssertEqual(store.sessions.count, 1)
        XCTAssertTrue(store.error?.contains("not configured") == true)
        XCTAssertFalse(FileManager.default.fileExists(atPath: source.appendingPathComponent("MODEL_CALLED").path))
    }
}
