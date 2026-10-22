import XCTest
import ResearchCore
@testable import AutoResearch

final class ResearchSubtaskTests: XCTestCase {
    @MainActor func testFocusedRunStaysLocalWithoutMethodsAndRetainsHistory() async throws {
        let fixture = IndependentParallelTests()
        let (store, project, _) = try await fixture.fixture(count: 5)
        let source = URL(fileURLWithPath: try XCTUnwrap(project.localPath))
        let setup = source.appendingPathComponent(".auto-research")
        try FileManager.default.createDirectory(at: setup, withIntermediateDirectories: true)
        try "# Prepared search fixture".write(to: source.appendingPathComponent("search.py"), atomically: true, encoding: .utf8)
        try "{}".write(to: setup.appendingPathComponent("plan.json"), atomically: true, encoding: .utf8)
        try #"{"python":"/usr/bin/python3","script":"search.py","plan":".auto-research/plan.json","summary":"Search finite candidate parameters with exact checks."}"#.write(to: setup.appendingPathComponent("cpu-search.json"), atomically: true, encoding: .utf8)
        var options = store.sharedConfiguration(project.id)
        var methods = try store.methodConfiguration(project.id)
        methods.forkAndFlush = true; options.methods = methods
        store.setSharedConfiguration(options, for: project.id)
        store.daytona.enabled = true
        let task = try XCTUnwrap(store.researchSubtasks.first { $0.execution != "script" })
        store.playSubtask(task, project: project)
        try await fixture.waitUntil { fixture.started(store, count: 1) }
        let session = try XCTUnwrap(store.session)
        XCTAssertNil(session.cloud); XCTAssertNil(session.sharedAgent)
        XCTAssertEqual(session.researchSubtaskID, task.id)
        let input = try String(contentsOf: URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory)).appendingPathComponent("fixture-input.txt"), encoding: .utf8)
        XCTAssertTrue(input.contains(task.request))
        XCTAssertTrue(input.contains("SKILL.md"))
        XCTAssertTrue(input.contains("overview.json"))
        XCTAssertTrue(input.contains("objective, method, meaning"))
        XCTAssertFalse(input.contains("Optional research methods are enabled"))
        store.playSubtask(task, project: project)
        XCTAssertEqual(store.sessions.count, 1, "Repeated Play cannot create a competing writer")
        XCTAssertNotNil(store.error)
        try fixture.release(session)
        try await fixture.waitUntil { !store.hasActiveRuns }
        let providerSessionID = try XCTUnwrap(store.session?.providerSessionID)
        XCTAssertEqual(store.subtaskSession(task, project: project)?.state, "Completed")
        store.closeSession(session.id)
        XCTAssertEqual(store.subtaskSession(task, project: project)?.id, session.id)
        XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory: fixture.root.appendingPathComponent("Resources"), defaults: store.settings)
        XCTAssertEqual(restored.subtaskSession(task, project: project)?.id, session.id)
        restored.selectSession(try XCTUnwrap(restored.subtaskSession(task, project: project)))
        XCTAssertNil(restored.session?.closedAt)
        XCTAssertEqual(restored.session?.providerSessionID, providerSessionID)
        try FileManager.default.removeItem(at: setup.appendingPathComponent("cpu-search.json"))
        restored.launch(session.id, project: project, request: "Verify again")
        XCTAssertFalse(restored.hasActiveRuns)
        XCTAssertEqual(restored.session?.state, "Needs attention")
        XCTAssertEqual(restored.session?.providerSessionID, providerSessionID)
    }

    func testExistingHistoryWithoutSubtaskFieldStillDecodes() throws {
        let old = ResearchSession(projectID: "test/research", title: "Existing research", provider: .codex)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as? [String: Any])
        json.removeValue(forKey: "researchSubtaskID")
        let decoded = try JSONDecoder().decode(ResearchSession.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertEqual(decoded.id, old.id)
        XCTAssertNil(decoded.researchSubtaskID)
    }

    func testOverviewLoadsModelDescriptionAndRejectsPartialWrites() throws {
        let root = IndependentParallelTests().root
        let directory = try Workspace.directory(root.path, "tmp/overview-test-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = LeanSearch.overviewURL(directory: directory, sessionID: UUID())
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        XCTAssertNil(LeanSearch.readOverview(at: url))
        try #"{"objective":"Search rational parameters","method":"Exact arithmetic","meaning":"Conditional certificate"}"#.write(to: url, atomically: true, encoding: .utf8)
        XCTAssertEqual(LeanSearch.readOverview(at: url)?.objective, "Search rational parameters")
        try #"{"objective":"unfinished""#.write(to: url, atomically: true, encoding: .utf8)
        XCTAssertNil(LeanSearch.readOverview(at: url))
    }
}
