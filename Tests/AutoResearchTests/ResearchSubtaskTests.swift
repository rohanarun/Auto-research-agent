import XCTest
import ResearchCore
@testable import AutoResearch

final class ResearchSubtaskTests: XCTestCase {
    @MainActor func testFocusedRunStaysLocalWithoutMethodsAndRetainsHistory() async throws {
        let fixture = IndependentParallelTests()
        let (store, project, _) = try await fixture.fixture(count: 5)
        var options = store.sharedConfiguration(project.id)
        var methods = try store.methodConfiguration(project.id)
        methods.forkAndFlush = true; options.methods = methods
        store.setSharedConfiguration(options, for: project.id)
        store.daytona.enabled = true
        let task = try XCTUnwrap(store.researchSubtasks.first)
        store.playSubtask(task, project: project)
        try await fixture.waitUntil { fixture.started(store, count: 1) }
        let session = try XCTUnwrap(store.session)
        XCTAssertNil(session.cloud); XCTAssertNil(session.sharedAgent)
        XCTAssertEqual(session.researchSubtaskID, task.id)
        let input = try String(contentsOf: URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory)).appendingPathComponent("fixture-input.txt"), encoding: .utf8)
        XCTAssertTrue(input.contains(task.request))
        XCTAssertTrue(input.contains("SKILL.md"))
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
    }

    func testExistingHistoryWithoutSubtaskFieldStillDecodes() throws {
        let old = ResearchSession(projectID: "test/research", title: "Existing research", provider: .codex)
        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(old)) as? [String: Any])
        json.removeValue(forKey: "researchSubtaskID")
        let decoded = try JSONDecoder().decode(ResearchSession.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertEqual(decoded.id, old.id)
        XCTAssertNil(decoded.researchSubtaskID)
    }
}
