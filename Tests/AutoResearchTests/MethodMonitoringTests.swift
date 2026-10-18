import XCTest
import ResearchCore
@testable import AutoResearch

final class MethodMonitoringTests: XCTestCase {
    @MainActor func testReplayUpdatesPersistedWorkerMessagesWithoutEndingCoordinator() async throws {
        let fixture = IndependentParallelTests()
        let (store, project, directory) = try await fixture.fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        var session = ResearchSession(projectID: project.id, title: "Coordinator", provider: .codex)
        session.state = "Running"; store.sessions = [session]
        func line(_ text: String) throws -> String {
            String(decoding: try JSONSerialization.data(withJSONObject: ["type":"research_method_event", "branch":"round-1-explore-epoch-1-branch-2", "provider":"codex", "key":"record", "event":["type":"item.completed", "item":["id":"answer", "type":"agent_message", "text":text]]]), as: UTF8.self)
        }
        store.consumeMethodLine(try line("first"), sessionID:session.id, operation:"owned")
        XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory:fixture.root.appendingPathComponent("Resources"),defaults:store.settings)
        restored.consumeMethodLine(try line("updated"),sessionID:session.id,operation:"owned")
        let messages = restored.sessions[0].messages.filter { $0.sourceKey != nil }
        XCTAssertEqual(messages.count,1); XCTAssertEqual(messages[0].text,"updated")
        XCTAssertTrue(messages[0].speaker?.contains("Worker 2") == true)
        let done = String(decoding: try JSONSerialization.data(withJSONObject:["type":"research_method_event","branch":"round-1-explore-epoch-1-branch-2","provider":"codex","key":"done","event":["type":"turn.completed"]]),as:UTF8.self)
        store.consumeMethodLine(done,sessionID:session.id,operation:"owned")
        XCTAssertEqual(store.sessions[0].state,"Running"); XCTAssertFalse(store.completed.contains(session.id))
    }
}
