import XCTest
import ResearchCore
@testable import AutoResearch

final class ChatClosingTests: XCTestCase {
    var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    @MainActor func fixture() throws -> (AppStore, URL, ResearchCore.Settings) {
        let directory = try Workspace.directory(root.path, "tmp/chat-close-test-\(UUID().uuidString)")
        var config = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.json")))
        config.workspaceRoot = directory.path
        let store = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: config)
        store.projects = [ResearchProject(repository: "test/research")]; store.selectedProjectID = "test/research"
        return (store, directory, config)
    }
    @MainActor func testClosingPersistsAndReopeningRetainsProviderHistoryAndFiles() async throws {
        let (store, directory, config) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkout = try Workspace.directory(directory.path, "checkout")
        try "Keep this experiment".write(to: checkout.appendingPathComponent("result.txt"), atomically: true, encoding: .utf8)
        var session = ResearchSession(projectID: "test/research", title: "Old chat", provider: .codex)
        session.providerSessionID = "provider-session"; session.workingDirectory = checkout.path
        session.messages = [ChatMessage(role: "assistant", text: "Saved finding")]; session.state = "Completed"
        store.sessions = [session]; store.selectSession(session)
        store.closeSession(session.id)
        XCTAssertTrue(store.projectSessions.isEmpty); XCTAssertEqual(store.closedProjectSessions.count, 1)
        XCTAssertNil(store.selectedSessionID); XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: config)
        XCTAssertEqual(restored.closedProjectSessions.count, 1)
        restored.selectSession(restored.closedProjectSessions[0])
        XCTAssertTrue(restored.closedProjectSessions.isEmpty); XCTAssertEqual(restored.session?.providerSessionID, "provider-session")
        XCTAssertEqual(restored.session?.messages.map(\.text), ["Saved finding"])
        XCTAssertEqual(try String(contentsOf: checkout.appendingPathComponent("result.txt"), encoding: .utf8), "Keep this experiment")
        try await Task.sleep(for: .milliseconds(350))
    }
    @MainActor func testActiveChatCannotCloseAndGroupClosePreservesCombinedHistory() async throws {
        let (store, directory, _) = try fixture()
        defer { try? FileManager.default.removeItem(at: directory) }
        let group = UUID()
        var a = ResearchSession(projectID: "test/research", title: "Agent one", provider: .codex)
        a.sharedAgent = SharedAgent(groupID: group, index: 1, generation: 1, previousSessionID: nil, contextFile: nil)
        a.messages = [ChatMessage(role: "assistant", text: "One")]
        var b = ResearchSession(projectID: "test/research", title: "Agent two", provider: .claude)
        b.sharedAgent = SharedAgent(groupID: group, index: 2, generation: 1, previousSessionID: nil, contextFile: nil)
        b.messages = [ChatMessage(role: "assistant", text: "Two")]
        store.sessions = [a, b]; store.runners[a.id] = ProcessRunner()
        store.closeSession(a.id); store.closeSharedGroup(group)
        XCTAssertEqual(store.projectSessions.count, 2); XCTAssertTrue(store.closedProjectSessions.isEmpty)
        store.runners.removeValue(forKey: a.id)
        store.selectSharedGroup(group, projectID: "test/research"); store.closeSharedGroup(group)
        XCTAssertNil(store.selectedSharedGroupID); XCTAssertTrue(store.projectSharedGroups.isEmpty)
        XCTAssertEqual(store.closedProjectSessions.count, 2)
        XCTAssertEqual(SharedTranscript.entries(in: store.sessions, projectID: "test/research", groupID: group).count, 2)
        store.selectSession(a); XCTAssertEqual(store.projectSessions.count, 1)
        try await Task.sleep(for: .milliseconds(350))
    }
}
