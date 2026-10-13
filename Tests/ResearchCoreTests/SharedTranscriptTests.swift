import XCTest
@testable import ResearchCore

final class SharedTranscriptTests: XCTestCase {
    func session(_ group: UUID, agent: Int, round: Int, project: String = "owner/repo", provider: Provider = .codex) -> ResearchSession {
        var session = ResearchSession(projectID: project, title: "Agent \(agent)", provider: provider)
        session.sharedAgent = SharedAgent(groupID: group, index: agent, generation: round, previousSessionID: nil, contextFile: nil)
        return session
    }
    func message(_ text: String, role: String = "assistant", time: Double) -> ChatMessage {
        var message = ChatMessage(role: role, text: text); message.date = Date(timeIntervalSince1970: time); return message
    }
    func testFeedInterleavesAgentsAndRoundsWithoutLeakingOtherGroupsOrProjects() {
        let group = UUID()
        var first = session(group, agent: 1, round: 1)
        first.messages = [message("first", time: 1), message("command output", role: "tool", time: 3)]
        var second = session(group, agent: 2, round: 1, provider: .claude)
        second.messages = [message("second", time: 2), message("failure", role: "error", time: 4)]
        var next = session(group, agent: 1, round: 2); next.messages = [message("new round", time: 5)]
        var unrelated = session(UUID(), agent: 1, round: 1); unrelated.messages = [message("other group", time: 0)]
        var otherProject = session(group, agent: 3, round: 1, project: "other/repo"); otherProject.messages = [message("other project", time: 0)]
        let entries = SharedTranscript.entries(in: [next, unrelated, second, otherProject, first], projectID: "owner/repo", groupID: group)
        XCTAssertEqual(entries.map { $0.message.text }, ["first", "second", "command output", "failure", "new round"])
        XCTAssertEqual(entries.map { $0.agent.index }, [1, 2, 1, 2, 1])
        XCTAssertEqual(entries[1].provider, .claude); XCTAssertEqual(entries.last?.agent.generation, 2)
    }
    func testStreamingUpdatesKeepStableIdentityAndTiesKeepAgentOrder() {
        let group = UUID(); var first = session(group, agent: 1, round: 1); var second = session(group, agent: 2, round: 1)
        let sharedID = UUID()
        var a = message("partial", time: 1); a.id = sharedID
        var b = message("second", time: 1); b.id = sharedID
        first.messages = [a]; second.messages = [b]
        let before = SharedTranscript.entries(in: [second, first], projectID: "owner/repo", groupID: group)
        first.messages[0].text = "complete streamed response"
        let after = SharedTranscript.entries(in: [first, second], projectID: "owner/repo", groupID: group)
        XCTAssertEqual(before.map(\.id), after.map(\.id)); XCTAssertEqual(Set(after.map(\.id)).count, 2)
        XCTAssertEqual(after[0].message.text, "complete streamed response")
        XCTAssertEqual(after.map { $0.agent.index }, [1, 2])
    }
    func testLatestAgentStatusAndSharedViewSurviveSavedHistory() throws {
        let group = UUID(); var first = session(group, agent: 1, round: 1); first.state = "Replaced"
        var next = session(group, agent: 1, round: 2); next.state = "Running"
        var second = session(group, agent: 2, round: 2); second.state = "Completed"
        var state = SavedState(projects: [], sessions: [second, next, first], selectedProjectID: "owner/repo", selectedSessionID: nil, provider: .codex)
        state.selectedSharedGroupID = group
        let restored = try JSONDecoder().decode(SavedState.self, from: JSONEncoder().encode(state))
        XCTAssertEqual(restored.selectedSharedGroupID, group)
        let latest = SharedTranscript.latestAgents(in: restored.sessions, projectID: "owner/repo", groupID: group)
        XCTAssertEqual(latest.map(\.id), [next.id, second.id]); XCTAssertEqual(latest.map(\.state), ["Running", "Completed"])
    }
}
