import XCTest
@testable import ResearchCore

final class CloudGroupResumeTests: XCTestCase {
    func fixture() -> (UUID, [ResearchSession]) {
        let group = UUID()
        let sessions = (1...5).map { index -> ResearchSession in
            var session = ResearchSession(projectID: "fixture", title: "Agent", provider: .codex)
            session.sharedAgent = SharedAgent(groupID: group, index: index, generation: 14, previousSessionID: UUID(), contextFile: nil, sharesContext: true, agentCount: 5)
            session.cloud = CloudSession(sandboxID: UUID().uuidString, stateID: session.id.uuidString)
            session.providerSessionID = UUID().uuidString
            return session
        }
        return (group, sessions)
    }
    func testSavedFiveOverridesDefaultsFourWithoutChangingIdentityOrMethods() throws {
        let (group, sessions) = fixture()
        let defaults = SharedContextOptions(enabled: true, agentCount: 4, intervalMinutes: 5)
        let plan = try CloudGroupResume(sessions: sessions, projectID: "fixture", groupID: group, options: defaults)
        XCTAssertEqual(plan.sessions.map(\.id), sessions.map(\.id))
        XCTAssertEqual(plan.sessions.map { $0.cloud!.sandboxID }, sessions.map { $0.cloud!.sandboxID })
        XCTAssertEqual(plan.sessions.map(\.providerSessionID), sessions.map(\.providerSessionID))
        XCTAssertEqual(plan.generation, 14); XCTAssertEqual(plan.options.agentCount, 5)
        XCTAssertEqual(plan.options.intervalMinutes, 5); XCTAssertNil(plan.options.methods)
        XCTAssertEqual(defaults.agentCount, 4)
    }
    func testIncompleteDuplicateOrMissingConversationFailsClosed() throws {
        let (group, original) = fixture()
        let options = SharedContextOptions(enabled: true, agentCount: 4, intervalMinutes: 5)
        func reject(_ sessions: [ResearchSession]) { XCTAssertThrowsError(try CloudGroupResume(sessions: sessions, projectID: "fixture", groupID: group, options: options)) }
        reject(Array(original.dropLast()))
        var changed = original; changed[1].cloud = original[0].cloud; reject(changed)
        changed = original; changed[0].providerSessionID = nil; reject(changed)
        changed = original; changed[0].sharedAgent?.generation = 13; reject(changed)
        changed = original; changed[0].sharedAgent?.contextFile = "/missing/context"; reject(changed)
    }
    @MainActor func testResumedCycleUsesOriginalIDsThenNextGenerationOnlyAfterDrainAndMerge() async {
        let ids = (0..<5).map { _ in UUID() }; let context = URL(fileURLWithPath: "/saved/round13.md")
        var events: [String] = []; var cycle: SharedContextCycle!
        let hooks = SharedContextCycle.Hooks(prepare: { events.append("preflight") }, launch: { generation, value, previous in
            events.append("launch\(generation)")
            XCTAssertEqual(previous, ids)
            if generation == 14 { XCTAssertEqual(value, context); return ids }
            return (0..<5).map { _ in UUID() }
        }, drain: { _ in events.append("drain") }, capture: { found, generation, value in
            XCTAssertEqual(found, ids); XCTAssertEqual(generation, 14); XCTAssertEqual(value, context)
            events.append("capture"); return URL(fileURLWithPath: "/saved/manifest14.json")
        }, merge: { _ in events.append("merge"); return URL(fileURLWithPath: "/saved/round14.md") }, retire: { _ in events.append("retire") }, cancelOperation: {}, update: { phase, generation, _, _ in
            if phase == .researching { Task { @MainActor in if generation == 14 { cycle.synchronize() } else { cycle.stop() } } }
        })
        cycle = SharedContextCycle(options: .init(enabled: true, agentCount: 5, intervalMinutes: 5), hooks: hooks)
        cycle.start(resuming: .init(generation: 14, sessionIDs: ids, context: context))
        cycle.start() // Duplicate click must not create another task.
        await cycle.waitUntilStopped()
        XCTAssertEqual(events, ["preflight", "launch14", "drain", "capture", "merge", "retire", "launch15", "drain"])
    }
    @MainActor func testFailedResumePreflightNeverLaunchesOrDrainsSavedRemoteWorkers() async {
        var launched = false; var drained: [UUID] = []; var phase: SharedContextCycle.Phase?
        let hooks = SharedContextCycle.Hooks(prepare: { throw ResearchError("Remote worker active") }, launch: { _, _, _ in launched = true; return [] }, drain: { drained += $0 }, capture: { _, _, _ in throw ResearchError("unexpected") }, merge: { $0 }, retire: { _ in }, cancelOperation: {}, update: { p, _, _, _ in phase = p })
        let cycle = SharedContextCycle(options: .init(enabled: true, agentCount: 5, intervalMinutes: 5), hooks: hooks)
        cycle.start(resuming: .init(generation: 14, sessionIDs: (0..<5).map { _ in UUID() }, context: nil))
        await cycle.waitUntilStopped()
        XCTAssertFalse(launched); XCTAssertTrue(drained.isEmpty); XCTAssertEqual(phase, .failed)
    }
}
