import XCTest
import ResearchCore
@testable import AutoResearch

final class FleetResponseTests: XCTestCase {
    @MainActor private func fixture() throws -> (AppStore, URL, ResearchCore.Settings, UUID) {
        let (store, root, settings) = try ChatClosingTests().fixture()
        let group = UUID()
        for index in 1...2 {
            var session = ResearchSession(projectID: "test/research", title: "Agent \(index)", provider: .codex)
            session.sharedAgent = SharedAgent(groupID: group, index: index, generation: 1, previousSessionID: nil, contextFile: nil, sharesContext: false)
            session.workingDirectory = try Workspace.directory(root.path, "agent-\(index)").path
            session.state = "Completed"
            store.sessions.append(session)
        }
        let cli = root.appendingPathComponent("fixture-cli")
        try #"""
        #!/usr/bin/python3
        import json,sys,uuid
        from pathlib import Path
        if 'mcp' in sys.argv:
            print('[]');sys.exit(0)
        with Path('requests.jsonl').open('a') as output:
            output.write(json.dumps(sys.stdin.read())+'\n')
        print(json.dumps({'type':'thread.started','thread_id':'fixture'}))
        print(json.dumps({'type':'item.completed','item':{'type':'agent_message','id':str(uuid.uuid4()),'text':'Fixture answer.\nRESEARCH_STATUS: finished'}}))
        print(json.dumps({'type':'turn.completed'}))
        """#.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cli.path)
        store.settings.codexPath = cli.path
        store.selectSharedGroup(group, projectID: "test/research")
        return (store, root, settings, group)
    }
    @MainActor private func assistant(_ store: AppStore, session: UUID, key: String, text: String = "Finished finding", complete: Bool = true) throws {
        let object: [String: Any] = ["type": "item.completed", "item": ["id": key, "type": "agent_message", "text": text]]
        store.consume(String(decoding: try JSONSerialization.data(withJSONObject: object), as: UTF8.self), stderr: false, provider: .codex, sessionID: session)
        if complete { store.consume(#"{"type":"turn.completed"}"#, stderr: false, provider: .codex, sessionID: session) }
    }

    @MainActor func testOnlyCompletedRepliesFollowLatestAndDuplicateDrainDoesNotNavigate() throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let a = store.sessions[0].id, b = store.sessions[1].id
        try assistant(store, session: a, key: "one", complete: false)
        store.consume(#"{"type":"item.completed","item":{"id":"tool","type":"command_execution","command":"fixture"}}"#, stderr: false, provider: .codex, sessionID: a)
        XCTAssertTrue(store.responseEntries.isEmpty)
        store.consume(#"{"type":"turn.completed"}"#, stderr: false, provider: .codex, sessionID: a)
        XCTAssertEqual(store.responseRecipient?.id, a)
        try assistant(store, session: b, key: "two")
        XCTAssertEqual(store.responseRecipient?.id, b)
        XCTAssertEqual(store.responseScrollTarget, store.responseEntries.last?.id)
        let focus = store.responseFocusRequest
        try assistant(store, session: a, key: "one")
        XCTAssertEqual(store.responseEntries.count, 2)
        XCTAssertEqual(store.responseRecipient?.id, b)
        XCTAssertEqual(store.responseFocusRequest, focus)
        var other = store.sessions[0]; other.id = UUID(); other.sharedAgent?.groupID = UUID()
        store.sessions.append(other)
        try assistant(store, session: other.id, key: "other")
        XCTAssertEqual(store.responseEntries.count, 2)
        XCTAssertEqual(store.responseRecipient?.id, b)
    }

    @MainActor func testDraftLocksRecipientAndExplicitNavigationPreservesBothDrafts() throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let a = store.sessions[0].id, b = store.sessions[1].id
        try assistant(store, session: a, key: "one")
        store.composer = "Draft for A"
        try assistant(store, session: b, key: "two")
        XCTAssertEqual(store.responseRecipient?.id, a)
        XCTAssertEqual(store.composer, "Draft for A")
        XCTAssertEqual(store.pendingResponseCount, 1)
        store.nextPendingResponse()
        XCTAssertEqual(store.responseRecipient?.id, b)
        XCTAssertEqual(store.composer, "")
        store.composer = "Draft for B"
        store.previousResponse()
        XCTAssertEqual(store.composer, "Draft for A")
        store.nextResponse()
        XCTAssertEqual(store.composer, "Draft for B")
        store.composer = ""; store.followResponses = false
        try assistant(store, session: a, key: "three")
        XCTAssertEqual(store.responseRecipient?.id, b)
        XCTAssertEqual(store.pendingResponseCount, 1)
    }

    @MainActor func testPersistenceRestoresDraftAndReadPositionWithoutReplayingOrLaunching() throws {
        let (store, root, settings, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        try assistant(store, session: store.sessions[0].id, key: "one")
        store.composer = "Preserved draft"
        store.followResponses = false
        try assistant(store, session: store.sessions[1].id, key: "two")
        XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory: store.resourceDirectory, defaults: settings)
        defer { restored.shutdown() }
        XCTAssertEqual(restored.composer, "Preserved draft")
        XCTAssertEqual(restored.selectedResponse?.id, store.selectedResponse?.id)
        XCTAssertEqual(restored.pendingResponseCount, 1)
        XCTAssertFalse(restored.followResponses)
        XCTAssertNil(restored.responseFocusRequest)
        XCTAssertTrue(restored.runners.isEmpty)
        XCTAssertTrue(restored.authorizedFleetFollowups.isEmpty)
    }

    @MainActor func testIdleWorkerReceivesFollowupWhilePeerRunsAndAdvancesToWaitingReply() async throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let a = store.sessions[0].id, b = store.sessions[1].id
        try assistant(store, session: a, key: "one")
        store.composer = "Please inspect A"
        try assistant(store, session: b, key: "two")
        let peer = ProcessRunner(); store.runners[b] = peer
        store.sendResponseAndAdvance()
        XCTAssertNotNil(store.runners[a]); XCTAssertTrue(store.runners[b] === peer)
        XCTAssertEqual(store.responseRecipient?.id, b)
        XCTAssertEqual(store.sessions.count, 2)
        await store.waitForFleetTurns([a])
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .delivered)
        let data = try String(contentsOf: URL(fileURLWithPath: store.sessions[0].workingDirectory!).appendingPathComponent("requests.jsonl"))
        XCTAssertEqual(data.split(separator: "\n").count, 1)
        XCTAssertTrue(data.contains("Please inspect A"))
        XCTAssertTrue(store.runners[b] === peer)
        store.runners.removeValue(forKey: b)
    }

    @MainActor func testBusyWorkerQueuesAndDeliversExactlyOnceAtTerminalBoundary() async throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let a = store.sessions[0].id
        let original = ProcessRunner(); store.runners[a] = original
        store.activePaths[a] = store.sessions[0].workingDirectory
        store.composer = "Queued for A"
        store.sendResponseAndAdvance()
        XCTAssertTrue(store.runners[a] === original)
        XCTAssertNil(store.workerTasks[a])
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .pending)
        store.runners.removeValue(forKey: a); store.activePaths.removeValue(forKey: a)
        store.completed.insert(a); store.state(a, "Completed")
        XCTAssertTrue(store.finishFleetTurn(a))
        await store.waitForFleetTurns([a])
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .delivered)
        XCTAssertFalse(store.finishFleetTurn(a))
        let data = try String(contentsOf: URL(fileURLWithPath: store.sessions[0].workingDirectory!).appendingPathComponent("requests.jsonl"))
        XCTAssertEqual(data.split(separator: "\n").count, 1)
    }

    @MainActor func testQueuedRequestTransfersToLogicalSuccessorAndIsNotClaimedTwice() throws {
        let (store, root, _, group) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let old = store.sessions[0]
        store.runners[old.id] = ProcessRunner()
        store.composer = "Next generation should inspect this"
        store.sendResponseAndAdvance()
        store.composer = "An unsent draft for successor"
        var next = ResearchSession(projectID: old.projectID, title: "Successor", provider: .codex)
        next.sharedAgent = SharedAgent(groupID: group, index: 1, generation: 2, previousSessionID: old.id, contextFile: nil)
        store.inheritFleetDraft(next); store.sessions.append(next)
        XCTAssertEqual(store.responseNavigation.drafts[next.id], "An unsent draft for successor")
        let first = try store.claimFleetFollowups(next.id, prompt: "Base")
        XCTAssertTrue(first.contains("Next generation should inspect this"))
        XCTAssertEqual(try store.claimFleetFollowups(next.id, prompt: "Base"), "Base")
        XCTAssertEqual(store.responseNavigation.followups.first?.sessionID, next.id)
        store.runners.removeAll()
    }

    @MainActor func testStoppedAndRestoredQueuesStaySavedWithoutAutomaticDispatch() throws {
        let (store, root, settings, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let a = store.sessions[0].id
        store.runners[a] = ProcessRunner(); store.composer = "Keep queued"
        store.sendResponseAndAdvance(); store.stopWorker(a); store.runners.removeAll()
        store.state(a, "Stopped")
        XCTAssertFalse(store.finishFleetTurn(a))
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .pending)
        XCTAssertTrue(store.authorizedFleetFollowups.isEmpty)
        XCTAssertTrue(store.persist())
        let restored = AppStore(resourceDirectory: store.resourceDirectory, defaults: settings)
        defer { restored.shutdown() }
        XCTAssertTrue(restored.runners.isEmpty)
        XCTAssertEqual(restored.pendingFollowupCount, 1)
        XCTAssertTrue(restored.canResumeQueuedFollowups)
        XCTAssertFalse(restored.dispatchFleetFollowup(to: a))
    }

    @MainActor func testFailedDeliveryIsRetainedForReviewAndDoesNotRetry() throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let a = store.sessions[0].id
        store.runners[a] = ProcessRunner(); store.composer = "Do not retry automatically"
        store.sendResponseAndAdvance()
        _ = try store.claimFleetFollowups(a, prompt: "Base")
        store.runners.removeAll(); store.failed.insert(a); store.state(a, "Needs attention")
        XCTAssertFalse(store.finishFleetTurn(a))
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .needsReview)
        XCTAssertFalse(store.canResumeQueuedFollowups)
        XCTAssertTrue(store.runners.isEmpty)
    }

    @MainActor func testAnotherCheckoutWriterDefersFollowupWithoutLaunching() throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let occupied = UUID()
        store.activePaths[occupied] = store.sessions[0].workingDirectory
        store.composer = "Wait for exclusive checkout"
        store.sendResponseAndAdvance()
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .pending)
    }

    @MainActor func testLegacyCompletedHistoryIsReadableButNeverPendingOrAutoSelected() throws {
        let (store, root, settings, group) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        store.sessions[0].messages = [ChatMessage(role: "assistant", text: "Historical completed reply")]
        var saved = SavedState(projects: store.projects, sessions: store.sessions, selectedProjectID: "test/research", selectedSessionID: nil, provider: .codex)
        saved.selectedSharedGroupID = group
        try Workspace.write(saved, to: root.appendingPathComponent("state.json"))
        let restored = AppStore(resourceDirectory: store.resourceDirectory, defaults: settings)
        defer { restored.shutdown() }
        XCTAssertEqual(restored.responseEntries.count, 1)
        XCTAssertEqual(restored.pendingResponseCount, 0)
        XCTAssertNil(restored.selectedResponse)
        XCTAssertNil(restored.responseFocusRequest)
        XCTAssertTrue(restored.runners.isEmpty)
    }

    @MainActor func testTerminalPermissionFailureDoesNotPublishAnAssistantReply() throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let id = store.sessions[0].id
        store.consume(#"{"type":"assistant","message":{"id":"reply","content":[{"type":"text","text":"Partial answer"}]}}"#, stderr: false, provider: .claude, sessionID: id)
        store.consume(#"{"type":"result","is_error":false,"permission_denials":[{"tool_name":"fixture"}]}"#, stderr: false, provider: .claude, sessionID: id)
        XCTAssertTrue(store.responseEntries.isEmpty)
        XCTAssertTrue(store.failed.contains(id))
    }

    @MainActor func testSynchronousLaunchRefusalKeepsRequestManuallyResumable() throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        store.gatewayConfigurationReadable = false
        store.composer = "Preserve this through connection repair"
        store.sendResponseAndAdvance()
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .pending)
        XCTAssertTrue(store.authorizedFleetFollowups.isEmpty)
        XCTAssertTrue(store.canResumeQueuedFollowups)
    }

    @MainActor func testReleasingAnotherWriterDispatchesWaitingRequestExactlyOnce() async throws {
        let (store, root, _, _) = try fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: root) }
        let target = store.sessions[0]
        var other = ResearchSession(projectID: target.projectID, title: "Other checkout writer", provider: .codex)
        other.workingDirectory = target.workingDirectory; other.state = "Completed"
        store.sessions.append(other)
        store.activePaths[other.id] = other.workingDirectory
        store.composer = "Deliver after the existing writer finishes"
        store.sendResponseAndAdvance()
        XCTAssertTrue(store.runners.isEmpty)
        store.activePaths.removeValue(forKey: other.id); store.completed.insert(other.id)
        XCTAssertFalse(store.finishFleetTurn(other.id))
        XCTAssertNotNil(store.runners[target.id])
        await store.waitForFleetTurns([target.id])
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .delivered)
        let data = try String(contentsOf: URL(fileURLWithPath: target.workingDirectory!).appendingPathComponent("requests.jsonl"))
        XCTAssertEqual(data.split(separator: "\n").count, 1)
    }
}
