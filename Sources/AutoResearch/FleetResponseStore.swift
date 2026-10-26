import Foundation
import ResearchCore

extension AppStore {
    func restoreFleetHistory() {
        // Legacy transcripts lack per-turn terminal markers. Only the final
        // assistant reply of an explicitly completed session is safe to list.
        // Historical entries are read; loading never selects or launches them.
        for session in sessions where session.sharedAgent != nil && session.state == "Completed" {
            guard let message = session.messages.last(where: { $0.role == "assistant" && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else { continue }
            let reply = FleetReply(session: session, message: message)
            if responseNavigation.receive(reply) { responseNavigation.unreadReplyIDs.remove(reply.id) }
        }
    }
    var responseGroupID: UUID? { selectedSharedGroupID ?? session?.sharedAgent?.groupID }
    var responseAgents: [ResearchSession] {
        guard let projectID = selectedProjectID, let group = responseGroupID else { return [] }
        return SharedTranscript.latestAgents(in: sessions, projectID: projectID, groupID: group)
    }
    var responseRecipient: ResearchSession? {
        guard let group = responseGroupID else { return nil }
        let index = responseNavigation.recipientIndices[group] ?? session?.sharedAgent?.index
        return responseAgents.first { $0.sharedAgent?.index == index } ?? responseAgents.first
    }
    var responseEntries: [FleetReply] {
        responseNavigation.replies.filter { $0.projectID == selectedProjectID && $0.groupID == responseGroupID }
    }
    var selectedResponse: FleetReply? {
        guard let group = responseGroupID, let id = responseNavigation.selectedReplyIDs[group] else { return nil }
        return responseEntries.first { $0.id == id }
    }
    var responsePosition: Int? { selectedResponse.flatMap { reply in responseEntries.firstIndex { $0.id == reply.id } } }
    var pendingResponseCount: Int { responseEntries.filter { responseNavigation.unreadReplyIDs.contains($0.id) && $0.id != selectedResponse?.id }.count }
    var followResponses: Bool {
        get { responseNavigation.followResponses }
        set { responseNavigation.followResponses = newValue; save() }
    }
    var responseSendingBlockedReason: String? {
        guard let recipient = responseRecipient else { return "Choose an agent to reply to." }
        if recipient.cloud.map({ cloudCleanup.contains($0.sandboxID) }) == true { return "Wait for this agent's cloud recovery to finish." }
        return nil
    }
    var canSendResponse: Bool { responseSendingBlockedReason == nil && !composer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
    var pendingFollowupCount: Int {
        guard let recipient = responseRecipient else { return 0 }
        return responseNavigation.followups.filter { $0.matches(recipient) && [.pending, .needsReview].contains($0.delivery) }.count
    }
    var canResumeQueuedFollowups: Bool {
        guard responseSendingBlockedReason == nil, let recipient = responseRecipient else { return false }
        return responseNavigation.followups.contains { $0.matches(recipient) && $0.delivery == .pending && !authorizedFleetFollowups.contains($0.id) }
    }
    var responseDeliveryStatus: String? {
        guard let recipient = responseRecipient else { return nil }
        let requests = responseNavigation.followups.filter { $0.matches(recipient) }
        let review = requests.filter { $0.delivery == .needsReview }.count
        if review > 0 { return "\(review) earlier request(s) need review after interruption; no automatic retry." }
        let pending = requests.filter { $0.delivery == .pending }
        if !pending.isEmpty {
            return pending.contains(where: { !authorizedFleetFollowups.contains($0.id) })
                ? "\(pending.count) saved follow-up(s). Resume queued to allow delivery."
                : "\(pending.count) follow-up(s) queued for this agent's next safe turn."
        }
        if requests.contains(where: { $0.delivery == .inFlight }) { return "Follow-up sent to this agent." }
        return nil
    }

    private var composerDraftID: UUID? { responseRecipient?.id ?? selectedSessionID }
    func captureResponseDraft() {
        guard !restoringResponseDraft, let id = composerDraftID else { return }
        if composer.isEmpty { responseNavigation.drafts.removeValue(forKey: id) }
        else { responseNavigation.drafts[id] = composer }
    }
    func responseComposerChanged() {
        guard !restoringResponseDraft else { return }
        if !composer.isEmpty, let reply = selectedResponse { responseNavigation.unreadReplyIDs.remove(reply.id) }
        captureResponseDraft(); save()
    }
    func restoreResponseDraft() {
        restoringResponseDraft = true
        composer = composerDraftID.flatMap { responseNavigation.drafts[$0] } ?? ""
        restoringResponseDraft = false
    }
    func selectResponseRecipient(_ id: UUID) {
        guard let recipient = responseAgents.first(where: { $0.id == id }), let agent = recipient.sharedAgent else { return }
        captureResponseDraft()
        responseNavigation.recipientIndices[agent.groupID] = agent.index
        if selectedSharedGroupID == nil { selectedSessionID = recipient.id; provider = recipient.provider }
        if let reply = responseEntries.last(where: { $0.agentIndex == agent.index }) {
            responseNavigation.select(reply); responseScrollTarget = reply.id
            if selectedSharedGroupID == nil { selectedSessionID = reply.sessionID }
        } else {
            responseNavigation.selectedReplyIDs.removeValue(forKey: agent.groupID)
            responseScrollTarget = nil
        }
        restoreResponseDraft(); responseFocusRequest = UUID(); save()
    }
    func selectResponse(_ id: String, markRead: Bool = true) {
        guard let reply = responseEntries.first(where: { $0.id == id }) else { return }
        captureResponseDraft(); responseNavigation.select(reply)
        if !markRead { responseNavigation.unreadReplyIDs.insert(reply.id) }
        if selectedSharedGroupID == nil {
            selectedSessionID = reply.sessionID
            provider = sessions.first(where: { $0.id == reply.sessionID })?.provider ?? provider
        }
        restoreResponseDraft()
        responseScrollTarget = reply.id; responseFocusRequest = UUID(); save()
    }
    func previousResponse() {
        guard !responseEntries.isEmpty else { return }
        let index = max(0, (responsePosition ?? responseEntries.count) - 1)
        selectResponse(responseEntries[index].id)
    }
    func nextResponse() {
        guard !responseEntries.isEmpty else { return }
        let index = min(responseEntries.count - 1, (responsePosition ?? -1) + 1)
        selectResponse(responseEntries[index].id)
    }
    func nextPendingResponse() {
        if let reply = responseEntries.first(where: { responseNavigation.unreadReplyIDs.contains($0.id) && $0.id != selectedResponse?.id }) { selectResponse(reply.id) }
    }
    /// Only a successful provider terminal event can publish a reply. Replaying
    /// drained CLI output is harmless because transcript message IDs deduplicate.
    func receiveCompletedResponse(_ id: UUID) {
        guard let session = sessions.first(where: { $0.id == id }), session.sharedAgent != nil,
              let turnIDs = eventMessages[id], let message = session.messages.last(where: {
                  $0.role == "assistant" && turnIDs.values.contains($0.id) && !$0.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
              }) else { return }
        let reply = FleetReply(session: session, message: message)
        guard responseNavigation.receive(reply) else { return }
        if followResponses, reply.projectID == selectedProjectID, reply.groupID == responseGroupID, composer.isEmpty {
            selectResponse(reply.id, markRead: false)
        }
        save()
    }

    func sendResponseAndAdvance() {
        guard let recipient = responseRecipient else { return }
        acceptFleetFollowup(recipient.id)
    }
    func acceptFleetFollowup(_ id: UUID) {
        guard let recipient = responseAgents.first(where: { $0.id == id }), recipient.projectID == selectedProjectID,
              recipient.sharedAgent?.groupID == responseGroupID else { error = "Choose an agent in this group before sending a follow-up."; return }
        let text = composer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        if recipient.cloud.map({ cloudCleanup.contains($0.sandboxID) }) == true { error = "Wait for this agent's cloud recovery to finish."; return }
        // Explicit send selects the intended worker before clearing its draft.
        if responseRecipient?.id != id { selectResponseRecipient(id); composer = text }
        let followup = FleetFollowup(session: recipient, text: text)
        responseNavigation.followups.append(followup)
        authorizedFleetFollowups.insert(followup.id)
        guard persist() else {
            responseNavigation.followups.removeAll { $0.id == followup.id }
            authorizedFleetFollowups.remove(followup.id); return
        }
        append(recipient.id, role: "user", text: "Queued follow-up for Agent \(followup.agentIndex):\n\n" + text)
        composer = ""; error = nil
        if let reply = selectedResponse { responseNavigation.unreadReplyIDs.remove(reply.id) }
        _ = dispatchFleetFollowup(to: recipient.id)
        nextPendingResponse(); save()
    }
    func resumeQueuedFollowups() {
        guard canResumeQueuedFollowups, let recipient = responseRecipient else { return }
        for item in responseNavigation.followups where item.matches(recipient) && item.delivery == .pending {
            authorizedFleetFollowups.insert(item.id)
        }
        _ = dispatchFleetFollowup(to: recipient.id); save()
    }
    private func mayDispatchFleetFollowup(_ session: ResearchSession) -> Bool {
        guard runners[session.id] == nil, let agent = session.sharedAgent else { return false }
        if !agent.isIndependent && sharedActive(session.projectID) { return false }
        if let cloud = session.cloud {
            return !cloudCleanup.contains(cloud.sandboxID) && !sessions.contains { $0.id != session.id && $0.cloud?.sandboxID == cloud.sandboxID && runners[$0.id] != nil }
        }
        guard let path = session.workingDirectory else { return false }
        return !activePaths.values.contains(URL(fileURLWithPath: path).resolvingSymlinksInPath().path)
    }
    @discardableResult func dispatchFleetFollowup(to id: UUID) -> Bool {
        guard let target = sessions.first(where: { $0.id == id }), mayDispatchFleetFollowup(target),
              responseNavigation.followups.contains(where: { $0.matches(target) && $0.delivery == .pending && authorizedFleetFollowups.contains($0.id) }),
              let project = projects.first(where: { $0.id == target.projectID }) else { return false }
        launch(id, project: project, request: "Continue with the user's queued follow-up below.")
        guard runners[id] != nil else {
            // A synchronous preflight refusal did not consume the request.
            // Keep it saved and let the user explicitly resume after repair.
            suspendFleetFollowups(session: target); save(); return false
        }
        return true
    }
    /// Called only by a real launch after its checkout is reserved. At a shared
    /// generation boundary matching uses logical worker identity, never old IDs.
    func claimFleetFollowups(_ id: UUID, prompt: String) throws -> String {
        guard let session = sessions.first(where: { $0.id == id }), session.sharedAgent != nil else { return prompt }
        let indices = responseNavigation.followups.indices.filter {
            responseNavigation.followups[$0].matches(session) && responseNavigation.followups[$0].delivery == .pending && authorizedFleetFollowups.contains(responseNavigation.followups[$0].id)
        }
        guard !indices.isEmpty else { return prompt }
        for index in indices { responseNavigation.followups[index].delivery = .inFlight; responseNavigation.followups[index].sessionID = id }
        guard persist() else {
            for index in indices { responseNavigation.followups[index].delivery = .pending; responseNavigation.followups[index].sessionID = nil }
            throw ResearchError("Could not save follow-up delivery. No queued request was sent.")
        }
        let requests = indices.map { responseNavigation.followups[$0].text }.joined(separator: "\n\n---\n\n")
        return prompt + "\n\nThe user sent these follow-ups to you, in order. Address them in this turn and retain the existing project constraints:\n\n" + requests
    }
    /// Returns true only when a further user-requested turn has reserved the
    /// same worker. Failure/stop retains requests without an automatic retry.
    @discardableResult func finishFleetTurn(_ id: UUID) -> Bool {
        let finished = sessions.first(where: { $0.id == id })
        let releasedCleanly = finished?.state == "Completed" && completed.contains(id) && !failed.contains(id) && !stopped.contains(id)
        defer { reconsiderFleetFollowups(after: finished, clean: releasedCleanly) }
        guard let session = sessions.first(where: { $0.id == id }), session.sharedAgent != nil else { return false }
        let success = completed.contains(id) && !failed.contains(id) && !stopped.contains(id) && ["Completed", "Needs attention"].contains(session.state)
        for index in responseNavigation.followups.indices where responseNavigation.followups[index].sessionID == id && responseNavigation.followups[index].delivery == .inFlight {
            responseNavigation.followups[index].delivery = success ? .delivered : .needsReview
            authorizedFleetFollowups.remove(responseNavigation.followups[index].id)
        }
        guard success, session.state == "Completed" else {
            if !fleetDrainingForRotation.contains(id) { suspendFleetFollowups(session: session) }
            save(); return false
        }
        save()
        return dispatchFleetFollowup(to: id)
    }
    private func reconsiderFleetFollowups(after finished: ResearchSession?, clean: Bool) {
        guard let finished else { return }
        let waiting = responseNavigation.followups.filter { $0.delivery == .pending && authorizedFleetFollowups.contains($0.id) }
        var visited = Set<UUID>()
        for request in waiting {
            guard let target = SharedTranscript.latestAgents(in: sessions, projectID: request.projectID, groupID: request.groupID).first(where: { $0.sharedAgent?.index == request.agentIndex }),
                  target.id != finished.id, visited.insert(target.id).inserted else { continue }
            if clean { _ = dispatchFleetFollowup(to: target.id) }
            else {
                let sameCloud = finished.cloud != nil && finished.cloud?.sandboxID == target.cloud?.sandboxID
                let sameLocal = finished.cloud == nil && target.cloud == nil && finished.workingDirectory.map { path in
                    target.workingDirectory.map { URL(fileURLWithPath: path).resolvingSymlinksInPath() == URL(fileURLWithPath: $0).resolvingSymlinksInPath() } ?? false
                } == true
                if sameCloud || sameLocal { suspendFleetFollowups(session: target) }
            }
        }
    }
    func suspendFleetFollowups(session: ResearchSession) {
        for item in responseNavigation.followups where item.matches(session) { authorizedFleetFollowups.remove(item.id) }
    }
    func suspendFleetGroup(_ group: UUID) {
        for item in responseNavigation.followups where item.groupID == group { authorizedFleetFollowups.remove(item.id) }
    }
    func inheritFleetDraft(_ session: ResearchSession) {
        guard let previous = session.sharedAgent?.previousSessionID, let draft = responseNavigation.drafts[previous] else { return }
        responseNavigation.drafts[session.id] = draft
    }
    func waitForFleetTurns(_ ids: [UUID]) async {
        while true {
            for id in ids {
                if let task = workerTasks[id] { await task.value }
            }
            if ids.allSatisfy({ runners[$0] == nil }) { return }
        }
    }
}
