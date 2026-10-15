import Foundation
import ResearchCore

struct SharedRunStatus {
    var groupID: UUID
    var sharesContext = true
    var generation = 1
    var phase: SharedContextCycle.Phase = .preparing
    var sessionIDs: [UUID] = []
    var message: String?
}

extension AppStore {
    func selectSharedGroup(_ groupID: UUID, projectID: String) {
        selectedProjectID = projectID; selectedSessionID = nil; selectedSharedGroupID = groupID
        showingProjectDetails = false; composer = ""; save()
    }
    var projectSharedGroups: [ResearchSession] {
        var seen = Set<UUID>()
        return projectSessions.filter { session in
            guard let group = session.sharedAgent?.groupID else { return false }
            return seen.insert(group).inserted
        }
    }
    func relevantSharedGroup(_ projectID: String) -> UUID? {
        if selectedProjectID == projectID, let id = selectedSharedGroupID { return id }
        if session?.projectID == projectID, let id = session?.sharedAgent?.groupID { return id }
        if sharedActive(projectID) { return sharedRuns[projectID]?.groupID }
        return sessions.filter { $0.projectID == projectID && $0.sharedAgent != nil && $0.closedAt == nil }.max(by: { $0.createdAt < $1.createdAt })?.sharedAgent?.groupID
    }
    func sharedConfiguration(_ projectID: String) -> SharedContextOptions { sharedOptions[projectID] ?? sharedDefaults }
    func sharedActive(_ projectID: String) -> Bool { sharedCycles[projectID]?.active == true }
    func setSharedConfiguration(_ options: SharedContextOptions, for projectID: String) {
        if sharedActive(projectID) {
            if !options.parallelEnabled { stopSharedResearch(projectID) } else { return }
        }
        sharedOptions[projectID] = options; save()
    }
    func stopSharedResearch(_ projectID: String) {
        sharedCycles[projectID]?.stop()
        // Independent cycles await unstructured worker tasks. Cancelling the
        // cycle does not cancel those tasks, so signal their owned runners now
        // rather than waiting for completion before reaching the drain hook.
        for id in sharedRuns[projectID]?.sessionIDs ?? [] { stopWorker(id) }
    }
    func synchronizeSharedResearch(_ projectID: String) { sharedCycles[projectID]?.synchronize() }

    func startSharedResearch(_ project: ResearchProject, request: String? = nil) {
        if daytona.enabled { startCloudResearch(project, request:request);return }
        guard !running(project) else { error = "Stop the existing run before starting parallel agents."; return }
        let options = sharedConfiguration(project.id)
        do { try options.validate() } catch { self.error = error.localizedDescription; return }
        let groupID = UUID(); let provider = self.provider
        let source = URL(fileURLWithPath: project.localPath ?? URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("projects/\(project.repository)").path).resolvingSymlinksInPath()
        guard !activePaths.values.contains(source.path) else { error = "Another run is using this checkout. Stop it before creating a parallel group."; return }
        activePaths[groupID] = source.path
        let operation = ProcessRunner(); sharedOperations[project.id] = operation
        sharedRuns[project.id] = SharedRunStatus(groupID: groupID, sharesContext: options.enabled)
        let directoryName = options.enabled ? "shared-context" : "parallel-agents"
        var directories: [URL] = []
        let hooks = SharedContextCycle.Hooks(
            prepare: {
                let env = try self.environment()
                guard FileManager.default.isExecutableFile(atPath: self.settings.executable(provider)) else { throw ResearchError("Install and connect \(provider.title) in Connections & settings first.") }
                let root = try Workspace.directory(self.settings.workspaceRoot, "\(directoryName)/\(groupID.uuidString)")
                if !FileManager.default.fileExists(atPath: source.path) {
                    guard project.localPath == nil else { throw ResearchError("The selected local repository is missing.") }
                    let parent = try Workspace.directory(self.settings.workspaceRoot, "projects/\(project.repository.components(separatedBy: "/")[0])")
                    let result = try await operation.run(Command("/usr/bin/git", ["clone", "--", "https://github.com/\(project.repository).git", source.path]), directory: parent, environment: env)
                    guard result.code == 0 else { throw ResearchError("Clone failed: \(result.stderr)") }
                }
                for number in 1...options.agentCount {
                    try Task.checkCancellation()
                    let destination = root.appendingPathComponent("agent-\(number)")
                    try await SharedWorkspace.prepare(source: source, destination: destination, runner: operation, environment: env)
                    directories.append(destination)
                }
            },
            launch: { generation, context, previous in
                var ids: [UUID] = []
                for (offset, directory) in directories.enumerated() {
                    var session = ResearchSession(projectID: project.id, title: options.enabled ? "Agent \(offset + 1) · round \(generation)" : "Agent \(offset + 1) · independent", provider: provider)
                    session.workingDirectory = directory.path
                    session.sharedAgent = SharedAgent(groupID: groupID, index: offset + 1, generation: generation, previousSessionID: previous.indices.contains(offset) ? previous[offset] : nil, contextFile: context?.path, sharesContext: options.enabled, agentCount: options.agentCount)
                    let mode = options.enabled ? "Context sharing" : "Independent research; no context sharing or session replacement"
                    session.messages.append(ChatMessage(role: "status", text: "\(mode) · agent \(offset + 1) of \(options.agentCount). Independent checkout: \(directory.path). New provider session; previous chats are preserved."))
                    self.sessions.append(session); ids.append(session.id)
                }
                // Persist every successor and its lineage before starting any provider.
                guard self.persist() else { throw ResearchError("Could not save successor sessions. No replacement agents were started.") }
                if self.selectedSharedGroupID == nil, self.selectedProjectID == project.id, self.selectedSessionID == nil || previous.contains(self.selectedSessionID!) {
                    let selectedIndex = previous.firstIndex(of: self.selectedSessionID ?? UUID()) ?? 0
                    self.selectedSessionID = ids[selectedIndex]
                }
                for id in ids { self.launch(id, project: project, request: request) }
                return ids
            },
            drain: { ids in
                let tasks = ids.compactMap { self.workerTasks[$0] }
                for id in ids { self.stopWorker(id) }
                for task in tasks { await task.value }
                for id in ids { self.workerTasks.removeValue(forKey: id) }
                self.persist()
            },
            capture: { ids, generation, previous in
                let folder = try Workspace.directory(self.settings.workspaceRoot, "shared-context/\(groupID.uuidString)/round-\(generation)")
                var agents: [SharedContextManifest.Agent] = []
                for id in ids {
                    guard let session = self.sessions.first(where: { $0.id == id }), let agent = session.sharedAgent, let checkout = session.workingDirectory else { throw ResearchError("An agent context is missing. No replacement sessions were started.") }
                    guard session.state != "Needs attention" else { throw ResearchError("Agent \(agent.index) needs attention. Its chat and files were preserved.") }
                    let transcript = folder.appendingPathComponent("agent-\(agent.index).json")
                    try Workspace.write(session, to: transcript)
                    let liveNotes = URL(fileURLWithPath: checkout).appendingPathComponent("RESEARCH_HANDOFF.md")
                    var notes: String?
                    if FileManager.default.fileExists(atPath: liveNotes.path) {
                        let savedNotes = folder.appendingPathComponent("agent-\(agent.index)-notes.md")
                        try Data(contentsOf: liveNotes).write(to: savedNotes, options: .atomic)
                        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: savedNotes.path)
                        notes = savedNotes.path
                    }
                    let documents = try SharedContextFiles.snapshotDocuments(checkout: URL(fileURLWithPath: checkout), round: folder, agent: agent.index)
                    let checkpoint = try SharedContextFiles.snapshotCheckpoint(checkout: URL(fileURLWithPath: checkout), round: folder, agent: agent.index)
                    agents.append(.init(number: agent.index, sessionID: id, transcript: transcript.path, checkout: checkout, notes: notes, contextFiles: documents, checkpoint: checkpoint))
                }
                let manifest = folder.appendingPathComponent("manifest.json")
                try Workspace.write(SharedContextManifest(generation: generation, project: project.repository, previousContext: previous?.path, agents: agents), to: manifest)
                return manifest
            },
            merge: { manifest in
                let policy = try String(contentsOf: self.resourceDirectory.appendingPathComponent("SharedContextPolicy.md"), encoding: .utf8)
                try Task.checkCancellation()
                return try SharedContextFiles.publish(manifest: manifest, policy: policy)
            },
            retire: { ids in for id in ids { self.state(id, "Replaced"); self.append(id, role: "status", text: "Context files saved and indexed. A fresh agent session continues in the same checkout.") }; self.persist() },
            cancelOperation: {
                operation.stop()
                if !options.enabled {
                    for id in self.sharedRuns[project.id]?.sessionIDs ?? [] { self.stopWorker(id) }
                }
            },
            update: { phase, generation, ids, message in
                self.sharedRuns[project.id] = SharedRunStatus(groupID: groupID, sharesContext: options.enabled, generation: generation, phase: phase, sessionIDs: ids, message: message)
                if phase == .failed || phase == .stopped || phase == .completed {
                    self.activePaths.removeValue(forKey: groupID)
                    self.sharedOperations.removeValue(forKey: project.id)
                    if let message { self.error = message }
                    self.persist()
                }
            },
            waitForCompletion: { ids in
                let tasks = ids.compactMap { self.workerTasks[$0] }
                for task in tasks { await task.value }
                for id in ids { self.workerTasks.removeValue(forKey: id) }
                if self.sessions.contains(where: { ids.contains($0.id) && $0.state == "Needs attention" }) {
                    throw ResearchError("An independent agent needs attention. Other agents finished their own runs; all chats and files are preserved.")
                }
            }
        )
        select(project)
        selectSharedGroup(groupID, projectID: project.id)
        let cycle = SharedContextCycle(options: options, hooks: hooks)
        sharedCycles[project.id] = cycle; cycle.start()
    }
}
