import Foundation
import ResearchCore

struct SharedRunStatus {
    var groupID: UUID
    var generation = 1
    var phase: SharedContextCycle.Phase = .preparing
    var sessionIDs: [UUID] = []
    var message: String?
}

private struct ContextManifest: Encodable {
    struct Agent: Encodable { let number: Int; let sessionID: UUID; let transcript: String; let checkout: String }
    let generation: Int
    let project: String
    let previousContext: String?
    let agents: [Agent]
}

extension AppStore {
    func sharedConfiguration(_ projectID: String) -> SharedContextOptions { sharedOptions[projectID] ?? sharedDefaults }
    func sharedActive(_ projectID: String) -> Bool { sharedCycles[projectID]?.active == true }
    func setSharedConfiguration(_ options: SharedContextOptions, for projectID: String) {
        if sharedActive(projectID) {
            if !options.enabled { stopSharedResearch(projectID) } else { return }
        }
        sharedOptions[projectID] = options; save()
    }
    func stopSharedResearch(_ projectID: String) { sharedCycles[projectID]?.stop() }
    func synchronizeSharedResearch(_ projectID: String) { sharedCycles[projectID]?.synchronize() }

    func startSharedResearch(_ project: ResearchProject, request: String? = nil) {
        guard !running(project) else { error = "Stop the existing run before starting parallel agents."; return }
        let options = sharedConfiguration(project.id)
        do { try options.validate() } catch { self.error = error.localizedDescription; return }
        let groupID = UUID(); let provider = self.provider
        let source = URL(fileURLWithPath: project.localPath ?? URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("projects/\(project.repository)").path).resolvingSymlinksInPath()
        guard !activePaths.values.contains(source.path) else { error = "Another run is using this checkout. Stop it before creating a parallel group."; return }
        activePaths[groupID] = source.path
        let operation = ProcessRunner(); sharedOperations[project.id] = operation
        sharedRuns[project.id] = SharedRunStatus(groupID: groupID)
        var directories: [URL] = []
        let hooks = SharedContextCycle.Hooks(
            prepare: {
                let env = try self.environment()
                guard FileManager.default.isExecutableFile(atPath: self.settings.executable(provider)) else { throw ResearchError("Install and connect \(provider.title) in Connections & settings first.") }
                let root = try Workspace.directory(self.settings.workspaceRoot, "shared-context/\(groupID.uuidString)")
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
                    var session = ResearchSession(projectID: project.id, title: "Agent \(offset + 1) · round \(generation)", provider: provider)
                    session.workingDirectory = directory.path
                    session.sharedAgent = SharedAgent(groupID: groupID, index: offset + 1, generation: generation, previousSessionID: previous.indices.contains(offset) ? previous[offset] : nil, contextFile: context?.path)
                    session.messages.append(ChatMessage(role: "status", text: "Shared context · agent \(offset + 1) of \(options.agentCount). Independent checkout: \(directory.path). New provider session; previous chats are preserved."))
                    self.sessions.append(session); ids.append(session.id)
                }
                // Persist every successor and its lineage before starting any provider.
                guard self.persist() else { throw ResearchError("Could not save successor sessions. No replacement agents were started.") }
                if self.selectedProjectID == project.id, self.selectedSessionID == nil || previous.contains(self.selectedSessionID!) {
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
                var agents: [ContextManifest.Agent] = []
                for id in ids {
                    guard let session = self.sessions.first(where: { $0.id == id }), let agent = session.sharedAgent, let checkout = session.workingDirectory else { throw ResearchError("An agent context is missing. No replacement sessions were started.") }
                    guard session.state != "Needs attention" else { throw ResearchError("Agent \(agent.index) needs attention. Its chat and files were preserved.") }
                    let transcript = folder.appendingPathComponent("agent-\(agent.index).json")
                    try Workspace.write(session, to: transcript)
                    agents.append(.init(number: agent.index, sessionID: id, transcript: transcript.path, checkout: checkout))
                }
                let manifest = folder.appendingPathComponent("manifest.json")
                try Workspace.write(ContextManifest(generation: generation, project: project.repository, previousContext: previous?.path, agents: agents), to: manifest)
                return manifest
            },
            merge: { manifest in
                var config = self.settings; config.codexSandbox = "read-only"
                let directory = manifest.deletingLastPathComponent()
                let env = try self.environment()
                let policy = try String(contentsOf: self.resourceDirectory.appendingPathComponent("SharedContextPolicy.md"), encoding: .utf8)
                var disabled: [String] = []
                if provider == .codex {
                    let discovery = try await operation.run(CLI.mcpDiscovery(config.codexPath), directory: directory, environment: env)
                    guard discovery.code == 0 else { throw ResearchError("Could not inspect MCP settings for context synthesis.") }
                    disabled = try CLI.configuredMCPServers(Data(discovery.stdout.utf8))
                }
                var command = CLI.chat(provider: provider, settings: config, sessionID: nil, prompt: "Read this context manifest and every transcript it references, then combine them according to the synthesis instructions: \(manifest.path)", policy: policy, disabledMCPServers: disabled)
                if provider == .claude { command.arguments += ["--tools", "Read,Grep,Glob", "--strict-mcp-config", "--mcp-config", "{\"mcpServers\":{}}"] }
                let result = try await operation.run(command, directory: directory, environment: env)
                try Workspace.write(result.stdout, to: directory.appendingPathComponent("synthesis-events.json"))
                try Workspace.write(result.stderr, to: directory.appendingPathComponent("synthesis-diagnostics.json"))
                try Task.checkCancellation()
                let events = result.stdout.split(separator: "\n").flatMap { EventDecoder.decode(String($0), provider: provider) }
                guard result.code == 0, events.contains(.completed), !events.contains(where: { if case .failure = $0 { return true }; return false }) else { throw ResearchError("Context synthesis failed. Old chats and snapshots were preserved; no replacements were started. \(result.stderr)") }
                var messages: [String: String] = [:]; var order: [String] = []
                for event in events { if case .message(let id, "assistant", let text) = event { if messages[id] == nil { order.append(id) }; messages[id] = text } }
                let summary = order.compactMap { messages[$0] }.joined(separator: "\n\n")
                guard EventDecoder.sharedContextReady(summary) else { throw ResearchError("The model did not confirm a complete shared handoff. Review the saved synthesis output; old contexts were preserved and no replacement agents were started.") }
                let handoff = directory.appendingPathComponent("shared-context.md")
                // Include durable source references independently of model compression.
                let text = summary + "\n\nSource manifest (all original transcripts and checkouts): \(manifest.path)\n"
                try text.write(to: handoff, atomically: true, encoding: .utf8)
                try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: handoff.path)
                return handoff
            },
            retire: { ids in for id in ids { self.state(id, "Replaced"); self.append(id, role: "status", text: "Context saved and combined. A fresh agent session continues in the same checkout.") }; self.persist() },
            cancelOperation: { operation.stop() },
            update: { phase, generation, ids, message in
                self.sharedRuns[project.id] = SharedRunStatus(groupID: groupID, generation: generation, phase: phase, sessionIDs: ids, message: message)
                if phase == .failed || phase == .stopped {
                    self.activePaths.removeValue(forKey: groupID)
                    self.sharedOperations.removeValue(forKey: project.id)
                    if let message { self.error = message }
                    self.persist()
                }
            }
        )
        select(project)
        let cycle = SharedContextCycle(options: options, hooks: hooks)
        sharedCycles[project.id] = cycle; cycle.start()
    }
}
