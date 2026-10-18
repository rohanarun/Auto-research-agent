import Foundation

/// A saved group owns its membership. Current defaults must never resize it.
public struct CloudGroupResume {
    public let groupID: UUID
    public let sessions: [ResearchSession]
    public let generation: Int
    public let context: URL?
    public let options: SharedContextOptions

    public init(sessions all: [ResearchSession], projectID: String, groupID: UUID, options current: SharedContextOptions) throws {
        let saved = SharedTranscript.latestAgents(in: all, projectID: projectID, groupID: groupID)
        guard let first = saved.first?.sharedAgent, first.generation > 0,
              let count = first.agentCount, count == saved.count,
              saved.map({ $0.sharedAgent!.index }) == Array(1...count),
              Set(saved.compactMap { $0.cloud?.sandboxID }).count == count,
              Set(saved.compactMap { $0.cloud?.stateID }).count == count,
              saved.allSatisfy({ session in
                  guard let agent = session.sharedAgent, let cloud = session.cloud,
                        let providerID = session.providerSessionID else { return false }
                  return session.provider == .codex && !providerID.isEmpty && !cloud.stateID.isEmpty && !cloud.sandboxID.isEmpty
                      && agent.generation == first.generation && agent.agentCount == count
                      && agent.sharesContext == first.sharesContext && agent.contextFile == first.contextFile
              }) else { throw ResearchError("Saved cloud group is incomplete or inconsistent. Its sessions and workspaces were preserved.") }
        guard !current.usesMethods else { throw ResearchError("Disable optional research methods before resuming this saved parallel group. Its existing method settings were not changed.") }
        if let path = first.contextFile, !FileManager.default.fileExists(atPath: path) {
            throw ResearchError("The saved group context file is missing. Restore it before resuming.")
        }
        var options = current
        options.agentCount = count; options.parallelEnabled = true; options.enabled = !first.isIndependent
        try options.validate()
        self.groupID = groupID; self.sessions = saved; self.generation = first.generation
        self.context = first.contextFile.map { URL(fileURLWithPath: $0) }; self.options = options
    }
}
