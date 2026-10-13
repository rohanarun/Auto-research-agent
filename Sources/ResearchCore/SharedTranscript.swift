import Foundation

public struct SharedTranscriptEntry: Identifiable {
    public var id: String { "\(sessionID.uuidString)/\(message.id.uuidString)" }
    public let sessionID: UUID
    public let agent: SharedAgent
    public let provider: Provider
    public let message: ChatMessage
    fileprivate let position: Int
}

public enum SharedTranscript {
    public static func sessions(in sessions: [ResearchSession], projectID: String, groupID: UUID) -> [ResearchSession] {
        sessions.filter { $0.projectID == projectID && $0.sharedAgent?.groupID == groupID }
    }
    public static func entries(in sessions: [ResearchSession], projectID: String, groupID: UUID) -> [SharedTranscriptEntry] {
        self.sessions(in: sessions, projectID: projectID, groupID: groupID).flatMap { session in
            session.messages.enumerated().map { position, message in
                SharedTranscriptEntry(sessionID: session.id, agent: session.sharedAgent!, provider: session.provider, message: message, position: position)
            }
        }.sorted { a, b in
            if a.message.date != b.message.date { return a.message.date < b.message.date }
            if a.agent.generation != b.agent.generation { return a.agent.generation < b.agent.generation }
            if a.agent.index != b.agent.index { return a.agent.index < b.agent.index }
            if a.sessionID != b.sessionID { return a.sessionID.uuidString < b.sessionID.uuidString }
            return a.position < b.position
        }
    }
    public static func latestAgents(in sessions: [ResearchSession], projectID: String, groupID: UUID) -> [ResearchSession] {
        let grouped = Dictionary(grouping: self.sessions(in: sessions, projectID: projectID, groupID: groupID)) { $0.sharedAgent!.index }
        return grouped.keys.sorted().compactMap { index in
            grouped[index]?.max {
                if $0.sharedAgent!.generation != $1.sharedAgent!.generation { return $0.sharedAgent!.generation < $1.sharedAgent!.generation }
                return $0.createdAt < $1.createdAt
            }
        }
    }
}
