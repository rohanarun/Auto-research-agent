import Foundation

/// A completed provider turn, identified by the actual transcript message.
public struct FleetReply: Codable, Identifiable, Equatable, Sendable {
    public var id: String { sessionID.uuidString + "/" + messageID.uuidString }
    public let sessionID: UUID
    public let messageID: UUID
    public let projectID: String
    public let groupID: UUID
    public let agentIndex: Int
    public let date: Date
    public init(session: ResearchSession, message: ChatMessage) {
        sessionID = session.id; messageID = message.id; projectID = session.projectID
        groupID = session.sharedAgent!.groupID; agentIndex = session.sharedAgent!.index
        date = message.date
    }
}

public struct FleetFollowup: Codable, Identifiable, Equatable, Sendable {
    public enum Delivery: String, Codable, Sendable { case pending, inFlight, delivered, needsReview }
    public let id: UUID
    public let projectID: String
    public let groupID: UUID
    public let agentIndex: Int
    public let text: String
    public let date: Date
    public var delivery: Delivery = .pending
    public var sessionID: UUID?
    public init(session: ResearchSession, text: String) {
        id = UUID(); projectID = session.projectID; groupID = session.sharedAgent!.groupID
        agentIndex = session.sharedAgent!.index; self.text = text; date = Date()
    }
    public func matches(_ session: ResearchSession) -> Bool {
        projectID == session.projectID && groupID == session.sharedAgent?.groupID && agentIndex == session.sharedAgent?.index
    }
}

/// Saved UI state and human requests contain no authorization to restart work.
/// The app separately holds a transient set of requests approved this launch.
public struct FleetResponseNavigation: Codable, Equatable, Sendable {
    public var followResponses = true
    public var replies: [FleetReply] = []
    public var unreadReplyIDs: Set<String> = []
    public var selectedReplyIDs: [UUID: String] = [:]
    public var recipientIndices: [UUID: Int] = [:]
    public var drafts: [UUID: String] = [:]
    public var followups: [FleetFollowup] = []
    public init() {}

    @discardableResult public mutating func receive(_ reply: FleetReply) -> Bool {
        guard !replies.contains(where: { $0.id == reply.id }) else { return false }
        replies.append(reply); unreadReplyIDs.insert(reply.id)
        return true
    }
    public mutating func select(_ reply: FleetReply) {
        selectedReplyIDs[reply.groupID] = reply.id
        recipientIndices[reply.groupID] = reply.agentIndex
        unreadReplyIDs.remove(reply.id)
    }
    public mutating func restore() {
        for i in followups.indices where followups[i].delivery == .inFlight {
            followups[i].delivery = .needsReview
        }
    }
}
