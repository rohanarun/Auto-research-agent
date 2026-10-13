import Foundation

public enum Provider: String, Codable, CaseIterable, Identifiable, Sendable {
    case codex, claude
    public var id: String { rawValue }
    public var title: String { self == .codex ? "Codex" : "Claude" }
}

public struct ResearchProject: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var name: String
    public var repository: String
    public var summary: String
    public var category: String
    public var localPath: String?
    public var isCustom: Bool
    public init(repository: String, name: String? = nil, summary: String = "", category: String = "Research", localPath: String? = nil, isCustom: Bool = false) {
        self.id = localPath ?? repository.lowercased()
        self.repository = repository
        self.name = name ?? repository.components(separatedBy: "/").last ?? repository
        self.summary = summary
        self.category = category
        self.localPath = localPath
        self.isCustom = isCustom
    }
    public var githubURL: URL? { repository.isEmpty ? nil : URL(string: "https://github.com/\(repository)") }
}

public struct ChatMessage: Codable, Identifiable, Sendable {
    public var id: UUID
    public var role: String
    public var text: String
    public var date: Date
    public init(role: String, text: String) { id = UUID(); self.role = role; self.text = text; date = Date() }
}

public struct ResearchSession: Codable, Identifiable, Sendable {
    public var id: UUID
    public var projectID: String
    public var title: String
    public var provider: Provider
    public var providerSessionID: String?
    public var workingDirectory: String?
    public var messages: [ChatMessage]
    public var createdAt: Date
    public var state: String
    public var sharedAgent: SharedAgent?
    public var closedAt: Date?
    public init(projectID: String, title: String, provider: Provider) {
        id = UUID(); self.projectID = projectID; self.title = title; self.provider = provider
        messages = []; createdAt = Date(); state = "Ready"
    }
}

public struct Settings: Codable, Sendable {
    public var workspaceRoot: String
    public var catalogURL: String
    public var codexPath: String
    public var claudePath: String
    public var codexModel: String
    public var claudeModel: String
    public var codexSandbox: String
    public var claudePermissionMode: String
    public func executable(_ provider: Provider) -> String { provider == .codex ? codexPath : claudePath }
}

public struct SavedState: Codable {
    public var projects: [ResearchProject]
    public var sessions: [ResearchSession]
    public var selectedProjectID: String?
    public var selectedSessionID: UUID?
    public var provider: Provider
    public var sharedOptions: [String: SharedContextOptions]?
    public var selectedSharedGroupID: UUID?
    public init(projects: [ResearchProject], sessions: [ResearchSession], selectedProjectID: String?, selectedSessionID: UUID?, provider: Provider) {
        self.projects = projects; self.sessions = sessions; self.selectedProjectID = selectedProjectID; self.selectedSessionID = selectedSessionID; self.provider = provider
    }
}

public struct ResearchError: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
