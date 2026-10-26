import Foundation

public enum Provider: String, Codable, CaseIterable, Identifiable, Sendable {
    case codex, claude, opencode
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .codex: "Codex"
        case .claude: "Claude"
        case .opencode: "OpenCode"
        }
    }
}

public struct ResearchProject: Codable, Identifiable, Hashable, Sendable {
    public var id: String
    public var name: String
    public var repository: String
    public var summary: String
    public var category: String
    public var localPath: String?
    public var isCustom: Bool
    public var builtinID: String?
    public init(repository: String, name: String? = nil, summary: String = "", category: String = "Research", localPath: String? = nil, isCustom: Bool = false, builtinID: String? = nil) {
        self.id = builtinID.map { "builtin:" + $0 } ?? localPath ?? repository.lowercased()
        self.repository = repository
        self.name = name ?? repository.components(separatedBy: "/").last ?? repository
        self.summary = summary
        self.category = category
        self.localPath = localPath
        self.isCustom = isCustom
        self.builtinID = builtinID
    }
    public var githubURL: URL? { repository.isEmpty ? nil : URL(string: "https://github.com/\(repository)") }
    public var managedWorkspaceComponent: String {
        builtinID.map { "projects/builtins/" + $0 } ?? "projects/" + repository
    }
}

public struct ChatMessage: Codable, Identifiable, Sendable {
    public var id: UUID
    public var role: String
    public var text: String
    public var date: Date
    public var sourceKey: String?
    public var speaker: String?
    public init(role: String, text: String) { id = UUID(); self.role = role; self.text = text; date = Date() }
}

public struct ResearchSession: Codable, Identifiable, Sendable {
    public var id: UUID
    public var projectID: String
    public var title: String
    public var provider: Provider
    public var providerSessionID: String?
    /// Absent on legacy chats, which used the direct CLI connection.
    public var connectionRoute: String?
    /// Last explicitly selected model; nil means the CLI chose its default or the chat predates this field.
    public var connectionModel: String?
    public var workingDirectory: String?
    public var messages: [ChatMessage]
    public var createdAt: Date
    public var state: String
    public var sharedAgent: SharedAgent?
    public var cloud: CloudSession?
    /// A saved chat stays on its original backend. The preference selects only
    /// chats that have not yet acquired a checkout or provider conversation.
    public func usesCloud(preferCloud: Bool) -> Bool {
        cloud != nil || (preferCloud && providerSessionID == nil && workingDirectory == nil)
    }
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
    public var opencodePath: String
    public var opencodeModel: String
    /// "reject" lets OpenCode auto-reject permission prompts (its headless default);
    /// "auto" passes --auto, approving anything the user's OpenCode config does not deny.
    public var opencodePermission: String
    public func executable(_ provider: Provider) -> String {
        switch provider {
        case .codex: codexPath
        case .claude: claudePath
        case .opencode: opencodePath
        }
    }
}

extension Settings {
    private enum CodingKeys: String, CodingKey {
        case workspaceRoot, catalogURL, codexPath, claudePath, codexModel, claudeModel, codexSandbox, claudePermissionMode, opencodePath, opencodeModel, opencodePermission
    }
    /// Settings saved before OpenCode support omit its keys; they decode to safe defaults.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        workspaceRoot = try c.decode(String.self, forKey: .workspaceRoot)
        catalogURL = try c.decode(String.self, forKey: .catalogURL)
        codexPath = try c.decode(String.self, forKey: .codexPath)
        claudePath = try c.decode(String.self, forKey: .claudePath)
        codexModel = try c.decode(String.self, forKey: .codexModel)
        claudeModel = try c.decode(String.self, forKey: .claudeModel)
        codexSandbox = try c.decode(String.self, forKey: .codexSandbox)
        claudePermissionMode = try c.decode(String.self, forKey: .claudePermissionMode)
        opencodePath = try c.decodeIfPresent(String.self, forKey: .opencodePath) ?? ""
        opencodeModel = try c.decodeIfPresent(String.self, forKey: .opencodeModel) ?? ""
        opencodePermission = try c.decodeIfPresent(String.self, forKey: .opencodePermission) ?? "reject"
    }
}

public struct SavedState: Codable {
    public var projects: [ResearchProject]
    public var sessions: [ResearchSession]
    public var selectedProjectID: String?
    public var selectedSessionID: UUID?
    public var provider: Provider
    public var sharedOptions: [String: SharedContextOptions]?
    public var selectedSharedGroupID: UUID?
    public var fleetNavigation: FleetResponseNavigation?
    public init(projects: [ResearchProject], sessions: [ResearchSession], selectedProjectID: String?, selectedSessionID: UUID?, provider: Provider) {
        self.projects = projects; self.sessions = sessions; self.selectedProjectID = selectedProjectID; self.selectedSessionID = selectedSessionID; self.provider = provider
    }
}

public struct ResearchError: LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}
