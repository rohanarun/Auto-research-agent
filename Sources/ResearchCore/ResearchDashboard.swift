import Foundation

public struct ResearchDashboard: Sendable {
    public let history: DashboardHistoryFeed
    public let research: DashboardResearchFeed
    public func scope(for repository: String) -> DashboardScope? {
        history.scopes.values.first { $0.repo?.lowercased() == repository.lowercased() }
    }
    public func groups(for repository: String) -> [DashboardGroup] {
        guard let scope = scope(for: repository) else { return [] }
        return history.groups.values.filter { $0.track == scope.id }.sorted { $0.id < $1.id }
    }
    public func track(for repository: String) -> DashboardTrack? {
        scope(for: repository).flatMap { research.tracks[$0.id] }
    }
    public static func decode(history: Data, research: Data) throws -> Self {
        let decoder = JSONDecoder()
        return try Self(history: decoder.decode(DashboardHistoryFeed.self, from: history), research: decoder.decode(DashboardResearchFeed.self, from: research))
    }
    public static func fetch(catalogURL: String) async throws -> Self {
        guard let base = URL(string: catalogURL), base.scheme == "https" else { throw ResearchError("Use an HTTPS backend address.") }
        async let history = data(base.appendingPathComponent("api/contributors"))
        async let research = data(base.appendingPathComponent("api/research"))
        return try await decode(history: history, research: research)
    }
    private static func data(_ url: URL) async throws -> Data {
        var request = URLRequest(url: url); request.setValue("application/json", forHTTPHeaderField: "Accept")
        request.cachePolicy = .reloadIgnoringLocalCacheData
        let (data, response) = try await URLSession.shared.data(for: request)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw ResearchError("The research backend is unavailable.") }
        return data
    }
}
public struct DashboardHistoryFeed: Decodable, Sendable {
    public let as_of: String
    public let source_error: String?
    public let refresh_after_seconds: Double
    public let scopes: [String: DashboardScope]
    public let groups: [String: DashboardGroup]
}
public struct DashboardScope: Decodable, Sendable {
    public let id: String
    public let name: String
    public let repo: String?
}
public struct DashboardGroup: Decodable, Identifiable, Sendable {
    public let id: String
    public let track: String
    public let name: String
    public let direction: String
    public let reconstructed: DashboardHistory
    public let observed: DashboardHistory
}
public struct DashboardHistory: Decodable, Sendable {
    public let timeline: [DashboardInterval]
    public let value: Double?
    public var validIntervals: [DashboardInterval] {
        timeline.filter { $0.value?.isFinite == true && $0.from.isFinite && $0.to.isFinite && $0.to >= $0.from }.sorted { $0.from < $1.from }
    }
}
public struct DashboardInterval: Decodable, Sendable {
    public let from: Double
    public let to: Double
    public let value: Double?
    public let sources: [String]
    public let leaders: [String]
    public var sourceURL: URL? { sources.compactMap(URL.init(string:)).first { $0.scheme == "https" } }
}
public struct DashboardResearchFeed: Decodable, Sendable {
    public let tracks: [String: DashboardTrack]
    public let error: String?
}
public struct DashboardTrack: Decodable, Sendable {
    public let pulls: [DashboardPull]
    public let checked_at: String?
    public var recentPulls: [DashboardPull] { pulls.sorted { ($0.updated_at ?? $0.created_at) > ($1.updated_at ?? $1.created_at) } }
    public func status(for interval: DashboardInterval?) -> String {
        guard let interval else { return "No claims" }
        let matching = pulls.filter { interval.sources.contains($0.html_url) }
        if matching.contains(where: { $0.state == "open" && $0.merged_at == nil }) { return "Open proposal" }
        if !matching.isEmpty && matching.allSatisfy({ $0.merged_at != nil }) { return "Merged PR" }
        return "Source claim"
    }
}
public struct DashboardPull: Decodable, Identifiable, Sendable {
    public let number: Int
    public let title: String
    public let html_url: String
    public let state: String
    public let merged_at: String?
    public let created_at: String
    public let updated_at: String?
    public let author: String
    public var id: Int { number }
    public var status: String { merged_at != nil ? "Merged" : state.capitalized }
    public var url: URL? { guard let u = URL(string: html_url), u.scheme == "https" else { return nil }; return u }
}
