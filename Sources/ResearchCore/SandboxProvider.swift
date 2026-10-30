import Foundation

public enum SandboxProvider: String, Codable, CaseIterable, Sendable {
    case daytona, agent37
    public var title: String { self == .daytona ? "Daytona" : "Agent37" }
    public var apiURL: URL { URL(string: self == .daytona ? "https://app.daytona.io/api" : "https://api.agent37.com/v1")! }
    public var dashboardURL: URL { URL(string: self == .daytona ? "https://app.daytona.io/dashboard/sandboxes" : "https://www.agent37.com/dashboard/cloud")! }
    public var configurationFile: String { rawValue + ".json" }
    public var defaultsFile: String { self == .daytona ? "DaytonaDefaults.json" : "Agent37Defaults.json" }
}
