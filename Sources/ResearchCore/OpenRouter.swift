import Foundation

/// Nonsecret, app-scoped connection settings. Credentials never belong in this value.
public struct OpenRouterConfiguration: Codable, Equatable, Sendable {
    public var enabled = false
    public var codexModel = ""
    public var claudeModel = ""
    public init(enabled: Bool = false, codexModel: String = "", claudeModel: String = "") {
        self.enabled = enabled; self.codexModel = codexModel; self.claudeModel = claudeModel
    }
    public func applies(to provider: Provider) -> Bool { enabled && provider != .opencode }
    public func model(for provider: Provider) -> String {
        (provider == .codex ? codexModel : claudeModel).trimmingCharacters(in: .whitespacesAndNewlines)
    }
    public func validate(for provider: Provider) throws {
        guard applies(to: provider) else { return }
        let value = model(for: provider)
        guard !value.isEmpty, !value.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) else {
            throw ResearchError("Enter an OpenRouter model ID for \(provider.title) in Connections → Gateway before starting this run.")
        }
    }
    public func route(for provider: Provider) -> String { applies(to: provider) ? "openrouter:\(provider.rawValue)" : "direct:\(provider.rawValue)" }
    public func validateResume(provider: Provider, savedRoute: String?, hasProviderSession: Bool) throws {
        let previous = savedRoute ?? "direct:\(provider.rawValue)"
        guard !hasProviderSession || previous == route(for: provider) else {
            throw ResearchError("This chat belongs to a different connection route. Restore its original Gateway setting to continue it, or start a new chat. Existing provider sessions are preserved.")
        }
    }
    public var codexOverrides: [String] {
        // Command-backed auth enables the gateway's Codex-native catalog. No credential appears in argv.
        ["model_provider=\"autoresearch_openrouter\"",
         "model_providers.autoresearch_openrouter={ name=\"OpenRouter\", base_url=\"https://openrouter.ai/api/v1\", wire_api=\"responses\", auth={command=\"/usr/bin/printenv\", args=[\"OPENROUTER_API_KEY\"]} }"]
    }
    public func environment(provider: Provider, key: String, claudeConfigDirectory: String) -> [String: String] {
        guard applies(to: provider) else { return [:] }
        if provider == .codex { return ["OPENROUTER_API_KEY": key] }
        let model = model(for: provider)
        return ["ANTHROPIC_BASE_URL": "https://openrouter.ai/api", "ANTHROPIC_AUTH_TOKEN": key,
                "ANTHROPIC_API_KEY": "", "CLAUDE_CONFIG_DIR": claudeConfigDirectory,
                "CLAUDE_CODE_OAUTH_TOKEN": "", "CLAUDE_CODE_USE_BEDROCK": "0",
                "CLAUDE_CODE_USE_VERTEX": "0", "CLAUDE_CODE_USE_FOUNDRY": "0",
                "ANTHROPIC_MODEL": model, "ANTHROPIC_SMALL_FAST_MODEL": model,
                "ANTHROPIC_DEFAULT_HAIKU_MODEL": model, "ANTHROPIC_DEFAULT_SONNET_MODEL": model,
                "ANTHROPIC_DEFAULT_OPUS_MODEL": model, "ANTHROPIC_DEFAULT_FABLE_MODEL": model,
                "CLAUDE_CODE_SUBAGENT_MODEL": model]
    }
}

public enum SecretRedaction {
    public static func redact(_ text: String, secrets: [String]) -> String {
        secrets.filter { !$0.isEmpty }.reduce(text) { $0.replacingOccurrences(of: $1, with: "[REDACTED]") }
    }
}
