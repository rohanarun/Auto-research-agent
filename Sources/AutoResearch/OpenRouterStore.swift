import Foundation
import ResearchCore

extension AppStore {
    func loadGatewayConfiguration() {
        let file = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("openrouter.json")
        do {
            if FileManager.default.fileExists(atPath: file.path) { gateway = try JSONDecoder().decode(OpenRouterConfiguration.self, from: Data(contentsOf: file)) }
            gatewayConfigurationReadable = true
            savedGateway = gateway
        } catch {
            gatewayConfigurationReadable = false
            self.error = "Saved gateway settings could not be read. They have been preserved; repair openrouter.json before starting research."
        }
        refreshGatewayStatus()
    }
    func refreshGatewayStatus() {
        gatewayKeyConfigured = gatewayCredentials.containsKey()
        gatewayStatus = gatewayKeyConfigured ? "Key configured in Keychain · no live request verified" : "No OpenRouter key configured"
    }
    @discardableResult func saveGatewayConfiguration() -> Bool {
        do {
            guard !hasActiveRuns, loginProvider == nil else { throw ResearchError("Stop active research and login before changing the gateway connection.") }
            guard gatewayConfigurationReadable else { throw ResearchError("Repair the unreadable openrouter.json first. Its contents have been preserved.") }
            if gateway.enabled {
                guard !gateway.codexModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !gateway.claudeModel.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ResearchError("Enter an OpenRouter model ID for at least one CLI.") }
                for provider in [Provider.codex, .claude] where !gateway.model(for: provider).isEmpty { try gateway.validate(for: provider) }
                guard gatewayCredentials.containsKey() else { throw ResearchError("Save an OpenRouter key in Keychain before enabling Gateway.") }
            }
            let root = try Workspace.validate(settings.workspaceRoot)
            try Workspace.write(gateway, to: root.appendingPathComponent("openrouter.json"))
            savedGateway = gateway
            refreshGatewayStatus(); return true
        } catch { gateway = savedGateway; self.error = error.localizedDescription; return false }
    }
    @discardableResult func saveGatewayKey(_ value: String) -> Bool {
        do {
            guard !hasActiveRuns, loginProvider == nil else { throw ResearchError("Stop active research and login before replacing the gateway key.") }
            let key = value.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !key.isEmpty, !key.unicodeScalars.contains(where: CharacterSet.controlCharacters.contains) else { throw ResearchError("Enter a valid OpenRouter API key.") }
            try gatewayCredentials.save(key); refreshGatewayStatus(); return true
        } catch { self.error = error.localizedDescription; return false }
    }
    @discardableResult func removeGatewayKey() -> Bool {
        do {
            guard !hasActiveRuns, loginProvider == nil else { throw ResearchError("Stop active research and login before removing the gateway key.") }
            try gatewayCredentials.remove(); refreshGatewayStatus(); return true
        } catch { self.error = error.localizedDescription; return false }
    }
    func gatewayPreflight(_ session: ResearchSession, cloud: Bool) throws {
        guard gatewayConfigurationReadable else { throw ResearchError("Repair the unreadable saved gateway settings before starting research.") }
        try gateway.validateResume(provider: session.provider, savedRoute: session.connectionRoute, hasProviderSession: session.providerSessionID != nil)
        try gateway.validate(for: session.provider)
        guard !(cloud && gateway.applies(to: session.provider)) else {
            throw ResearchError("OpenRouter is configured for This Mac only. Switch execution to This Mac for a new chat, or turn Gateway off to use the existing Daytona connection. No gateway key is uploaded to Daytona.")
        }
    }
}
