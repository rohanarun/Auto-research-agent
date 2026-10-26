import SwiftUI
import ResearchCore

struct GatewaySettingsView: View {
    @ObservedObject var store: AppStore
    @State private var key = ""
    @State private var configuration = OpenRouterConfiguration()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                VStack(alignment: .leading, spacing: 6) {
                    Text("OpenRouter gateway").font(.headline)
                    Text("Use your OpenRouter account for Codex and Claude runs on this Mac, including local parallel research. Requests are billed to your OpenRouter account.")
                        .font(.system(size: 13)).foregroundStyle(.secondary)
                }
                VStack(alignment: .leading, spacing: 10) {
                    HStack {
                        Text(store.gatewayKeyConfigured ? "API key saved in Keychain" : "Add your own API key")
                            .font(.system(size: 12, weight: .medium))
                        Spacer()
                        Link("Get a key", destination: URL(string: "https://openrouter.ai/settings/keys")!)
                    }
                    HStack {
                        SecureField("OpenRouter API key", text: $key)
                            .accessibilityLabel("OpenRouter API key")
                        Button("Save key") {
                            if store.saveGatewayKey(key) { key = "" }
                        }.disabled(key.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                    }
                    Text("The key stays in macOS Keychain. It is supplied to the selected local CLI at launch, never saved in project settings or prompts.")
                        .font(.system(size: 11)).foregroundStyle(.secondary)
                    Button("Remove saved key") { store.removeGatewayKey() }
                        .disabled(!store.gatewayKeyConfigured)
                }.padding(16).background(Color.gray.opacity(0.06), in: RoundedRectangle(cornerRadius: 12))

                Toggle("Use OpenRouter gateway for Codex and Claude", isOn: $configuration.enabled)
                    .toggleStyle(.switch)
                VStack(alignment: .leading, spacing: 10) {
                    Text("OpenRouter model IDs").font(.system(size: 12, weight: .medium))
                    TextField("Codex model ID from OpenRouter", text: $configuration.codexModel)
                        .accessibilityLabel("Codex OpenRouter model ID")
                    TextField("Claude model ID from OpenRouter", text: $configuration.claudeModel)
                        .accessibilityLabel("Claude OpenRouter model ID")
                    Text("Enter the model for each CLI you plan to use. Gateway models are separate from your direct-login defaults. Claude Code works best with compatible Anthropic models.")
                        .font(.system(size: 11)).foregroundStyle(.secondary)
                }
                HStack {
                    Button("Save gateway settings") {
                        store.gateway = configuration
                        if store.saveGatewayConfiguration() { configuration = store.gateway }
                    }.buttonStyle(.borderedProminent)
                    Text(store.gatewayStatus).font(.system(size: 12)).foregroundStyle(.secondary)
                }
                Text("Turn gateway mode off to use your existing CLI sign-in. To change a chat’s connection route, start a new chat. Removing the key blocks gateway runs until you add another key or turn gateway mode off. Gateway credentials are not uploaded to Daytona; gateway mode is for local runs.")
                    .font(.system(size: 12)).foregroundStyle(.secondary)
                HStack(spacing: 16) {
                    Link("Codex setup guide", destination: URL(string: "https://openrouter.ai/docs/cookbook/coding-agents/codex-cli")!)
                    Link("Claude setup guide", destination: URL(string: "https://openrouter.ai/docs/cookbook/coding-agents/claude-code-integration")!)
                }.font(.system(size: 12))
            }.textFieldStyle(.roundedBorder).padding(.vertical, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
                .disabled(store.hasActiveRuns || store.loginProvider != nil)
        }
        .onAppear { configuration = store.gateway }
        .onDisappear { key = "" }
    }
}
