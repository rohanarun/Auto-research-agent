import SwiftUI
import AppKit
import ResearchCore

struct ConnectionsView: View {
    @ObservedObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var tab = "Connections"
    init(store: AppStore, initialTab: String = "Connections") { self.store = store;_tab = State(initialValue:initialTab) }
    @State private var daytonaKey = ""
    @State private var loginInput = ""
    private func installURL(_ provider: Provider) -> URL {
        switch provider {
        case .codex: URL(string: "https://developers.openai.com/codex/cli")!
        case .claude: URL(string: "https://code.claude.com/docs/en/overview")!
        case .opencode: URL(string: "https://opencode.ai/docs")!
        }
    }
    private func pathBinding(_ provider: Provider) -> Binding<String> {
        switch provider {
        case .codex: $store.settings.codexPath
        case .claude: $store.settings.claudePath
        case .opencode: $store.settings.opencodePath
        }
    }
    private func modelBinding(_ provider: Provider) -> Binding<String> {
        switch provider {
        case .codex: $store.settings.codexModel
        case .claude: $store.settings.claudeModel
        case .opencode: $store.settings.opencodeModel
        }
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            HStack { Text("Your research workspace").font(.system(size: 23, weight: .semibold)); Spacer(); Button("Done") { store.saveSettings() }.keyboardShortcut(.defaultAction).disabled(store.hasActiveRuns) }
            Picker("Settings", selection: $tab) { Text("Connections").tag("Connections"); Text("Cloud").tag("Daytona"); Text("Lean server").tag("Lean"); Text("MCP tools").tag("MCP"); Text("Research policy").tag("Policy"); Text("Workspace").tag("Workspace") }.pickerStyle(.segmented)
            if tab == "Connections" {
                // Three CLIs no longer fit the fixed sheet height, so the cards scroll.
                ScrollView { VStack(alignment: .leading, spacing: 14) { ForEach(Provider.allCases) { provider in
                    VStack(alignment: .leading, spacing: 12) {
                        HStack {
                            Image(systemName: "terminal").font(.system(size: 21))
                            VStack(alignment: .leading, spacing: 4) { Text(provider.title).font(.system(size: 16, weight: .semibold)); Text(store.auth[provider] ?? "Checking…").font(.system(size: 11)).foregroundStyle(.secondary) }
                            Spacer()
                            Button("Sign in") { store.login(provider) }.disabled(store.loginProvider != nil)
                            Link("Install CLI", destination: installURL(provider))
                        }
                        TextField("CLI executable path", text: pathBinding(provider)).textFieldStyle(.roundedBorder)
                        HStack {
                            TextField(provider == .opencode ? "Model as provider/model (empty uses your CLI default)" : "Model (empty uses your CLI default)", text: modelBinding(provider)).textFieldStyle(.roundedBorder)
                            if provider == .codex {
                                Picker("Access", selection: $store.settings.codexSandbox) { Text("Workspace writes").tag("workspace-write"); Text("Read only").tag("read-only") }.frame(width: 215)
                            } else if provider == .opencode {
                                Picker("Permissions", selection: $store.settings.opencodePermission) { Text("Reject prompts").tag("reject"); Text("Auto-approve").tag("auto") }.frame(width: 230)
                            } else {
                                Picker("Permissions", selection: $store.settings.claudePermissionMode) { Text("Automatic review").tag("auto"); Text("Accept edits").tag("acceptEdits"); Text("Plan only").tag("plan"); Text("Deny unapproved").tag("dontAsk") }.frame(width: 230)
                            }
                        }
                    }.padding(18).background(Color.gray.opacity(0.06), in: RoundedRectangle(cornerRadius: 14)).disabled(store.hasActiveRuns)
                } } }
                Text("Sign-in uses the official CLI and your browser. Credentials stay with each CLI; this app does not copy or store them. Permission errors remain visible in chat.").font(.system(size: 12)).foregroundStyle(.secondary)
                Button("Refresh connection status") { Task { await store.refreshAuth() } }
                if let provider = store.loginProvider {
                    Divider()
                    HStack { Text("\(provider.title) login").font(.headline); Spacer(); Button("Close login") { store.cancelLogin() } }
                    ScrollView { Text(.init(store.loginOutput)).font(.system(size: 11, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading) }.frame(height: 115)
                    HStack { SecureField("Response if the CLI requests a code", text: $loginInput); Button("Submit") { store.sendLoginInput(loginInput); loginInput = "" } }
                }
            } else if tab == "Daytona" {
                VStack(alignment:.leading,spacing:16) {
                    Picker("Run research",selection:Binding(get:{store.daytona.enabled},set:{store.setCloudExecution($0)})) {
                        Text("This Mac").tag(false);Text("\(store.daytona.sandboxProvider.title) sandboxes · Codex").tag(true)
                    }.pickerStyle(.segmented).disabled(store.hasActiveRuns)
                    Picker("Sandbox provider",selection:Binding(get:{store.daytona.sandboxProvider},set:{store.selectSandboxProvider($0);daytonaKey = ""})) {
                        ForEach(SandboxProvider.allCases,id:\.self) { Text($0.title).tag($0) }
                    }.disabled(store.hasActiveRuns)
                    Text("Private parallel sandboxes").font(.headline)
                    Text("Each agent has its own Linux checkout. Share context uses live-written checkpoint files, combines them directly, then supplies the bundle to fresh sessions. Your Mac account files are never uploaded.").font(.system(size:13)).foregroundStyle(.secondary)
                    HStack {
                        SecureField("\(store.daytona.sandboxProvider.title) API key",text:$daytonaKey).textFieldStyle(.roundedBorder)
                        Button("Save key") { store.saveDaytonaKey(daytonaKey);daytonaKey = "" }.disabled(daytonaKey.isEmpty || store.hasActiveRuns || store.daytonaBusy)
                    }
                    HStack {
                        Text(store.daytonaStatus).font(.system(size:12)).textSelection(.enabled)
                        Spacer()
                        Button("Connect") { store.connectDaytona(signIn:false) }.disabled(store.daytonaBusy || store.hasActiveRuns)
                        Button("Sign in to cloud Codex") { store.connectDaytona(signIn:true) }.disabled(store.daytonaBusy || store.hasActiveRuns)
                    }
                    Text("The API key is stored in macOS Keychain. Cloud model access uses the official device login separately. Cloud compute is billed to your \(store.daytona.sandboxProvider.title) account. Completed work is downloaded and verified before workers are \(store.daytona.sandboxProvider == .daytona ? "archived" : "stopped with their disks retained"); failed exports preserve the sandbox. Idle sandboxes \(store.daytona.sandboxProvider == .daytona ? "stop" : "sleep") after \(store.daytona.idleStopMinutes) minutes.").font(.system(size:12)).foregroundStyle(.secondary)
                    if !store.daytonaLoginOutput.isEmpty {
                        ScrollView { Text(store.daytonaLoginOutput).font(.system(size:11,design:.monospaced)).textSelection(.enabled).frame(maxWidth:.infinity,alignment:.leading) }.frame(height:150)
                        if store.daytonaBusy { Button("Cancel cloud sign-in") { store.cancelCloudLogin() } }
                        Link("Open Codex device sign-in",destination:URL(string:"https://auth.openai.com/codex/device")!)
                    }
                    HStack {
                        Button("Stop owned cloud compute") { store.stopCloudCompute() }.disabled(store.hasActiveRuns || store.daytonaBusy || !store.cloudCleanup.isEmpty)
                        Button("Remove saved key") { store.removeDaytonaKey() }.disabled(store.hasActiveRuns || store.daytonaBusy || !store.cloudCleanup.isEmpty)
                    }
                    if !store.cloudCleanup.isEmpty { Text("Verifying and stopping \(store.cloudCleanup.count) worker(s)…").font(.caption).foregroundStyle(.secondary) }
                    Spacer()
                }
            } else if tab == "Lean" {
                LeanServerView(server: store.leanServer).onAppear {
                    do { try store.leanServer.configure(workspace: store.settings.workspaceRoot, resources: store.resourceDirectory) }
                    catch { store.error = error.localizedDescription }
                }
            } else if tab == "MCP" {
                Text("Choose which direct Codex MCP connections this app may start. None are enabled by default. CLI plugins are not loaded in research runs. Your global configuration is preserved.").font(.system(size: 13)).foregroundStyle(.secondary)
                ScrollView {
                    VStack(alignment: .leading, spacing: 16) {
                        ForEach(store.availableMCPServers, id: \.self) { name in
                            Toggle(name, isOn: Binding(get: { store.enabledMCPServers.contains(name) }, set: { if $0 { store.enabledMCPServers.insert(name) } else { store.enabledMCPServers.remove(name) } })).toggleStyle(.checkbox).disabled(store.hasActiveRuns)
                        }
                    }.frame(maxWidth: .infinity, alignment: .leading).padding(16)
                }
                Button("Refresh available servers") { Task { await store.refreshMCPServers() } }
            } else if tab == "Policy" {
                Text("These instructions guide both models. Record failures and expected behavior here to improve future runs.").font(.system(size: 13)).foregroundStyle(.secondary)
                TextEditor(text: $store.policy).font(.system(size: 12, design: .monospaced)).padding(8).background(Color.gray.opacity(0.05)).disabled(store.hasActiveRuns)
            } else {
                VStack(alignment: .leading, spacing: 16) {
                    Text("External SSD workspace").font(.headline)
                    Text(store.settings.workspaceRoot).font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                    Text("Projects, chat history, raw run logs, caches and temporary files stay here. The app stops new writes if the volume disappears. Your CLI keeps its own account configuration in its normal location.").font(.system(size: 13)).foregroundStyle(.secondary)
                    Button("Show workspace in Finder") { NSWorkspace.shared.open(URL(fileURLWithPath: store.settings.workspaceRoot)) }
                    Divider()
                    Text("Live research catalog").font(.headline)
                    TextField("HTTPS catalog URL", text: $store.settings.catalogURL).textFieldStyle(.roundedBorder).disabled(store.hasActiveRuns)
                    Text("The catalog is refreshed on launch and when you press Refresh. Your existing chats are preserved. New repositories can also be added manually.").font(.system(size: 13)).foregroundStyle(.secondary)
                }
                Spacer()
            }
            if store.hasActiveRuns { Text("Stop active research before changing settings.").font(.system(size: 12)).foregroundStyle(.secondary); Button("Close") { dismiss() } }
        }.padding(28).frame(width: 710, height: store.loginProvider == nil && !(tab == "Daytona" && !store.daytonaLoginOutput.isEmpty) ? 610 : 790)
    }
}

struct AddProjectView: View {
    @ObservedObject var store: AppStore
    @Environment(\.dismiss) private var dismiss
    @State private var repository = ""
    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text("Make room for a new idea.").font(.system(size: 24, weight: .semibold))
            Text("Add a GitHub repository or connect an existing project on the external SSD.").foregroundStyle(.secondary)
            TextField("https://github.com/owner/repository", text: $repository).textFieldStyle(.roundedBorder).onSubmit { store.addRepository(repository) }
            HStack {
                Button("Choose local folder…") { store.addFolder() }
                Spacer()
                Button("Cancel") { dismiss() }
                Button("Add repository") { store.addRepository(repository) }.keyboardShortcut(.defaultAction).disabled(repository.isEmpty)
            }
        }.padding(30).frame(width: 550)
    }
}
