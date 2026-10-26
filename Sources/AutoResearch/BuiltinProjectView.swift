import AppKit
import SwiftUI
import ResearchCore

struct BuiltinProjectView: View {
    @ObservedObject var store: AppStore
    let project: ResearchProject
    @State private var showingGuide = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                Text(project.name).font(.system(size: 28, weight: .semibold))
                Text(project.summary).font(.system(size: 14)).foregroundStyle(.secondary)
                if let definition = store.builtinProject(project) {
                    VStack(alignment: .leading, spacing: 10) {
                        Label("No GPU rental required", systemImage: "laptopcomputer").font(.headline)
                        Text("Your local agent edits the public starter code. Open Kernel Challenge runs checks and scores on its GPUs using your account's quota. A verified leaderboard improvement is the goal; results are not guaranteed.")
                        Text("Start with your own OKC account and accept the rules in Terminal. The agent checks current limits before experiments. Confirm your account, choose your budgets, then press Play to see the agent work in chat. Public scored submissions stay off unless you enable them.")
                    }.font(.system(size: 13)).lineSpacing(4)
                        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 14))
                    KernelSetupView(store: store, project: project, definition: definition).id(project.id)
                    HStack(spacing: 16) {
                        Link("Challenge & leaderboard", destination: definition.website)
                        if definition.leaderboard != definition.website { Link("Leaderboard", destination: definition.leaderboard) }
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Choose a starting prompt").font(.headline)
                        Text("Choose one to fill the research prompt above, then press Play.").font(.system(size: 12)).foregroundStyle(.secondary)
                        ForEach(Array(definition.suggestedPrompts.enumerated()), id: \.offset) { _, prompt in
                            Button {
                                var draft = store.builtinDraft(for: project)
                                draft.researchPrompt = prompt
                                store.setBuiltinDraft(draft, for: project)
                            } label: {
                                HStack(alignment: .top, spacing: 12) {
                                    Text(prompt).multilineTextAlignment(.leading).frame(maxWidth: .infinity, alignment: .leading)
                                    Image(systemName: "arrow.up.right")
                                }.font(.system(size: 13)).padding(14).contentShape(Rectangle())
                            }.buttonStyle(.plain)
                                .background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 10))
                        }
                    }
                    DisclosureGroup("Bundled research guide", isExpanded: $showingGuide) {
                        if let instructions = try? definition.instructions(resources: store.resourceDirectory) {
                            Text(instructions).font(.system(size: 12, design: .monospaced)).textSelection(.enabled).padding(.top, 12)
                        }
                    }
                    Button("Edit workspace guide…") {
                        do {
                            let directory = try store.prepareBuiltinWorkspace(project)
                            NSWorkspace.shared.open(directory.appendingPathComponent("auto-research.md"))
                        } catch { store.error = error.localizedDescription }
                    }
                    Text("Each project keeps its own editable auto-research.md. Existing instructions, source files and results are preserved when you return.")
                        .font(.system(size: 12)).foregroundStyle(.secondary)
                } else {
                    Text("The bundled starter resources are unavailable. Reinstall the app to restore them; your saved workspace is preserved.")
                }
            }.frame(maxWidth: 800, alignment: .leading).padding(28).frame(maxWidth: .infinity, alignment: .leading)
        }
    }
}


private struct KernelSetupView: View {
    @ObservedObject var store: AppStore
    let project: ResearchProject
    let definition: BuiltinProject
    @State private var generatedPrompt = ""
    @State private var showingPreview = false
    @State private var showingLoginInstructions = false
    @State private var currentAccount: String?
    @State private var accountStatus = "Refresh to verify the account signed in on this Mac."
    @State private var refreshingAccount = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Set up a kernel run").font(.headline)
                Spacer()
                Button("Choose account…") { store.openOKCLogin(project, chooseAccount: true) }
                Button { store.play(project) } label: { Label("Play", systemImage: "play.fill") }
                    .buttonStyle(.borderedProminent).disabled(store.builtinPlayPending.contains(project.id))
            }
            Text("Choose account opens the official login in Terminal and your browser, even if this Mac is already signed in. Select your own account and review the rules yourself.")
                .font(.system(size: 12)).foregroundStyle(.secondary)
            HStack {
                Text(currentAccount.map { "Current OKC account: @" + $0 } ?? "Current OKC account: not verified")
                    .font(.system(size: 12, weight: .medium))
                Spacer()
                Button("Refresh account") { Task { await refreshAccount() } }.disabled(refreshingAccount)
                if refreshingAccount { ProgressView().controlSize(.small) }
            }
            Text(accountStatus).font(.system(size: 11)).foregroundStyle(.secondary)
            VStack(alignment: .leading, spacing: 6) {
                Text("Your expected GitHub/OKC username").font(.system(size: 12, weight: .medium))
                TextField("Enter the username you intend to use", text: field(\.expectedAccount))
                Text("This is an identity confirmation, not a password or token. Play verifies it against the signed-in account before starting.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
                if let currentAccount, !draft.expectedAccount.isEmpty,
                   OKCLogin.normalizeAccount(currentAccount) != OKCLogin.normalizeAccount(draft.expectedAccount) {
                    Text("The signed-in account differs from your expected username. Choose your account, then refresh.")
                        .font(.system(size: 11)).foregroundStyle(.red)
                }
            }
            DisclosureGroup("Installation and login instructions", isExpanded: $showingLoginInstructions) {
                if let text = try? definition.loginInstructions(resources: store.resourceDirectory) {
                    Text(text).font(.system(size: 12)).textSelection(.enabled).padding(.top, 8)
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Kernel source (optional)").font(.system(size: 12, weight: .medium))
                HStack {
                    TextField("Leave blank to use the public starter", text: field(\.sourcePath))
                    Button("Choose…") { chooseSource() }
                }
                Text("A path to your source file. The agent preserves the original and works on a copy.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Official check budget").font(.system(size: 12, weight: .medium))
                TextField("Enter a whole number; 0 for analysis only", text: field(\.checkBudget))
            }
            Toggle("Allow public scored submissions", isOn: field(\.allowScoredSubmissions))
            if draft.allowScoredSubmissions {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Scored submission budget").font(.system(size: 12, weight: .medium))
                    TextField("Enter a positive whole number", text: field(\.scoreBudget))
                    Text("Submission criterion").font(.system(size: 12, weight: .medium)).padding(.top, 4)
                    TextField("Describe your first-baseline or improvement requirement", text: field(\.scoreCriterion), axis: .vertical)
                        .lineLimit(2...5)
                    Text("Each score requires a clean official check of the exact source and current evidence meeting this criterion.")
                        .font(.system(size: 11)).foregroundStyle(.secondary)
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                HStack {
                    Text("Research prompt").font(.system(size: 12, weight: .medium))
                    Spacer()
                    Menu("Use a starter prompt") {
                        ForEach(Array(definition.suggestedPrompts.enumerated()), id: \.offset) { _, prompt in
                            Button(prompt) { field(\.researchPrompt).wrappedValue = prompt }
                        }
                    }
                }
                TextEditor(text: field(\.researchPrompt)).font(.system(size: 13)).frame(minHeight: 84)
                    .padding(6).background(.background, in: RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.2)))
            }
            HStack {
                Button("Review request…") { preparePrompt() }.buttonStyle(.bordered)
                if store.builtinPlayPending.contains(project.id) {
                    ProgressView().controlSize(.small)
                    Text("Verifying account…").font(.system(size: 12)).foregroundStyle(.secondary)
                }
            }
            Text("Play uses this saved setup and opens a chat with live progress. Review request lets you inspect or edit the generated prompt first. Login and preparation spend no judge quota.")
                .font(.system(size: 12)).foregroundStyle(.secondary)
        }
        .textFieldStyle(.roundedBorder)
        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 14))
        .sheet(isPresented: $showingPreview) {
            VStack(alignment: .leading, spacing: 16) {
                Text("Review your kernel request").font(.title2.bold())
                Text("Edit this prompt as needed. Use in chat copies it to the composer; send it there when ready.")
                    .foregroundStyle(.secondary)
                TextEditor(text: $generatedPrompt).font(.system(size: 12, design: .monospaced))
                    .frame(minHeight: 300).border(Color.secondary.opacity(0.2))
                HStack {
                    Button("Back") { showingPreview = false }
                    Spacer()
                    Button("Use in chat") {
                        store.select(project)
                        store.newChat()
                        store.composer = generatedPrompt
                        showingPreview = false
                    }.buttonStyle(.borderedProminent).disabled(generatedPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }
            }.padding(24).frame(minWidth: 620, idealWidth: 760, minHeight: 500)
        }
    }

    private var draft: KernelResearchDraft { store.builtinDraft(for: project) }

    private func field<Value>(_ keyPath: WritableKeyPath<KernelResearchDraft, Value>) -> Binding<Value> {
        Binding(get: { store.builtinDraft(for: project)[keyPath: keyPath] }, set: { value in
            var draft = store.builtinDraft(for: project)
            draft[keyPath: keyPath] = value
            store.setBuiltinDraft(draft, for: project)
        })
    }

    private func refreshAccount() async {
        refreshingAccount = true
        defer { refreshingAccount = false }
        do {
            currentAccount = try await store.refreshOKCAccount()
            accountStatus = "Account verified with okc whoami. Enter your expected username before Play."
        } catch {
            currentAccount = nil
            accountStatus = error.localizedDescription
        }
    }

    private func preparePrompt() {
        do {
            let input = try draft.validatedSetup()
            generatedPrompt = try definition.setupPrompt(input, resources: store.resourceDirectory)
            showingPreview = true
        } catch { store.error = error.localizedDescription }
    }

    private func chooseSource() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose your kernel source. The agent will preserve this file."
        if panel.runModal() == .OK, let url = panel.url { field(\.sourcePath).wrappedValue = url.path }
    }
}
