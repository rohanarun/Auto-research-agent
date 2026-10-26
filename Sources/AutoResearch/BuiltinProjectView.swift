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
                        Text("Start with your own OKC account and accept the rules in Terminal. The agent checks current limits before experiments. Public scored submissions require your instruction; starting with the guide alone does not publish a score.")
                    }.font(.system(size: 13)).lineSpacing(4)
                        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 14))
                    KernelSetupView(store: store, project: project, definition: definition).id(project.id)
                    HStack(spacing: 16) {
                        Button { store.play(project) } label: { Label("Start with guide only", systemImage: "play.fill") }
                            .buttonStyle(.bordered)
                            .help("Starts a guide discussion without using the setup fields above.")
                        Link("Challenge & leaderboard", destination: definition.website)
                        if definition.leaderboard != definition.website { Link("Leaderboard", destination: definition.leaderboard) }
                    }
                    VStack(alignment: .leading, spacing: 12) {
                        Text("Choose a starting prompt").font(.headline)
                        Text("Review or edit the prompt in chat before sending.").font(.system(size: 12)).foregroundStyle(.secondary)
                        ForEach(Array(definition.suggestedPrompts.enumerated()), id: \.offset) { _, prompt in
                            Button {
                                store.newChat()
                                store.composer = prompt
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
    @State private var sourcePath = ""
    @State private var checkBudget = ""
    @State private var allowScores = false
    @State private var scoreBudget = ""
    @State private var scoreCriterion = ""
    @State private var researchPrompt = ""
    @State private var generatedPrompt = ""
    @State private var showingPreview = false
    @State private var showingLoginInstructions = false

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Text("Set up a kernel run").font(.headline)
                Spacer()
                Button("Sign in to OKC…") { store.openOKCLogin(project) }
            }
            Text("Login happens in Terminal. Verify the active account there, then return to prepare your request.")
                .font(.system(size: 12)).foregroundStyle(.secondary)
            DisclosureGroup("Installation and login instructions", isExpanded: $showingLoginInstructions) {
                if let text = try? definition.loginInstructions(resources: store.resourceDirectory) {
                    Text(text).font(.system(size: 12)).textSelection(.enabled).padding(.top, 8)
                }
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Kernel source (optional)").font(.system(size: 12, weight: .medium))
                HStack {
                    TextField("Leave blank to use the public starter", text: $sourcePath)
                    Button("Choose…") { chooseSource() }
                }
                Text("A path to your source file. The agent preserves the original and works on a copy.")
                    .font(.system(size: 11)).foregroundStyle(.secondary)
            }
            VStack(alignment: .leading, spacing: 6) {
                Text("Official check budget").font(.system(size: 12, weight: .medium))
                TextField("Enter a whole number; 0 for analysis only", text: $checkBudget)
            }
            Toggle("Allow public scored submissions", isOn: $allowScores)
            if allowScores {
                VStack(alignment: .leading, spacing: 6) {
                    Text("Scored submission budget").font(.system(size: 12, weight: .medium))
                    TextField("Enter a positive whole number", text: $scoreBudget)
                    Text("Submission criterion").font(.system(size: 12, weight: .medium)).padding(.top, 4)
                    TextField("Describe your first-baseline or improvement requirement", text: $scoreCriterion, axis: .vertical)
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
                            Button(prompt) { researchPrompt = prompt }
                        }
                    }
                }
                TextEditor(text: $researchPrompt).font(.system(size: 13)).frame(minHeight: 84)
                    .padding(6).background(.background, in: RoundedRectangle(cornerRadius: 6))
                    .overlay(RoundedRectangle(cornerRadius: 6).stroke(Color.secondary.opacity(0.2)))
            }
            Button("Review request") { preparePrompt() }.buttonStyle(.borderedProminent)
            Text("You review the generated prompt before sending it. Preparation and login spend no judge quota.")
                .font(.system(size: 12)).foregroundStyle(.secondary)
        }
        .textFieldStyle(.roundedBorder)
        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 14))
        .onAppear { if researchPrompt.isEmpty { researchPrompt = definition.suggestedPrompts.first ?? "" } }
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

    private func preparePrompt() {
        do {
            let input = try KernelResearchSetup(sourcePath: sourcePath, checkBudget: checkBudget,
                allowScoredSubmissions: allowScores, scoreBudget: scoreBudget,
                researchPrompt: researchPrompt, scoreCriterion: scoreCriterion)
            generatedPrompt = try definition.setupPrompt(input, resources: store.resourceDirectory)
            showingPreview = true
        } catch { store.error = error.localizedDescription }
    }

    private func chooseSource() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.message = "Choose your kernel source. The agent will preserve this file."
        if panel.runModal() == .OK, let url = panel.url { sourcePath = url.path }
    }
}
