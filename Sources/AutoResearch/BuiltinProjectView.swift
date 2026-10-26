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
                        Text("Start with your own OKC account and accept the rules in Terminal. The agent checks current limits before experiments. Public scored submissions require your instruction; Start research alone does not publish a score.")
                    }.font(.system(size: 13)).lineSpacing(4)
                        .padding(20).frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 14))
                    HStack(spacing: 16) {
                        Button { store.play(project) } label: { Label("Start research", systemImage: "play.fill") }
                            .buttonStyle(.borderedProminent)
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
