import SwiftUI
import ResearchCore

struct SharedChatView: View {
    @ObservedObject var store: AppStore
    let projectID: String
    let groupID: UUID
    @State private var followOutput = true
    var body: some View {
        let entries = SharedTranscript.entries(in: store.sessions, projectID: projectID, groupID: groupID)
        VStack(spacing: 0) {
            ScrollViewReader { proxy in
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 24) {
                        if entries.isEmpty {
                            Text("Agent messages will appear here as they arrive.").font(.system(size: 13)).foregroundStyle(.secondary)
                        }
                        ForEach(entries) { entry in
                            MessageView(message: entry.message, provider: entry.provider,
                                        speaker: "Agent \(entry.agent.index) · \(entry.provider.title) · \(entry.agent.isIndependent ? "Independent" : "Round \(entry.agent.generation)")")
                                .id(entry.id)
                        }
                        Color.clear.frame(height: 1).id("latest")
                    }.frame(maxWidth: 760, alignment: .leading).padding(.horizontal, 34).padding(.vertical, 22).frame(maxWidth: .infinity)
                }
                .defaultScrollAnchor(.bottom)
                .onChange(of: entries.map { $0.message.text }) { _, _ in
                    if followOutput { proxy.scrollTo("latest", anchor: .bottom) }
                }
            }
            HStack {
                Toggle("Follow output", isOn: $followOutput).toggleStyle(.checkbox)
                Spacer()
                Text("Choose an agent below to send a follow-up.").foregroundStyle(.secondary)
            }.font(.system(size: 10)).padding(.horizontal, 28).padding(.vertical, 14)
        }
    }
}
