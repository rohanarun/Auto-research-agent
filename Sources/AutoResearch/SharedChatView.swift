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
                .onAppear { scrollToResponse(using: proxy, entryIDs: Set(entries.map(\.id))) }
                .onChange(of: store.responseFocusRequest) { _, _ in
                    scrollToResponse(using: proxy, entryIDs: Set(entries.map(\.id)))
                }
            }
            HStack {
                Toggle("Follow output", isOn: $followOutput).toggleStyle(.checkbox)
                Spacer()
                Text("Choose a reply above or an agent below.").foregroundStyle(.secondary)
            }.font(.system(size: 10)).padding(.horizontal, 28).padding(.vertical, 14)
        }
    }

    private func scrollToResponse(using proxy: ScrollViewProxy, entryIDs: Set<String>) {
        guard let target = store.responseScrollTarget, entryIDs.contains(target) else { return }
        followOutput = false
        proxy.scrollTo(target, anchor: .top)
    }
}

struct FleetReplyNavigator: View {
    @ObservedObject var store: AppStore

    var body: some View {
        VStack(alignment: .leading, spacing: 9) {
            HStack {
                Text(recipientTitle).font(.system(size: 12, weight: .semibold)).lineLimit(1)
                Spacer()
                Toggle("Follow replies", isOn: Binding(get: { store.followResponses }, set: { store.followResponses = $0 }))
                    .toggleStyle(.checkbox).font(.system(size: 11))
                    .help("Open the latest completed agent reply when your current draft is empty. Turn off to browse without automatic switching.")
            }
            HStack(spacing: 12) {
                Button { store.previousResponse() } label: { Label("Previous reply", systemImage: "chevron.left") }
                    .disabled((store.responsePosition ?? 0) <= 0)
                Button { store.nextResponse() } label: { Label("Next reply", systemImage: "chevron.right") }
                    .disabled(store.responseEntries.isEmpty || (store.responsePosition ?? -1) >= store.responseEntries.count - 1)
                Text(positionTitle).foregroundStyle(.secondary).monospacedDigit()
                Spacer(minLength: 4)
                if store.pendingResponseCount > 0 {
                    Button("\(store.pendingResponseCount) waiting") { store.nextPendingResponse() }
                        .help("Open the next unread completed reply")
                }
                Menu("Reply history") {
                    if store.responseEntries.isEmpty { Text("No completed replies yet") }
                    ForEach(store.responseEntries.reversed()) { reply in
                        Button(historyTitle(reply)) { store.selectResponse(reply.id) }
                    }
                }.menuStyle(.borderlessButton).fixedSize()
            }.font(.system(size: 11)).buttonStyle(.borderless)
        }.padding(.horizontal, 24).padding(.vertical, 12)
            .background(Color.black.opacity(0.018))
    }

    private var recipientTitle: String {
        guard let recipient = store.responseRecipient, let agent = recipient.sharedAgent else { return "Agent replies" }
        return "Replying to Agent \(agent.index) · \(recipient.provider.title)"
    }

    private var positionTitle: String {
        if let position = store.responsePosition { return "Reply \(position + 1) of \(store.responseEntries.count)" }
        return "\(store.responseEntries.count) replies"
    }

    private func historyTitle(_ reply: FleetReply) -> String {
        let session = store.sessions.first { $0.id == reply.sessionID }
        let index = session?.sharedAgent?.index.description ?? "?"
        let text = session?.messages.first { $0.id == reply.messageID }?.text
            .split(whereSeparator: \.isNewline).first.map(String.init) ?? "Completed response"
        return "Agent \(index) · \(reply.date.formatted(date: .omitted, time: .shortened)) · \(text.prefix(80))"
    }
}
