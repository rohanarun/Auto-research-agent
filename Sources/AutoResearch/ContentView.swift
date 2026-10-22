import SwiftUI
import AppKit
import ResearchCore

private let ink = Color(red: 0.13, green: 0.14, blue: 0.15)
private let muted = Color(red: 0.46, green: 0.47, blue: 0.48)
private let sidebarColor = Color(red: 0.955, green: 0.954, blue: 0.946)
private let accent = Color(red: 0.30, green: 0.34, blue: 0.78)

struct ContentView: View {
    @ObservedObject var store: AppStore
    @State private var followOutput = true
    @State private var recipientIndex = 1
    var body: some View {
        HStack(spacing: 0) {
            sidebar.frame(width: 282)
            Rectangle().fill(Color.black.opacity(0.075)).frame(width: 1)
            VStack(spacing: 0) {
                header
                Divider().opacity(0.4)
                if let groupID = store.selectedSharedGroupID, let project = store.project {
                    SharedChatView(store: store, projectID: project.id, groupID: groupID).id(groupID)
                    composer
                } else if let session = store.session {
                    if let task = store.researchSubtasks.first(where: { $0.id == session.researchSubtaskID }) {
                        ResearchSubtaskOverviewView(task: task, session: session).id(session.id)
                    }
                    transcript(session)
                    if store.isCPUSubtask { cpuScriptControls } else { composer }
                } else if store.showingProjectDetails, let project = store.project {
                    ProjectDashboardView(project: project, catalogURL: store.settings.catalogURL)
                } else { welcome; composer }
            }.frame(maxWidth: .infinity, maxHeight: .infinity).background(.white)
        }
        .foregroundStyle(ink)
        .sheet(isPresented: $store.showSettings) { ConnectionsView(store: store) }
        .sheet(isPresented: $store.showAddProject) { AddProjectView(store: store) }
        .alert("Auto Research", isPresented: Binding(get: { store.error != nil }, set: { if !$0 { store.error = nil } })) {
            Button("OK") { store.error = nil }
        } message: { Text(store.error ?? "") }
        .onReceive(NotificationCenter.default.publisher(for: NSApplication.didBecomeActiveNotification)) { _ in Task { await store.refreshAuth() } }
    }
    var sidebar: some View {
        VStack(alignment: .leading, spacing: 0) {
            HStack(spacing: 9) {
                Image(systemName: "circle.hexagongrid.fill").foregroundStyle(accent).font(.system(size: 23))
                Text("Auto Research").font(.system(size: 16, weight: .semibold))
                Spacer()
                Button { store.newChat() } label: { Image(systemName: "square.and.pencil") }.buttonStyle(.plain).help("New chat (⌘N)")
            }.padding(.top, 42).padding(.horizontal, 20).padding(.bottom, 24)
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass").foregroundStyle(muted)
                TextField("Find a project", text: $store.search).textFieldStyle(.plain)
                if !store.search.isEmpty { Button { store.search = "" } label: { Image(systemName: "xmark.circle.fill") }.buttonStyle(.plain).foregroundStyle(muted) }
            }.padding(9).background(.white.opacity(0.75), in: RoundedRectangle(cornerRadius: 9)).padding(.horizontal, 14).padding(.bottom, 20)
            HStack {
                Text("Open research").font(.system(size: 12, weight: .semibold)).foregroundStyle(muted)
                Spacer()
                Button { Task { await store.refreshCatalog() } } label: { Image(systemName: "arrow.clockwise").rotationEffect(.degrees(store.refreshing ? 180 : 0)).animation(.easeInOut, value: store.refreshing) }.buttonStyle(.plain).disabled(store.refreshing).help("Refresh live catalog")
                Button { store.showAddProject = true } label: { Image(systemName: "plus") }.buttonStyle(.plain).help("Add a project")
            }.padding(.horizontal, 20).padding(.bottom, 9)
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 4) {
                    ForEach(store.visibleProjects.filter { !$0.isCustom }) { project in projectRow(project) }
                    if !store.visibleProjects.filter({ $0.isCustom }).isEmpty {
                        Text("Your projects").font(.system(size: 12, weight: .semibold)).foregroundStyle(muted).padding(.top, 22).padding(.bottom, 6).padding(.leading, 8)
                        ForEach(store.visibleProjects.filter { $0.isCustom }) { project in projectRow(project) }
                    }
                    if !store.projectSessions.isEmpty {
                        Text("Chats in this project").font(.system(size: 12, weight: .semibold)).foregroundStyle(muted).padding(.top, 22).padding(.bottom, 6).padding(.leading, 8)
                        ForEach(store.projectSharedGroups) { representative in
                            sharedGroupRow(representative)
                        }
                        ForEach(store.projectSessions) { session in
                            chatRow(session)
                        }
                    }
                    if !store.closedProjectSessions.isEmpty {
                        Menu {
                            ForEach(store.closedProjectSessions) { session in
                                Button("\(session.title) · \(session.createdAt.formatted(date: .abbreviated, time: .shortened))") { store.selectSession(session) }
                            }
                        } label: {
                            Label("Closed chats (\(store.closedProjectSessions.count))", systemImage: "archivebox")
                                .font(.system(size: 12)).foregroundStyle(muted)
                        }.menuStyle(.borderlessButton).padding(.horizontal, 8).padding(.top, 12)
                    }
                    if store.visibleProjects.isEmpty { Text("No matching projects").font(.system(size: 12)).foregroundStyle(muted).padding() }
                }.padding(.horizontal, 12)
            }
            VStack(alignment: .leading, spacing: 13) {
                Text(store.catalogStatus).font(.system(size: 10)).foregroundStyle(muted)
                Button { store.showSettings = true } label: {
                    HStack(spacing: 10) {
                        Image(systemName: "slider.horizontal.3")
                        Text("Connections & settings").font(.system(size: 12, weight: .medium))
                        Spacer()
                        Circle().fill(store.auth[store.provider] == "Connected" ? Color.green : Color.orange).frame(width: 6, height: 6)
                    }
                }.buttonStyle(.plain)
            }.padding(20).background(sidebarColor)
        }.background(sidebarColor)
    }
    func projectRow(_ project: ResearchProject) -> some View {
        VStack(alignment: .leading, spacing: 2) {
          HStack(spacing: 6) {
            Button { withAnimation(.easeOut(duration: 0.16)) { store.select(project) } } label: {
                HStack(spacing: 9) {
                    Image(systemName: project.isCustom ? "folder" : "circle.dotted").font(.system(size: 14)).foregroundStyle(store.selectedProjectID == project.id ? accent : muted)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(project.name.replacingOccurrences(of: "-", with: " ")).font(.system(size: 12, weight: store.selectedProjectID == project.id ? .semibold : .regular)).lineLimit(1)
                        Text(project.repository.isEmpty ? "Local workspace" : project.repository.components(separatedBy: "/").first ?? "").font(.system(size: 10)).foregroundStyle(muted).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }.contentShape(Rectangle())
            }.buttonStyle(.plain)
            Button { store.play(project) } label: {
                Image(systemName: store.running(project) ? "waveform" : "play.fill").font(.system(size: 10, weight: .semibold)).foregroundStyle(store.running(project) ? accent : ink).frame(width: 25, height: 28).contentShape(Rectangle())
            }.buttonStyle(.plain).help("Start research with \(store.provider.title)").accessibilityLabel("Start research for \(project.name)")
        }.padding(.vertical, 8).padding(.leading, 9).padding(.trailing, 3)
            .background(store.selectedProjectID == project.id ? Color.black.opacity(0.055) : .clear, in: RoundedRectangle(cornerRadius: 8))
            .contextMenu {
                Button("Start research") { store.play(project) }
                if let url = project.githubURL { Link("View on GitHub", destination: url) }
            }
          if store.selectedProjectID == project.id {
              ForEach(store.researchSubtasks) { task in
                  subtaskRow(task, project: project)
              }
              .transition(.opacity.combined(with: .move(edge: .top)))
          }
        }
    }
    func subtaskRow(_ task: ResearchSubtask, project: ResearchProject) -> some View {
        let session = store.subtaskSession(task, project: project)
        let active = session.map { store.runners[$0.id] != nil } ?? false
        return HStack(spacing: 4) {
            Rectangle().fill(muted.opacity(0.22)).frame(width: 1)
            Button {
                if let session { store.selectSession(session) }
                else { store.playSubtask(task, project: project) }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: task.symbol).font(.system(size: 12)).foregroundStyle(active ? accent : muted)
                    VStack(alignment: .leading, spacing: 3) {
                        Text(task.title).font(.system(size: 11, weight: .medium))
                        Text(session.map { $0.state == "Completed" ? "Run finished · view findings" : $0.state } ?? (task.execution == "script" ? "Script · No model" : "Not run · \(store.provider.title)"))
                            .font(.system(size: 10)).foregroundStyle(muted).lineLimit(1)
                    }
                    Spacer(minLength: 0)
                }.padding(.leading, 7).padding(.vertical, 7).contentShape(Rectangle())
            }.buttonStyle(.plain).disabled(session == nil && store.running(project))
                .help(task.summary).accessibilityLabel("\(task.title) for \(project.name)")
            Button {
                if active, let session { store.stop(session.id) }
                else { store.playSubtask(task, project: project) }
            } label: {
                Image(systemName: active ? "stop.fill" : "play.fill")
                    .font(.system(size: 9, weight: .semibold)).foregroundStyle(active ? accent : ink)
                    .frame(width: 25, height: 30).contentShape(Rectangle())
            }.buttonStyle(.plain).disabled(!active && store.running(project))
                .help(active ? "Stop this subtask" : store.running(project) ? "Stop active research before starting this subtask" : task.execution == "script" ? "Run the configured Python script; no model" : "Start \(task.title.lowercased()) with \(store.provider.title)")
                .accessibilityLabel("\(active ? "Stop" : "Start") \(task.title.lowercased()) for \(project.name)")
        }.padding(.leading, 27).padding(.trailing, 3)
            .background(session != nil && store.selectedSessionID == session?.id ? accent.opacity(0.07) : .clear, in: RoundedRectangle(cornerRadius: 7))
    }
    func chatRow(_ session: ResearchSession) -> some View {
        HStack(spacing: 2) {
            Button { store.selectSession(session) } label: {
                HStack(spacing: 9) {
                    Image(systemName: "bubble.left").font(.system(size: 12))
                    Text(session.title).lineLimit(1).font(.system(size: 12))
                    Spacer(minLength: 0)
                    if session.state == "Running" { Circle().fill(accent).frame(width: 5, height: 5) }
                }.padding(9).contentShape(Rectangle())
            }.buttonStyle(.plain)
            Button { store.closeSession(session.id) } label: { Image(systemName: "xmark").font(.system(size: 10)).frame(width: 26, height: 28) }
                .buttonStyle(.plain).foregroundStyle(muted).disabled(!store.canClose(session))
                .help(store.canClose(session) ? "Close chat; keep history" : "Stop research before closing this chat")
                .accessibilityLabel("Close chat: \(session.title)")
        }.background(store.selectedSessionID == session.id ? Color.black.opacity(0.055) : .clear, in: RoundedRectangle(cornerRadius: 7))
    }
    func sharedGroupRow(_ session: ResearchSession) -> some View {
        let group = session.sharedAgent!.groupID
        return HStack(spacing: 2) {
            Button { store.selectSharedGroup(group, projectID: session.projectID) } label: {
                Label("All agents · \(session.createdAt.formatted(date: .abbreviated, time: .shortened))", systemImage: "person.2")
                    .font(.system(size: 12)).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading).padding(9).contentShape(Rectangle())
            }.buttonStyle(.plain)
            Button { store.closeSharedGroup(group) } label: { Image(systemName: "xmark").font(.system(size: 10)).frame(width: 26, height: 28) }
                .buttonStyle(.plain).foregroundStyle(muted).disabled(!store.canCloseGroup(group))
                .help("Close this group's chats; keep history").accessibilityLabel("Close parallel chats")
        }.background(store.selectedSharedGroupID == group ? Color.black.opacity(0.055) : .clear, in: RoundedRectangle(cornerRadius: 7))
    }
    var header: some View {
        HStack(spacing: 12) {
            Text(store.project?.name.replacingOccurrences(of: "-", with: " ") ?? "Your research workspace")
                .font(.system(size: 13, weight: .semibold)).lineLimit(1)
                .truncationMode(.tail).frame(maxWidth: .infinity, alignment: .leading)
            if let project = store.project {
                if store.session?.scripted != true { SharedContextControls(store: store, project: project).fixedSize() }
                if store.showingProjectDetails && store.session == nil && store.selectedSharedGroupID == nil {
                    Button { store.play(project) } label: { Image(systemName: "play.fill") }
                        .help("Start research").accessibilityLabel("Start selected research project")
                } else if store.selectedSharedGroupID == nil && store.session?.sharedAgent == nil {
                    if store.isRunning { ProgressView().controlSize(.mini) }
                    Text(store.session?.state ?? "Ready").foregroundStyle(muted).lineLimit(1)
                    if store.isRunning {
                        Button { if let id = store.selectedSessionID { store.stop(id) } } label: { Image(systemName: "stop.fill") }
                            .help("Stop research").accessibilityLabel("Stop research")
                    }
                }
            }
            if store.session?.scripted == true {
                Label("CPU script · No model", systemImage: "cpu").foregroundStyle(muted)
            } else { Menu {
                ForEach(Provider.allCases) { provider in Button(provider.title) { store.provider = provider; store.newChat() } }
            } label: {
                HStack(spacing: 5) {
                    Circle().fill(store.auth[store.session?.provider ?? store.provider] == "Connected" ? Color.green : .orange).frame(width: 5, height: 5)
                    Text((store.session?.provider ?? store.provider).title)
                }
            }.menuStyle(.borderlessButton).fixedSize() }
            Button { store.newChat() } label: { Image(systemName: "square.and.pencil") }
                .help("New chat").accessibilityLabel("New chat")
            Menu {
                if let project = store.project {
                    Button("Project details") { store.showProjectDetails() }
                    Button("Start research") { store.play(project) }
                    if let recent = store.projectSessions.first { Button("Open latest chat") { store.selectSession(recent) } }
                    Divider()
                }
                if let url = store.project?.githubURL { Link("View repository", destination: url) }
                Button("Show project in Finder") { store.revealProject() }.disabled(store.session?.workingDirectory == nil && store.project?.localPath == nil)
                Button("Show run logs") { store.revealLogs() }.disabled(store.session == nil)
                Button("Continue in Terminal") { store.openTerminal() }.disabled(store.isRunning || store.sharedActive(store.project?.id ?? "") || store.session?.providerSessionID == nil)
                if let session = store.session {
                    Button("Close chat") { store.closeSession(session.id) }.disabled(!store.canClose(session))
                } else if let group = store.selectedSharedGroupID {
                    Button("Close parallel chats") { store.closeSharedGroup(group) }.disabled(!store.canCloseGroup(group))
                }
                Button("Connections & settings") { store.showSettings = true }
            } label: { Image(systemName: "ellipsis") }.menuStyle(.borderlessButton).fixedSize()
        }
        .font(.system(size: 11)).buttonStyle(.plain).controlSize(.small)
        .padding(.horizontal, 20).padding(.vertical, 12)
        .frame(height: 48)
    }
    var welcome: some View {
        VStack(alignment: .leading, spacing: 0) {
            Spacer()
            HStack(spacing: 6) { Image(systemName: "sparkle"); Text("A little curiosity. A lot of possibility.") }.font(.system(size: 12)).foregroundStyle(muted).padding(.bottom, 22)
            Text("Where will your\nresearch lead?").font(.system(size: 44, weight: .medium, design: .rounded)).tracking(-1.8).lineSpacing(-1).fixedSize(horizontal: false, vertical: true).padding(.bottom, 20)
            Text(store.project?.summary.isEmpty == false ? store.project!.summary : "Give an idea room to grow. Your agent can explore the code, run experiments, and keep the evidence together.")
                .font(.system(size: 15)).foregroundStyle(muted).lineSpacing(5).frame(maxWidth: 540, alignment: .leading)
            HStack(spacing: 12) {
                Button { if let project = store.project { store.play(project) } } label: { Label("Start research", systemImage: "play.fill").font(.system(size: 13, weight: .medium)).padding(.horizontal, 18).padding(.vertical, 12).background(ink, in: Capsule()).foregroundStyle(.white) }.buttonStyle(.plain).disabled(store.project == nil)
                if let url = store.project?.githubURL { Link(destination: url) { Label("Explore repository", systemImage: "arrow.up.right").font(.system(size: 13, weight: .medium)).padding(.horizontal, 16).padding(.vertical, 12).background(sidebarColor, in: Capsule()).foregroundStyle(ink) } }
            }.padding(.top, 28)
            HStack(alignment: .top, spacing: 22) {
                hint("doc.text", "Repository-led", "Starts from its research guide")
                hint("terminal", "Your CLI, your account", "Codex, Claude or OpenCode underneath")
                hint("externaldrive", "A durable workspace", "Chats, code, and experiment logs")
            }.padding(.top, 58)
            Spacer()
        }.frame(maxWidth: 690, alignment: .leading).padding(.horizontal, 36).frame(maxWidth: .infinity)
    }
    func hint(_ symbol: String, _ title: String, _ detail: String) -> some View {
        VStack(alignment: .leading, spacing: 9) {
            Image(systemName: symbol).font(.system(size: 17)).foregroundStyle(muted).padding(.bottom, 3)
            Text(title).font(.system(size: 11, weight: .medium))
            Text(detail).font(.system(size: 10)).foregroundStyle(muted).lineSpacing(3)
        }.frame(maxWidth: .infinity, alignment: .leading)
    }
    func transcript(_ session: ResearchSession) -> some View {
        ScrollViewReader { proxy in
            VStack(spacing: 0) {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 24) {
                        ForEach(session.messages) { message in MessageView(message: message, provider: session.provider, speaker: session.scripted == true ? "CPU script" : session.sharedAgent.map { "Agent \($0.index) · \(session.provider.title) · \($0.isIndependent ? "Independent" : "Round \($0.generation)")" }) }
                        if store.isRunning { HStack(spacing: 9) { ProgressView().controlSize(.small); Text(session.state == "Preparing" ? "Preparing workspace…" : session.scripted == true ? "Running script…" : "Researching…").font(.system(size: 12)).foregroundStyle(muted) }.padding(.vertical, 12) }
                        Color.clear.frame(height: 1).id("bottom")
                    }.frame(maxWidth: 760, alignment: .leading).padding(.horizontal, 34).padding(.vertical, 20).frame(maxWidth: .infinity)
                }
                .onChange(of: session.messages.count) { _, _ in if followOutput { withAnimation(.easeOut(duration: 0.18)) { proxy.scrollTo("bottom", anchor: .bottom) } } }
                .onChange(of: session.id) { _, _ in proxy.scrollTo("bottom", anchor: .bottom) }
                HStack {
                    Toggle("Follow output", isOn: $followOutput).toggleStyle(.checkbox)
                    Spacer()
                    if !store.isCPUSubtask && (session.sharedAgent == nil || store.sharedConfiguration(session.projectID).usesMethods) {
                        Toggle("Keep researching", isOn: Binding(get: { store.keepResearching.contains(session.id) }, set: { if $0 { store.keepResearching.insert(session.id) } else { store.keepResearching.remove(session.id) } })).toggleStyle(.checkbox).help("Send the editable continuation prompt after each successful turn until you stop. Uses your CLI account and configured resources.")
                    }
                }.font(.system(size: 10)).foregroundStyle(muted).padding(.horizontal, 34).padding(.vertical, 8)
            }
        }
    }
    var composerAgents: [ResearchSession] {
        guard let group = store.selectedSharedGroupID, let project = store.project else { return [] }
        return SharedTranscript.latestAgents(in: store.sessions, projectID: project.id, groupID: group)
    }
    var cpuScriptControls: some View {
        HStack(spacing: 14) {
            Text("Runs the saved Python script and parameter plan. No model calls.")
                .font(.system(size: 11)).foregroundStyle(muted)
            Spacer()
            Button("Edit search plan") { store.revealCPUSearchPlan() }.disabled(store.isRunning)
            if store.isRunning, let id = store.selectedSessionID {
                Button("Stop script") { store.stop(id) }
            } else if let project = store.project, let task = store.researchSubtasks.first(where: { $0.id == store.session?.researchSubtaskID }) {
                Button("Run script") { store.playSubtask(task, project: project) }.disabled(store.running(project))
            }
        }.controlSize(.small).padding(24)
    }
    var composerRecipient: ResearchSession? {
        composerAgents.first { $0.sharedAgent?.index == recipientIndex } ?? composerAgents.first
    }
    var composerBusy: Bool {
        store.sharedActive(store.project?.id ?? "") || store.sessions.contains { $0.projectID == store.selectedProjectID && store.runners[$0.id] != nil }
    }
    var canSendComposer: Bool {
        !composerBusy && store.project != nil && !store.composer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty && (store.selectedSharedGroupID == nil || composerRecipient != nil)
    }
    func sendComposer() {
        guard canSendComposer else { return }
        store.send(to: store.selectedSharedGroupID == nil ? nil : composerRecipient?.id)
    }
    var composer: some View {
        VStack(spacing: 10) {
            VStack(alignment: .leading, spacing: 10) {
                if store.selectedSharedGroupID != nil {
                    Picker("Send to", selection: Binding(get: { composerRecipient?.sharedAgent?.index ?? recipientIndex }, set: { recipientIndex = $0 })) {
                        ForEach(composerAgents) { session in
                            Text("Agent \(session.sharedAgent!.index) · \(session.provider.title)").tag(session.sharedAgent!.index)
                        }
                    }.font(.system(size: 12)).fixedSize().accessibilityLabel("Message recipient")
                }
                ZStack(alignment: .topLeading) {
                    if store.composer.isEmpty { Text(composerBusy ? "Draft a follow-up while research runs…" : "Ask a question, or give your research a direction…").font(.system(size: 14)).foregroundStyle(muted).padding(.top, 8).padding(.leading, 5).allowsHitTesting(false) }
                    ChatInput(text: $store.composer, isEditable: true, onSend: sendComposer)
                        .frame(minHeight: 56, maxHeight: 94)
                }
                HStack(spacing: 8) {
                    Image(systemName: "folder").font(.system(size: 11)).foregroundStyle(muted)
                    Text(store.project?.repository.isEmpty == false ? store.project!.repository : store.project?.name ?? "Select a project").font(.system(size: 11)).foregroundStyle(muted).lineLimit(1)
                    Spacer()
                    if store.sharedActive(store.project?.id ?? ""), let project = store.project {
                        Button("Stop agents") { store.stopSharedResearch(project.id) }.buttonStyle(.bordered).help("Stop the group before sending your draft")
                    } else if let running = store.sessions.first(where: { $0.projectID == store.selectedProjectID && store.runners[$0.id] != nil }) {
                        let id = running.id
                        Button { store.stop(id) } label: { Image(systemName: "stop.fill").font(.system(size: 11)).frame(width: 31, height: 31).background(ink, in: Circle()).foregroundStyle(.white) }.buttonStyle(.plain).help("Stop this run").accessibilityLabel("Stop research")
                    } else {
                        Button(action: sendComposer) { Image(systemName: "arrow.up").font(.system(size: 14, weight: .semibold)).frame(width: 31, height: 31).background(store.composer.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? Color.gray.opacity(0.2) : ink, in: Circle()).foregroundStyle(store.composer.isEmpty ? muted : .white) }.buttonStyle(.plain).keyboardShortcut(.return, modifiers: .command).disabled(!canSendComposer).help("Enter to send · Shift+Enter for a new line").accessibilityLabel("Send message")
                    }
                }
            }.padding(14).background(Color.white, in: RoundedRectangle(cornerRadius: 20)).overlay(RoundedRectangle(cornerRadius: 20).stroke(Color.black.opacity(0.13), lineWidth: 1)).shadow(color: .black.opacity(0.035), radius: 10, y: 3)
            if composerBusy { Text("Stop research to send your draft.").font(.system(size: 11)).foregroundStyle(muted) }
            Text("\((composerRecipient?.provider ?? store.session?.provider ?? store.provider).title) · \(store.auth[composerRecipient?.provider ?? store.session?.provider ?? store.provider] ?? "Checking connection") · Uses your CLI’s configured model")
                .font(.system(size: 10)).foregroundStyle(muted)
        }.frame(maxWidth: 800).padding(.horizontal, 32).padding(.bottom, 20).padding(.top, 12).frame(maxWidth: .infinity)
    }
}

struct MessageView: View {
    let message: ChatMessage
    let provider: Provider
    var speaker: String? = nil
    private var displayedSpeaker: String? { message.speaker ?? speaker }
    @State private var expanded = false
    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let speaker = displayedSpeaker, message.role == "tool" || message.role == "status" {
                Text(speaker).font(.system(size: 12, weight: .semibold)).foregroundStyle(ink)
            }
            content
        }
    }
    @ViewBuilder var content: some View {
        if message.role == "tool" {
            DisclosureGroup(isExpanded: $expanded) {
                Text(message.text).font(.system(size: 11, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading).padding(.top, 8)
            } label: {
                Label(String(message.text.split(separator: "\n").first ?? "Tool activity"), systemImage: "terminal").font(.system(size: 11, design: .monospaced)).lineLimit(1).foregroundStyle(muted)
            }.padding(12).background(sidebarColor.opacity(0.65), in: RoundedRectangle(cornerRadius: 10))
        } else if message.role == "status" {
            Label(message.text, systemImage: "arrow.triangle.branch").font(.system(size: 11)).foregroundStyle(muted).textSelection(.enabled)
        } else {
            VStack(alignment: .leading, spacing: 10) {
                HStack {
                    Text(message.role == "user" ? displayedSpeaker.map { "You → \($0)" } ?? "You" : message.role == "error" ? displayedSpeaker.map { "\($0) · Needs attention" } ?? "Needs attention" : displayedSpeaker ?? provider.title).font(.system(size: 12, weight: .semibold)).foregroundStyle(message.role == "error" ? Color.red : ink)
                    Spacer()
                    Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(message.text, forType: .string) } label: { Image(systemName: "doc.on.doc").font(.system(size: 10)).foregroundStyle(muted) }.buttonStyle(.plain).help("Copy message")
                }
                if message.role == "user", message.text.count > 1000 {
                    DisclosureGroup("Research instructions", isExpanded: $expanded) { Text(message.text).font(.system(size: 13)).textSelection(.enabled).padding(.top, 8) }
                } else {
                    Text(.init(message.role == "assistant" ? message.text.split(separator: "\n", omittingEmptySubsequences: false).filter { !$0.hasPrefix("RESEARCH_STATUS: ") }.joined(separator: "\n") : message.text)).font(.system(size: 14)).lineSpacing(5).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                }
            }.padding(message.role == "user" || message.role == "error" ? 16 : 0)
                .background(message.role == "user" ? sidebarColor.opacity(0.85) : message.role == "error" ? Color.red.opacity(0.05) : .clear, in: RoundedRectangle(cornerRadius: 14))
        }
    }
}
