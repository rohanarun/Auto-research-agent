import AppKit
import SwiftUI
import ResearchCore

@MainActor final class AppStore: ObservableObject {
    @Published var settings: ResearchCore.Settings
    @Published var projects: [ResearchProject] = []
    @Published var sessions: [ResearchSession] = []
    @Published var selectedProjectID: String?
    @Published var selectedSessionID: UUID?
    @Published var selectedSharedGroupID: UUID?
    @Published var provider: Provider = .codex
    @Published var search = ""
    @Published var composer = ""
    @Published var showingProjectDetails = true
    @Published var error: String?
    @Published var catalogStatus = "Loading catalog"
    @Published var refreshing = false
    @Published var showSettings = false
    @Published var showAddProject = false
    @Published var auth: [Provider: String] = [:]
    @Published var loginProvider: Provider?
    @Published var loginOutput = ""
    @Published var policy: String = ""
    @Published var availableMCPServers: [String] = []
    @Published var enabledMCPServers: Set<String> = []
    @Published var keepResearching = Set<UUID>()
    @Published var daytona: DaytonaConfiguration
    @Published var daytonaStatus = "Not connected"
    @Published var daytonaLoginOutput = ""
    @Published var daytonaBusy = false
    @Published var cloudCleanup = Set<String>()
    let cloudRecoveryQueue = CloudRecoveryQueue()
    var cloudDrains: [UUID:Bool] = [:]
    var cloudRuntime: [String:String]?
    var daytonaClient: DaytonaClient?
    var daytonaLoginCancelled = false
    var daytonaLoginRunner: ProcessRunner?
    var cloudOwnedIDs = Set<String>()
    var runners: [UUID: ProcessRunner] = [:]
    var workerTasks: [UUID: Task<Void, Never>] = [:]
    @Published var sharedOptions: [String: SharedContextOptions] = [:]
    @Published var sharedRuns: [String: SharedRunStatus] = [:]
    var sharedCycles: [String: SharedContextCycle] = [:]
    var sharedOperations: [String: ProcessRunner] = [:]
    var sharedDefaults: SharedContextOptions
    var activePaths: [UUID: String] = [:]
    var stopped = Set<UUID>()
    var eventMessages: [UUID: [String: UUID]] = [:]
    var failed = Set<UUID>()
    var completed = Set<UUID>()
    private var loginRunner: ProcessRunner?
    private var saveTask: Task<Void, Never>?
    private var checkingAuth = false
    private var historyIsReadable = true
    let resourceDirectory: URL
    private let catalogPolicy: CatalogPolicy
    private var settingsFile: URL
    private var needsWorkspace = false
    private var starting = false

    init(resourceDirectory overrideResourceDirectory: URL? = nil, defaults overrideDefaults: ResearchCore.Settings? = nil) {
        let executable = URL(fileURLWithPath: CommandLine.arguments[0]).resolvingSymlinksInPath()
        let bundled = Bundle.main.resourceURL ?? executable.deletingLastPathComponent()
        resourceDirectory = overrideResourceDirectory ?? (FileManager.default.fileExists(atPath: bundled.appendingPathComponent("Defaults.json").path)
            ? bundled : executable.deletingLastPathComponent().appendingPathComponent("Resources"))
        catalogPolicy = try! JSONDecoder().decode(CatalogPolicy.self, from: Data(contentsOf: resourceDirectory.appendingPathComponent("CatalogPolicy.json")))
        sharedDefaults = try! JSONDecoder().decode(SharedContextOptions.self, from: Data(contentsOf: resourceDirectory.appendingPathComponent("SharedContextDefaults.json")))
        // The build embeds installation defaults; the runtime settings remain user editable.
        let bundledDefaults = overrideDefaults ?? (try! JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: resourceDirectory.appendingPathComponent("Defaults.json"))))
        daytona = try! JSONDecoder().decode(DaytonaConfiguration.self,from:Data(contentsOf:resourceDirectory.appendingPathComponent("DaytonaDefaults.json")))
        settings = bundledDefaults
        settingsFile = URL(fileURLWithPath: bundledDefaults.workspaceRoot).appendingPathComponent("settings.json")
        needsWorkspace = bundledDefaults.workspaceRoot.isEmpty
        if !needsWorkspace { loadWorkspace(bundledDefaults) }
    }

    private func loadWorkspace(_ defaults: ResearchCore.Settings) {
        settingsFile = URL(fileURLWithPath: defaults.workspaceRoot).appendingPathComponent("settings.json")
        settings = (try? Data(contentsOf: settingsFile)).flatMap { try? JSONDecoder().decode(ResearchCore.Settings.self, from: $0) } ?? defaults
        // Workspaces saved before OpenCode support pick up the detected CLI instead of an empty path.
        if settings.opencodePath.isEmpty { settings.opencodePath = defaults.opencodePath }
        if let saved = try? Data(contentsOf:URL(fileURLWithPath:settings.workspaceRoot).appendingPathComponent("daytona.json")),let config = try? JSONDecoder().decode(DaytonaConfiguration.self,from:saved) { daytona = config }
        if let ids = try? Data(contentsOf:URL(fileURLWithPath:settings.workspaceRoot).appendingPathComponent("daytona-owned.json")),let decoded = try? JSONDecoder().decode(Set<String>.self,from:ids) { cloudOwnedIDs = decoded }
        policy = (try? String(contentsOf: URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("research-policy.md"), encoding: .utf8))
            ?? (try? String(contentsOf: resourceDirectory.appendingPathComponent("ResearchPolicy.md"), encoding: .utf8)) ?? ""
        if let data = try? Data(contentsOf: URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("mcp-selection.json")), let names = try? JSONDecoder().decode([String].self, from: data) { enabledMCPServers = Set(names) }
        do {
            _ = try Workspace.directory(settings.workspaceRoot, "sessions")
            let stateURL = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("state.json")
            if FileManager.default.fileExists(atPath: stateURL.path) {
                let saved = try JSONDecoder().decode(SavedState.self, from: Data(contentsOf: stateURL))
                projects = saved.projects; sessions = saved.sessions
                sharedOptions = saved.sharedOptions ?? [:]
                selectedProjectID = saved.selectedProjectID; selectedSessionID = saved.selectedSessionID; provider = saved.provider
                selectedSharedGroupID = saved.selectedSharedGroupID
                for i in sessions.indices where ["Running", "Preparing", "Stopping"].contains(sessions[i].state) {
                    sessions[i].state = "Interrupted"
                    sessions[i].messages.append(ChatMessage(role: "status", text: "The app closed during this run. Review its files before resuming; no work has been restarted automatically."))
                }
            }
            if projects.isEmpty, let html = try? String(contentsOf: resourceDirectory.appendingPathComponent("catalog-source.html"), encoding: .utf8) {
                projects = try Catalog.parse(html)
                catalogStatus = "Saved catalog · refreshing"
            }
            selectInitialProject()
        } catch { historyIsReadable = false; self.error = "Saved history could not be loaded. It will not be overwritten. \(error.localizedDescription)" }
    }

    var project: ResearchProject? { projects.first { $0.id == selectedProjectID } }
    var session: ResearchSession? { sessions.first { $0.id == selectedSessionID } }
    var visibleProjects: [ResearchProject] { projects.filter { catalogPolicy.includes($0) && (search.isEmpty || "\($0.name) \($0.repository) \($0.category)".localizedCaseInsensitiveContains(search)) } }
    private func selectInitialProject() {
        // Keep an existing chat accessible while excluding its repository from discovery.
        if selectedSessionID == nil, selectedSharedGroupID == nil, project.map({ !catalogPolicy.includes($0) }) ?? true {
            selectedProjectID = projects.first(where: catalogPolicy.includes)?.id
        }
    }
    var projectSessions: [ResearchSession] { sessions.filter { $0.projectID == selectedProjectID && $0.closedAt == nil }.sorted { $0.createdAt > $1.createdAt } }
    var closedProjectSessions: [ResearchSession] { sessions.filter { $0.projectID == selectedProjectID && $0.closedAt != nil }.sorted { $0.closedAt! > $1.closedAt! } }
    var isRunning: Bool { selectedSessionID.map { runners[$0] != nil } ?? false }
    var hasActiveRuns: Bool { daytonaBusy || !cloudCleanup.isEmpty || !runners.isEmpty || sharedCycles.values.contains { $0.active } }
    func running(_ project: ResearchProject) -> Bool { sharedActive(project.id) || sessions.contains { $0.projectID == project.id && runners[$0.id] != nil } }

    func start() async {
        guard !starting else { return }; starting = true
        if needsWorkspace {
            let resolved = await PortableSetup.resolve(settings)
            loadWorkspace(resolved); needsWorkspace = false
        }
        await refreshCatalog(); await refreshAuth(); await refreshMCPServers()
    }
    func refreshCatalog() async {
        guard !refreshing else { return }; refreshing = true; defer { refreshing = false }
        do {
            let fetched = try await Catalog.fetch(settings.catalogURL)
            let local = projects.filter { old in old.isCustom || !fetched.contains(where: { $0.id == old.id }) && sessions.contains(where: { $0.projectID == old.id }) }
            projects = fetched + local.filter { item in !fetched.contains { $0.id == item.id } }
            selectInitialProject()
            catalogStatus = "\(fetched.filter(catalogPolicy.includes).count) repositories · live"
            save()
        } catch { catalogStatus = "Offline · saved catalog"; self.error = error.localizedDescription }
    }
    func select(_ project: ResearchProject) {
        selectedProjectID = project.id
        selectedSharedGroupID = nil
        selectedSessionID = nil; showingProjectDetails = true
        composer = ""; save()
    }
    func selectSession(_ session: ResearchSession) {
        if let index = sessions.firstIndex(where: { $0.id == session.id }) { sessions[index].closedAt = nil }
        selectedSharedGroupID = nil; selectedProjectID = session.projectID; selectedSessionID = session.id; provider = session.provider; composer = ""; save()
    }
    func canClose(_ session: ResearchSession) -> Bool {
        runners[session.id] == nil && !(session.sharedAgent?.groupID == sharedRuns[session.projectID]?.groupID && sharedActive(session.projectID))
    }
    func closeSession(_ id: UUID) {
        guard let index = sessions.firstIndex(where: { $0.id == id }), canClose(sessions[index]) else { return }
        sessions[index].closedAt = Date()
        let group = sessions[index].sharedAgent?.groupID
        if selectedSessionID == id || (group != nil && selectedSharedGroupID == group && !sessions.contains(where: { $0.sharedAgent?.groupID == group && $0.closedAt == nil })) { showProjectDetails() }
        save()
    }
    func canCloseGroup(_ groupID: UUID) -> Bool {
        sessions.filter { $0.sharedAgent?.groupID == groupID }.allSatisfy(canClose)
    }
    func closeSharedGroup(_ groupID: UUID) {
        guard canCloseGroup(groupID) else { return }
        let ids = sessions.filter { $0.sharedAgent?.groupID == groupID }.map(\.id)
        for id in ids { closeSession(id) }
        if selectedSharedGroupID == groupID { showProjectDetails() }
        save()
    }
    func newChat() { selectedSharedGroupID = nil; selectedSessionID = nil; showingProjectDetails = false; composer = ""; save() }
    func showProjectDetails() { selectedSharedGroupID = nil; selectedSessionID = nil; showingProjectDetails = true; save() }
    func addRepository(_ text: String) {
        var repo = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if repo.hasPrefix("https://github.com/") { repo = String(repo.dropFirst("https://github.com/".count)) }
        if repo.hasSuffix(".git") { repo = String(repo.dropLast(4)) }
        repo = repo.trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        guard repo.range(of: "^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$", options: .regularExpression) != nil, !repo.split(separator: "/").contains("..") else { error = "Enter a GitHub URL or owner/repository."; return }
        if let existing = projects.first(where: { $0.repository.lowercased() == repo.lowercased() }) { select(existing) }
        else { let added = ResearchProject(repository: repo, isCustom: true); projects.append(added); select(added) }
        showAddProject = false; save()
    }
    func addFolder() {
        let picker = NSOpenPanel(); picker.canChooseFiles = false; picker.canChooseDirectories = true
        picker.directoryURL = URL(fileURLWithPath: settings.workspaceRoot)
        guard picker.runModal() == .OK, let url = picker.url else { return }
        do {
            let valid = try Workspace.validate(url.path)
            let added = ResearchProject(repository: "", name: valid.lastPathComponent, category: "Local project", localPath: valid.path, isCustom: true)
            if !projects.contains(where: { $0.id == added.id }) { projects.append(added) }
            select(added); showAddProject = false; save()
        } catch { self.error = error.localizedDescription }
    }

    func save() {
        saveTask?.cancel()
        saveTask = Task { try? await Task.sleep(for: .milliseconds(300)); guard !Task.isCancelled else { return }; persist() }
    }
    @discardableResult func persist() -> Bool {
        guard historyIsReadable, !needsWorkspace else { return false }
        do {
            let root = try Workspace.validate(settings.workspaceRoot)
            var state = SavedState(projects: projects, sessions: sessions, selectedProjectID: selectedProjectID, selectedSessionID: selectedSessionID, provider: provider)
            state.sharedOptions = sharedOptions
            state.selectedSharedGroupID = selectedSharedGroupID
            try Workspace.write(state, to: root.appendingPathComponent("state.json"))
            return true
        } catch { self.error = "Could not save chat history: \(error.localizedDescription)"; return false }
    }
    func saveSettings() {
        do {
            guard !hasActiveRuns, loginRunner == nil else { throw ResearchError("Stop active runs and login before changing connection settings.") }
            _ = try Workspace.directory(settings.workspaceRoot, "sessions")
            try Workspace.write(settings, to: settingsFile)
            try Workspace.write(enabledMCPServers.sorted(), to: URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("mcp-selection.json"))
            let policyURL = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("research-policy.md")
            try policy.write(to: policyURL, atomically: true, encoding: .utf8)
            persist(); showSettings = false
            Task { await refreshAuth() }
        } catch { self.error = error.localizedDescription }
    }
    func append(_ id: UUID, role: String, text: String) {
        guard let index = sessions.firstIndex(where: { $0.id == id }) else { return }
        sessions[index].messages.append(ChatMessage(role: role, text: text)); save()
    }
    func state(_ id: UUID, _ value: String) { if let i = sessions.firstIndex(where: { $0.id == id }) { sessions[i].state = value }; save() }
    func play(_ project: ResearchProject) {
        if sharedActive(project.id) {
            if let id = sharedRuns[project.id]?.groupID { selectSharedGroup(id, projectID: project.id) }
            return
        }
        if let existing = sessions.first(where: { $0.projectID == project.id && runners[$0.id] != nil }) { selectSession(existing); return }
        if sharedConfiguration(project.id).parallelEnabled { startSharedResearch(project); return }
        select(project)
        let session = ResearchSession(projectID: project.id, title: "Research · \(project.name)", provider: provider)
        sessions.append(session); selectedSessionID = session.id
        launch(session.id, project: project, request: nil)
    }
    func send(to recipientID: UUID? = nil) {
        let request = composer.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !request.isEmpty, let project, !isRunning else { return }
        guard !sharedActive(project.id) else { error = "Stop the parallel group before sending a manual follow-up."; return }
        if let group = selectedSharedGroupID {
            guard let recipientID, SharedTranscript.latestAgents(in: sessions, projectID: project.id, groupID: group).contains(where: { $0.id == recipientID }) else {
                error = "Choose an agent in this group before sending a follow-up."; return
            }
            guard !sessions.contains(where: { $0.projectID == project.id && runners[$0.id] != nil }) else {
                error = "Stop research before sending a manual follow-up."; return
            }
            composer = ""
            if let index = sessions.firstIndex(where: { $0.id == recipientID }) { sessions[index].closedAt = nil }
            launch(recipientID, project: project, request: request)
            return
        }
        if recipientID != nil { error = "Select the agent's group before sending a follow-up."; return }
        if selectedSessionID == nil, sharedConfiguration(project.id).parallelEnabled { composer = ""; startSharedResearch(project, request: request); return }
        if let other = sessions.first(where: { $0.projectID == project.id && runners[$0.id] != nil }) { selectSession(other); error = "This project already has an active run. Stop it before starting another writer."; return }
        composer = ""
        let id: UUID
        if let selectedSessionID { id = selectedSessionID }
        else {
            let session = ResearchSession(projectID: project.id, title: String(request.prefix(65)), provider: provider)
            sessions.append(session); id = session.id; selectedSessionID = id
        }
        launch(id, project: project, request: request)
    }
    func environment() throws -> [String: String] {
        let cache = try Workspace.directory(settings.workspaceRoot, "cache")
        let tmp = try Workspace.directory(settings.workspaceRoot, "tmp")
        let inherited = ProcessInfo.processInfo.environment["PATH"] ?? "/usr/bin:/bin"
        let paths = [URL(fileURLWithPath: settings.codexPath).deletingLastPathComponent().path, URL(fileURLWithPath: settings.claudePath).deletingLastPathComponent().path, URL(fileURLWithPath: settings.opencodePath).deletingLastPathComponent().path, "/opt/homebrew/bin", "/usr/local/bin", inherited]
        return ["PATH": paths.joined(separator: ":"), "TMPDIR": tmp.path + "/", "XDG_CACHE_HOME": cache.path, "PIP_CACHE_DIR": cache.appendingPathComponent("pip").path, "npm_config_cache": cache.appendingPathComponent("npm").path, "CLANG_MODULE_CACHE_PATH": cache.appendingPathComponent("clang").path, "SWIFT_MODULECACHE_PATH": cache.appendingPathComponent("swift").path]
    }
    func launch(_ id: UUID, project: ResearchProject, request: String?) {
        if sessions.first(where: { $0.id == id })?.usesCloud(preferCloud:daytona.enabled) == true { launchCloud(id, project:project, request:request);return }
        guard runners[id] == nil else { return }
        let targetPath = sessions.first(where: { $0.id == id })?.workingDirectory ?? project.localPath ?? URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("projects/\(project.repository)").path
        let target = URL(fileURLWithPath: targetPath).resolvingSymlinksInPath().path
        if activePaths.values.contains(target) { append(id, role: "error", text: "Another chat is already using this checkout. Stop that run before starting another writer."); state(id, "Needs attention"); return }
        // Record the reserved path before the first suspension so folder aliases cannot race a clone.
        // For managed projects the checkout itself is still prepared below.
        activePaths[id] = target
        let runner = ProcessRunner(); runners[id] = runner; stopped.remove(id); failed.remove(id); completed.remove(id); eventMessages[id] = [:]
        state(id, "Preparing")
        workerTasks[id] = Task {
            var nextDecision: String?
            do {
                let env = try environment()
                guard let current = sessions.first(where: { $0.id == id }) else { return }
                guard FileManager.default.isExecutableFile(atPath: settings.executable(current.provider)) else { throw ResearchError("\(current.provider.title) CLI is not installed at the configured path. Open Connections to locate it or view installation instructions.") }
                let directory: URL
                if let path = current.workingDirectory ?? project.localPath { directory = try Workspace.validate(path) }
                else {
                    let parent = try Workspace.directory(settings.workspaceRoot, "projects")
                    directory = parent.appendingPathComponent(project.repository)
                    if !FileManager.default.fileExists(atPath: directory.path) {
                        _ = try Workspace.directory(settings.workspaceRoot, "projects/\(project.repository.components(separatedBy: "/")[0])")
                        append(id, role: "status", text: "Cloning \(project.repository) into the SSD workspace…")
                        let result = try await runner.run(Command("/usr/bin/git", ["clone", "--", "https://github.com/\(project.repository).git", directory.path]), directory: parent, environment: env, onLine: { line, _ in Task { @MainActor in self.append(id, role: "tool", text: line) } })
                        guard result.code == 0 else { throw ResearchError("Clone failed. \(result.stderr)") }
                    }
                    _ = try Workspace.validate(directory.path)
                    guard FileManager.default.fileExists(atPath: directory.appendingPathComponent(".git").path) else { throw ResearchError("An incomplete checkout already exists at \(directory.path). Inspect it before retrying; it has been preserved.") }
                }
                guard !stopped.contains(id) else { throw CancellationError() }
                if let i = sessions.firstIndex(where: { $0.id == id }) { sessions[i].workingDirectory = directory.path }
                var prompt = request ?? ""
                if request == nil {
                    let files = try await runner.run(Command("/usr/bin/git", ["ls-files", "-z"]), directory: directory, environment: env)
                    let tracked = files.stdout.split(separator: "\0").map(String.init)
                    if let instruction = Workspace.instructionFile(in: directory, trackedFiles: tracked) {
                        let body = try String(contentsOf: instruction, encoding: .utf8)
                        guard !body.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw ResearchError("\(instruction.lastPathComponent) is empty. Add research instructions before starting.") }
                        prompt = "Start an autonomous research run in \(directory.path). Read applicable AGENTS.md and follow this repository instruction file: \(instruction.path).\n\n\(body)"
                        append(id, role: "status", text: "Loaded \(instruction.path.replacingOccurrences(of: directory.path + "/", with: ""))")
                    } else {
                        prompt = "Start a research run in \(directory.path). The harness did not find a tracked auto-research.md file. Inspect the repository and its AGENTS.md, locate its actual research instructions (including README-linked guides or program.md), explain the source you choose, and carry out the documented research workflow. If there is no actionable research objective, ask me instead of inventing instructions."
                        append(id, role: "status", text: "auto-research.md was not found. The agent will inspect the repository for its research guide.")
                    }
                }
                guard !stopped.contains(id) else { throw CancellationError() }
                if let shared = current.sharedAgent {
                    let count = shared.agentCount ?? sharedConfiguration(project.id).agentCount
                    prompt += shared.isIndependent
                        ? "\n\nIndependent parallel agent \(shared.index) of \(count). Solve the same user problem independently in your assigned checkout: \(directory.path). No peer context is supplied or exchanged."
                        : "\n\nParallel research agent \(shared.index) of \(count), generation \(shared.generation). Your assigned checkout is \(directory.path)."
                    if !shared.isIndependent, let path = shared.contextFile {
                        prompt += "\nShared context copied into this fresh session from \(path):\n\n" + (try String(contentsOfFile: path, encoding: .utf8))
                    }
                }
                append(id, role: "user", text: prompt)
                let logDir = try Workspace.directory(settings.workspaceRoot, "sessions/\(id.uuidString)")
                let logURL = logDir.appendingPathComponent("\(UUID().uuidString).jsonl")
                FileManager.default.createFile(atPath: logURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
                let log = try FileHandle(forWritingTo: logURL); let logLock = NSLock()
                defer { try? log.close() }
                let sharedPolicy = current.sharedAgent == nil ? "" : try String(contentsOf: resourceDirectory.appendingPathComponent(current.sharedAgent?.isIndependent == true ? "IndependentAgentPolicy.md" : "SharedAgentPolicy.md"), encoding: .utf8)
                let fullPolicy = policy + "\n\n" + sharedPolicy + "\n\nConfigured workspace: \(settings.workspaceRoot). Current checkout: \(directory.path)."
                var disabledMCPServers: [String] = []
                if current.provider == .codex {
                    let discovered = try await runner.run(CLI.mcpDiscovery(settings.codexPath), directory: directory, environment: env)
                    guard discovered.code == 0 else { throw ResearchError("Could not inspect Codex MCP settings. No research process was started. \(discovered.stderr)") }
                    let names = try CLI.configuredMCPServers(Data(discovered.stdout.utf8))
                    availableMCPServers = names
                    disabledMCPServers = names.filter { !enabledMCPServers.contains($0) }
                }
                let command = CLI.chat(provider: current.provider, settings: settings, sessionID: current.providerSessionID, prompt: prompt, policy: fullPolicy, disabledMCPServers: disabledMCPServers, enabledMCPServers: availableMCPServers.filter { enabledMCPServers.contains($0) })
                state(id, "Running")
                let result = try await runner.run(command, directory: directory, environment: env, onLine: { line, stderr in
                    logLock.lock(); try? log.write(contentsOf: Data((line + "\n").utf8)); logLock.unlock()
                    Task { @MainActor in self.consume(line, stderr: stderr, provider: current.provider, sessionID: id) }
                })
                // Derive terminal state from the fully drained stream, independent of UI scheduling.
                for line in result.stdout.split(separator: "\n") {
                    consume(String(line), stderr: false, provider: current.provider, sessionID: id)
                    for event in EventDecoder.decode(String(line), provider: current.provider) {
                        switch event {
                        case .completed: completed.insert(id)
                        case .failure: failed.insert(id)
                        case .session(let providerID):
                            if let i = sessions.firstIndex(where: { $0.id == id }) { sessions[i].providerSessionID = providerID }
                        case .message(_, "assistant", let text): nextDecision = EventDecoder.continuationDecision(text)
                        default: break
                        }
                    }
                }
                if stopped.contains(id) { state(id, "Stopped") }
                else if result.code != 0 || failed.contains(id) {
                    state(id, "Needs attention")
                    if result.code != 0 { append(id, role: "error", text: "\(current.provider.title) exited with status \(result.code). \(result.stderr)") }
                } else if !completed.contains(id) { state(id, "Needs attention"); append(id, role: "error", text: "The CLI exited without a completed-turn event. Review the run log before resuming.") }
                else { state(id, "Completed") }
            } catch {
                if stopped.contains(id) || error is CancellationError { state(id, "Stopped") }
                else { state(id, "Needs attention"); append(id, role: "error", text: error.localizedDescription) }
            }
            runners.removeValue(forKey: id); activePaths.removeValue(forKey: id); persist()
            let finishedSession = sessions.first(where: { $0.id == id })
            if let agent = finishedSession?.sharedAgent, !agent.isIndependent, finishedSession?.state == "Needs attention" {
                sharedCycles[project.id]?.stop(failure: "An agent needs attention. Review its chat before starting another group.")
            }
            if keepResearching.contains(id), finishedSession?.state == "Completed", nextDecision == "continue" {
                let next = (try? String(contentsOf: resourceDirectory.appendingPathComponent("ContinuePolicy.md"), encoding: .utf8)) ?? ""
                if !next.isEmpty { launch(id, project: project, request: next) }
            } else if keepResearching.contains(id) {
                keepResearching.remove(id)
                append(id, role: "status", text: "Continuous research paused. The agent did not request another experiment, or this run needs attention.")
            }
        }
    }
    func consume(_ line: String, stderr: Bool, provider: Provider, sessionID id: UUID) {
        guard let i = sessions.firstIndex(where: { $0.id == id }), !line.isEmpty else { return }
        if stderr {
            if let messageID = eventMessages[id]?["cli-diagnostics"], let index = sessions[i].messages.firstIndex(where: { $0.id == messageID }) {
                sessions[i].messages[index].text += "\n" + line
            } else {
                let message = ChatMessage(role: "tool", text: "CLI diagnostics\n" + line)
                sessions[i].messages.append(message); eventMessages[id, default: [:]]["cli-diagnostics"] = message.id
            }
            save(); return
        }
        for event in EventDecoder.decode(line, provider: provider) {
            switch event {
            case .session(let value): sessions[i].providerSessionID = value
            case .completed: completed.insert(id)
            case .failure(let text): failed.insert(id); append(id, role: "error", text: text)
            case let .message(key, role, text):
                if let messageID = eventMessages[id]?[key], let index = sessions[i].messages.firstIndex(where: { $0.id == messageID }) { sessions[i].messages[index].text = text }
                else { let message = ChatMessage(role: role, text: text); sessions[i].messages.append(message); eventMessages[id, default: [:]][key] = message.id }
            }
        }
        save()
    }
    func stopWorker(_ id: UUID) { keepResearching.remove(id); guard let runner = runners[id] else { return }; stopped.insert(id); state(id, "Stopping"); runner.stop() }
    func stop(_ id: UUID) {
        if let session = sessions.first(where: { $0.id == id }), session.sharedAgent != nil, sharedActive(session.projectID) { stopSharedResearch(session.projectID) }
        else { stopWorker(id) }
    }
    func shutdown() { for cycle in sharedCycles.values { cycle.stop() }; for id in Array(runners.keys) { stopWorker(id) }; loginRunner?.stop(); daytonaLoginCancelled = true;daytonaLoginRunner?.stop()
        if let client = daytonaClient,let connection = daytona.connectionSandboxID {
            cloudCleanup.insert(connection)
            Task {
                do { try await client.stop(connection,archive:false) } catch { self.error = "Cloud connection stop could not be confirmed: " + error.localizedDescription }
                cloudCleanup.remove(connection)
            }
        }
        persist() }

    func refreshAuth() async {
        guard !needsWorkspace else { return }
        guard !checkingAuth else { return }; checkingAuth = true; defer { checkingAuth = false }
        for provider in Provider.allCases {
            guard FileManager.default.isExecutableFile(atPath: settings.executable(provider)) else { auth[provider] = "Not installed"; continue }
            do {
                let directory = try Workspace.directory(settings.workspaceRoot, "auth")
                let result = try await ProcessRunner().run(CLI.auth(provider, settings: settings, login: false), directory: directory, environment: try environment())
                if provider == .claude, let data = result.stdout.data(using: .utf8), let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
                    auth[provider] = object["loggedIn"] as? Bool == true ? "Connected" : "Sign in required"
                } else if provider == .opencode {
                    auth[provider] = result.code == 0 && CLI.opencodeSignedIn(result.stdout) ? "Connected" : "Sign in required"
                } else { auth[provider] = result.code == 0 ? "Connected" : "Sign in required" }
            } catch { auth[provider] = "Unavailable" }
        }
    }
    func refreshMCPServers() async {
        do {
            let directory = try Workspace.directory(settings.workspaceRoot, "auth")
            let result = try await ProcessRunner().run(CLI.mcpDiscovery(settings.codexPath), directory: directory, environment: try environment())
            guard result.code == 0 else { return }
            availableMCPServers = try CLI.configuredMCPServers(Data(result.stdout.utf8))
        } catch { /* Connection status already reports a missing CLI. */ }
    }
    func login(_ provider: Provider) {
        guard loginRunner == nil else { return }
        loginProvider = provider; loginOutput = "Starting \(provider.title) login. Complete sign-in in your browser.\n"
        let runner = ProcessRunner(); loginRunner = runner
        Task {
            do {
                let directory = try Workspace.directory(settings.workspaceRoot, "auth")
                let result = try await runner.run(CLI.auth(provider, settings: settings, login: true), directory: directory, environment: try environment(), interactive: true, onLine: { line, _ in Task { @MainActor in self.loginOutput += line + "\n" } })
                loginOutput += result.code == 0 ? "\nLogin command finished. Checking account…" : "\nLogin exited with status \(result.code)."
            } catch { loginOutput += "\n" + error.localizedDescription }
            loginRunner = nil; await refreshAuth()
        }
    }
    func sendLoginInput(_ input: String) { loginRunner?.sendInput(input) }
    func cancelLogin() { loginRunner?.stop(); loginProvider = nil }
    func revealProject() { if session?.cloud != nil { NSWorkspace.shared.open(URL(string:"https://app.daytona.io/dashboard/sandboxes")!);return }; if let path = session?.workingDirectory ?? project?.localPath { NSWorkspace.shared.open(URL(fileURLWithPath: path)) } }
    func revealLogs() {
        guard let id = selectedSessionID else { return }
        NSWorkspace.shared.open(URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("sessions/\(id.uuidString)"))
    }
    func openTerminal() {
        guard session?.cloud == nil else { error = "This chat runs in Daytona. Open its sandbox from the project button; its CLI history stays in that private sandbox.";return }
        guard let session, let directory = session.workingDirectory, let providerID = session.providerSessionID else { error = "Start a chat first to open its CLI session in Terminal."; return }
        guard !sharedActive(session.projectID) else { error = "Stop the shared-context group before continuing one of its sessions in Terminal."; return }
        do {
            let folder = try Workspace.directory(settings.workspaceRoot, "terminal")
            let file = folder.appendingPathComponent("\(session.id.uuidString).command")
            func quote(_ text: String) -> String { "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'" }
            let args = CLI.resumeArguments(session.provider, sessionID: providerID)
            let env = try environment().map { "export \($0.key)=\(quote($0.value))" }.joined(separator: "\n")
            let script = "#!/bin/zsh\n\(env)\ncd -- \(quote(directory)) || exit\nexec \(quote(settings.executable(session.provider))) \(args.map(quote).joined(separator: " "))\n"
            try script.write(to: file, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: file.path)
            NSWorkspace.shared.open(file)
        } catch { self.error = error.localizedDescription }
    }
}
