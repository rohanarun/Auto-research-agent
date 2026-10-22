import AppKit
import Foundation
import ResearchCore

@MainActor final class LeanServerController: ObservableObject {
    @Published var status = "Stopped"
    @Published var localURL = ""
    @Published var publicURL = ""
    @Published var running = false
    @Published var sharing = false
    @Published var busy = false
    @Published var leanPath: String = {
        let home = FileManager.default.homeDirectoryForCurrentUser
        let installed = ((try? FileManager.default.contentsOfDirectory(at: home.appendingPathComponent(".elan/toolchains"), includingPropertiesForKeys: nil)) ?? []).map { $0.appendingPathComponent("bin/lean") }.filter { FileManager.default.isExecutableFile(atPath: $0.path) }
        return installed.count == 1 ? installed[0].path : home.appendingPathComponent(".elan/bin/lean").path
    }()
    @Published var storagePath = ""
    private var workspace: String?
    @Published var ngrokPath = "ngrok"
    private var stopRequested = false
    private var server: Process?
    private var tunnel: Process?
    private var root: URL?
    private var resources: URL?
    private var logs: [FileHandle] = []
    var isActive: Bool { server?.isRunning == true || tunnel?.isRunning == true || busy }
    var connectionPath: String? { running ? root?.appendingPathComponent("connection.json").path : nil }

    func configure(workspace: String, resources: URL) throws {
        guard !isActive else { return }
        self.workspace = workspace
        let saved = URL(fileURLWithPath: workspace).appendingPathComponent("lean-server-location.json")
        if let data = try? Data(contentsOf: saved), let object = try? JSONDecoder().decode([String: String].self, from: data), let path = object["path"] {
            root = try validateStorage(path)
        } else { root = try Workspace.directory(workspace, "lean-verifier") }
        storagePath = root!.path
        self.resources = resources
    }

    private func validateStorage(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        guard !url.pathComponents.contains(where: { ["Documents", "Desktop", "Mobile Documents", "CloudStorage"].contains($0) }) else { throw ResearchError("Choose a server directory outside synced folders") }
        if url.path.hasPrefix("/Volumes/") { _ = try Workspace.validate(url.path) }
        var existing = url
        while !FileManager.default.fileExists(atPath: existing.path) { existing.deleteLastPathComponent() }
        guard FileManager.default.isWritableFile(atPath: existing.path) else { throw ResearchError("Server storage is not writable") }
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    func chooseStorage() {
        guard !isActive, let workspace else { return }
        let panel = NSOpenPanel(); panel.canChooseDirectories = true; panel.canChooseFiles = false; panel.canCreateDirectories = true
        panel.message = "Choose a dedicated Lean server folder outside Documents, Desktop and iCloud. Existing server data stays in its current folder."
        guard panel.runModal() == .OK, let selected = panel.url else { return }
        do {
            root = try validateStorage(selected.path); storagePath = root!.path
            try Workspace.write(["path": storagePath], to: URL(fileURLWithPath: workspace).appendingPathComponent("lean-server-location.json"))
        } catch { status = error.localizedDescription }
    }
    private func saveConnection(_ url: String) throws {
        guard let root else { return }
        let path = root.appendingPathComponent("connection.json")
        let data = try JSONSerialization.data(withJSONObject: ["url": url, "token_file": root.appendingPathComponent("server-token").path])
        try data.write(to: path, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
    }
    func start() {
        guard !isActive, let root, let resources else { return }
        stopRequested = false; busy = true; status = "Preparing Lean verification server"
        Task { [self] in
            do {
                let tools = root.appendingPathComponent("tools")
                try LeanSearch.install(resources: resources, directory: tools)
                let config = root.appendingPathComponent("server-config.json")
                if !FileManager.default.fileExists(atPath: config.path) {
                    let result = try await ProcessRunner().run(Command(leanPath, ["--print-prefix"]), directory: root)
                    guard result.code == 0 else { throw ResearchError("Select an installed Lean executable. \(result.stderr)") }
                    let binary = URL(fileURLWithPath: result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)).appendingPathComponent("bin/lean").path
                    let setup = try await ProcessRunner().run(Command("/usr/bin/env", ["python3", "-B", tools.appendingPathComponent("cli.py").path, "init-server", "--directory", root.path, "--lean", binary]), directory: root)
                    guard setup.code == 0 else { throw ResearchError(setup.stderr) }
                }
                guard !stopRequested else { throw CancellationError() }
                let process = try launch(["python3", "-B", tools.appendingPathComponent("cli.py").path, "serve", "--config", config.path], log: "server.log")
                server = process
                process.terminationHandler = { [weak self] _ in Task { @MainActor in self?.running = false; self?.busy = false; self?.localURL = ""; self?.status = "Server stopped; jobs and logs are retained" } }
                while process.isRunning {
                    if let data = try? Data(contentsOf: root.appendingPathComponent("server.json")),
                       let value = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                       value["pid"] as? Int32 == process.processIdentifier, let url = value["url"] as? String {
                        let token = try String(contentsOf: root.appendingPathComponent("server-token"), encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
                        var request = URLRequest(url: URL(string: url + "/v1/health")!)
                        request.setValue("Bearer " + token, forHTTPHeaderField: "Authorization")
                        let (_, response) = try await URLSession.shared.data(for: request)
                        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw ResearchError("Lean server health check failed") }
                        localURL = url; running = true; busy = false; status = "Ready · localhost only"
                        try saveConnection(url)
                        return
                    }
                    try await Task.sleep(for: .milliseconds(100))
                }
                throw ResearchError("Server failed to start. Inspect server.log in the server folder.")
            } catch { busy = false; status = error.localizedDescription; server?.terminate() }
        }
    }

    private func launch(_ arguments: [String], log name: String) throws -> Process {
        guard let root else { throw ResearchError("Choose a workspace first") }
        let path = root.appendingPathComponent(name)
        FileManager.default.createFile(atPath: path.path, contents: nil, attributes: [.posixPermissions: 0o600])
        let log = try FileHandle(forWritingTo: path); logs.append(log)
        let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = arguments; process.currentDirectoryURL = root
        var env = ProcessInfo.processInfo.environment
        env["PATH"] = ["/opt/homebrew/bin", "/usr/local/bin", env["PATH"] ?? "/usr/bin:/bin"].joined(separator: ":")
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        process.environment = env; process.standardOutput = log; process.standardError = log
        try process.run(); return process
    }

    func share() {
        guard running, !sharing, let root else { return }
        do {
            let process = try launch([ngrokPath, "http", localURL, "--inspect=false", "--log=stdout", "--log-format=json"], log: "ngrok.log")
            tunnel = process; sharing = true; status = "Starting ngrok"
            process.terminationHandler = { [weak self] _ in Task { @MainActor in self?.sharing = false; self?.publicURL = ""; self?.status = "ngrok stopped; inspect ngrok.log for connection errors" } }
            Task {
                while process.isRunning {
                    let text = (try? String(contentsOf: root.appendingPathComponent("ngrok.log"), encoding: .utf8)) ?? ""
                    for line in text.split(separator: "\n") {
                        if let value = try? JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
                           let url = value["url"] as? String, url.hasPrefix("https://") {
                            publicURL = url; status = "Shared through ngrok · bearer token required"; return
                        }
                    }
                    try? await Task.sleep(for: .milliseconds(250))
                }
            }
        } catch { status = error.localizedDescription }
    }
    func remoteConnectionFiles() throws -> [String: Data] {
        guard running, sharing, !publicURL.isEmpty, let root else { return [:] }
        return ["connection.json": try JSONSerialization.data(withJSONObject: ["url": publicURL, "token_file": "server-token"]),
                "server-token": try Data(contentsOf: root.appendingPathComponent("server-token"))]
    }
    func stopSharing() { tunnel?.terminate(); publicURL = "" }
    func stop() { stopRequested = true; stopSharing(); server?.terminate(); if server?.isRunning == true { status = "Draining the active verification job" } }
    func openFolder() { if let root { NSWorkspace.shared.open(root) } }
    func copyToken() {
        guard let root, let token = try? String(contentsOf: root.appendingPathComponent("server-token"), encoding: .utf8) else { return }
        NSPasteboard.general.clearContents(); NSPasteboard.general.setString(token.trimmingCharacters(in: .whitespacesAndNewlines), forType: .string)
    }
}
