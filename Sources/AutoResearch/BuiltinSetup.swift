import AppKit
import Foundation
import ResearchCore

extension AppStore {
    /// Writes only a terminal launcher. Authentication stays in the official CLI
    /// and user-visible Terminal; the app never collects its output or secrets.
    func prepareOKCLogin(_ project: ResearchProject, chooseAccount: Bool = false) throws -> URL {
        let directory = try prepareBuiltinWorkspace(project)
        let folder = try Workspace.directory(settings.workspaceRoot, "terminal")
        let file = folder.appendingPathComponent("okc-login-" + UUID().uuidString + ".command")
        let script = OKCLogin.commandScript(directory: directory, home: FileManager.default.homeDirectoryForCurrentUser, chooseAccount: chooseAccount)
        try Data(script.utf8).write(to: file, options: .withoutOverwriting)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: file.path)
        return file
    }

    /// Read-only account inspection. Only a parsed username leaves this method;
    /// neither raw output nor credentials are persisted or sent to the model.
    func refreshOKCAccount() async throws -> String {
        let directory = try Workspace.validate(settings.workspaceRoot)
        let environment = try environment()
        let searchPath = environment["PATH"] ?? ProcessInfo.processInfo.environment["PATH"] ?? ""
        var candidates = searchPath.split(separator: ":").map { URL(fileURLWithPath: String($0)).appendingPathComponent("okc") }
        candidates.append(FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".okc/bin/okc"))
        guard let executable = candidates.first(where: { FileManager.default.isExecutableFile(atPath: $0.path) }) else {
            throw ResearchError("OKC CLI is not installed. Follow the installation instructions, then choose your account.")
        }
        let runner = ProcessRunner()
        let result = try await withTaskCancellationHandler {
            try await runner.run(Command(executable.path, ["whoami"]), directory: directory, environment: environment)
        } onCancel: {
            runner.stop()
        }
        try Task.checkCancellation()
        guard result.code == 0 else {
            throw ResearchError("OKC could not verify a signed-in account. Use Choose account, complete login in Terminal, then refresh the account.")
        }
        guard let account = OKCLogin.account(from: result.stdout) else {
            throw ResearchError("The OKC account could not be identified. Run okc whoami in Terminal and verify your account before trying again.")
        }
        return account
    }

    func openOKCLogin(_ project: ResearchProject, chooseAccount: Bool = false) {
        do {
            let file = try prepareOKCLogin(project, chooseAccount: chooseAccount)
            guard let terminal = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else {
                throw ResearchError("Terminal is unavailable. Run okc login in your preferred terminal.")
            }
            NSWorkspace.shared.open([file], withApplicationAt: terminal, configuration: NSWorkspace.OpenConfiguration()) { _, error in
                if let error { Task { @MainActor in self.error = error.localizedDescription } }
            }
        } catch { self.error = error.localizedDescription }
    }
}
