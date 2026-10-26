import AppKit
import Foundation
import ResearchCore

extension AppStore {
    /// Writes only a terminal launcher. Authentication stays in the official CLI
    /// and user-visible Terminal; the app never collects its output or secrets.
    func prepareOKCLogin(_ project: ResearchProject) throws -> URL {
        let directory = try prepareBuiltinWorkspace(project)
        let folder = try Workspace.directory(settings.workspaceRoot, "terminal")
        let file = folder.appendingPathComponent("okc-login-" + UUID().uuidString + ".command")
        let script = OKCLogin.commandScript(directory: directory, home: FileManager.default.homeDirectoryForCurrentUser)
        try Data(script.utf8).write(to: file, options: .withoutOverwriting)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: file.path)
        return file
    }

    func openOKCLogin(_ project: ResearchProject) {
        do {
            let file = try prepareOKCLogin(project)
            guard let terminal = NSWorkspace.shared.urlForApplication(withBundleIdentifier: "com.apple.Terminal") else {
                throw ResearchError("Terminal is unavailable. Run okc login in your preferred terminal.")
            }
            NSWorkspace.shared.open([file], withApplicationAt: terminal, configuration: NSWorkspace.OpenConfiguration()) { _, error in
                if let error { Task { @MainActor in self.error = error.localizedDescription } }
            }
        } catch { self.error = error.localizedDescription }
    }
}
