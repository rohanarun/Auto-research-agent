import AppKit
import ResearchCore

/// Public downloads resolve installation paths on the recipient's Mac.
@MainActor enum PortableSetup {
    static func resolve(_ bundled: ResearchCore.Settings) async -> ResearchCore.Settings {
        guard bundled.workspaceRoot.isEmpty else { return bundled }
        var settings = bundled
        let preferences = UserDefaults.standard
        if let saved = preferences.string(forKey: "researchWorkspace"), (try? Workspace.validate(saved)) != nil {
            settings.workspaceRoot = saved
        } else {
            NSApplication.shared.setActivationPolicy(.regular)
            NSApplication.shared.activate(ignoringOtherApps: true)
            let picker = NSOpenPanel()
            picker.title = "Choose your research workspace"
            picker.message = "Choose or create a folder on a mounted external drive. Research projects, chats and experiment files will stay in this folder."
            picker.prompt = "Use workspace"
            picker.canChooseFiles = false; picker.canChooseDirectories = true
            picker.canCreateDirectories = true; picker.allowsMultipleSelection = false
            picker.directoryURL = URL(fileURLWithPath: "/Volumes", isDirectory: true)
            while settings.workspaceRoot.isEmpty {
                let response: NSApplication.ModalResponse = await withCheckedContinuation { continuation in
                    picker.begin { continuation.resume(returning: $0) }
                }
                guard response == .OK, let url = picker.url else { exit(EXIT_SUCCESS) }
                do {
                    settings.workspaceRoot = try Workspace.validate(url.path).path
                    preferences.set(settings.workspaceRoot, forKey: "researchWorkspace")
                } catch {
                    let alert = NSAlert(); alert.messageText = "Choose an external-drive folder"
                    alert.informativeText = error.localizedDescription; alert.runModal()
                }
            }
        }
        settings.codexPath = executable(.codex)
        settings.claudePath = executable(.claude)
        settings.opencodePath = executable(.opencode)
        return settings
    }
    private static func executable(_ provider: Provider) -> String {
        let process = Process(); let output = Pipe()
        process.executableURL = URL(fileURLWithPath: "/bin/zsh")
        process.arguments = ["-lc", "command -v " + provider.rawValue]
        process.standardOutput = output; process.standardError = FileHandle.nullDevice
        do {
            try process.run()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else { return "" }
            return String(decoding: data, as: UTF8.self).split(separator: "\n").map(String.init).last {
                $0.hasPrefix("/") && FileManager.default.isExecutableFile(atPath: $0)
            } ?? ""
        } catch { return "" }
    }
}
