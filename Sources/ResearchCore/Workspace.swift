import Foundation
import Darwin

public enum Workspace {
    public static func validate(_ path: String) throws -> URL {
        let url = URL(fileURLWithPath: path).standardizedFileURL.resolvingSymlinksInPath()
        let parts = url.pathComponents
        guard parts.count > 3, parts[1] == "Volumes" else { throw ResearchError("Choose a workspace on your mounted external SSD. Documents, Desktop and iCloud locations are not supported.") }
        let mount = "/Volumes/" + parts[2]
        var info = statfs()
        guard statfs(mount, &info) == 0 else { throw ResearchError("The workspace volume is not mounted. Reconnect the SSD.") }
        let actual = withUnsafePointer(to: &info.f_mntonname) { pointer in
            pointer.withMemoryRebound(to: CChar.self, capacity: Int(MAXPATHLEN)) { String(cString: $0) }
        }
        guard actual == mount else { throw ResearchError("The workspace path is not an active volume mount. Reconnect the SSD.") }
        var existing = url
        while !FileManager.default.fileExists(atPath: existing.path), existing.path != mount { existing.deleteLastPathComponent() }
        guard FileManager.default.isWritableFile(atPath: existing.path) else { throw ResearchError("The workspace is not writable.") }
        return url
    }
    public static func directory(_ root: String, _ component: String) throws -> URL {
        let base = try validate(root)
        let destination = base.appendingPathComponent(component).standardizedFileURL.resolvingSymlinksInPath()
        guard destination.path.hasPrefix(base.path + "/") else { throw ResearchError("The destination escapes the workspace.") }
        _ = try validate(destination.path)
        try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
        return destination
    }
    public static func write<T: Encodable>(_ value: T, to url: URL) throws {
        _ = try validate(url.deletingLastPathComponent().path)
        let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(value).write(to: url, options: .atomic)
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
    }
    public static func instructionFile(in root: URL, trackedFiles: [String]) -> URL? {
        // Prefer the literal repository entry point, then case-insensitive equivalents.
        let candidates = trackedFiles.filter { URL(fileURLWithPath: $0).lastPathComponent.lowercased() == "auto-research.md" }
            .sorted { a, b in
                if a == "auto-research.md" { return true }
                if b == "auto-research.md" { return false }
                return a.split(separator: "/").count == b.split(separator: "/").count ? a < b : a.split(separator: "/").count < b.split(separator: "/").count
            }
        for path in candidates {
            let file = root.appendingPathComponent(path).resolvingSymlinksInPath()
            if file.path.hasPrefix(root.resolvingSymlinksInPath().path + "/"), FileManager.default.fileExists(atPath: file.path) { return file }
        }
        let literal = root.appendingPathComponent("auto-research.md").resolvingSymlinksInPath()
        return literal.path.hasPrefix(root.resolvingSymlinksInPath().path + "/") && FileManager.default.fileExists(atPath: literal.path) ? literal : nil
    }
}
