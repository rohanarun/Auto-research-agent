import Foundation

public enum SharedWorkspace {
    /// Independent Git metadata and files, including dirty tracked and non-ignored
    /// untracked work. The source checkout is never reset, committed or written.
    public static func prepare(source: URL, destination: URL, runner: ProcessRunner, environment: [String: String]) async throws {
        _ = try Workspace.validate(source.path); _ = try Workspace.validate(destination.path)
        guard !FileManager.default.fileExists(atPath: destination.path) else { throw ResearchError("The agent checkout already exists; it has been preserved: \(destination.path)") }
        func git(_ args: [String], in directory: URL, input: String? = nil) async throws -> String {
            let result = try await runner.run(Command("/usr/bin/git", args, input: input), directory: directory, environment: environment)
            guard result.code == 0 else { throw ResearchError("Could not prepare an independent agent checkout: \(result.stderr)") }
            return result.stdout
        }
        let top = try await git(["rev-parse", "--show-toplevel"], in: source).trimmingCharacters(in: .whitespacesAndNewlines)
        guard URL(fileURLWithPath: top).resolvingSymlinksInPath().path == source.resolvingSymlinksInPath().path else { throw ResearchError("Choose the Git repository root for parallel research.") }
        let index = try await git(["ls-files", "--stage"], in: source)
        guard !index.split(separator: "\n").contains(where: { $0.hasPrefix("160000 ") }) else { throw ResearchError("Parallel checkout snapshots do not yet support Git submodules. Use a single agent for this project.") }
        let patch = try await git(["diff", "--no-ext-diff", "--no-textconv", "--binary", "HEAD", "--"], in: source)
        let untracked = try await git(["ls-files", "--others", "--exclude-standard", "-z"], in: source)
        _ = try await git(["clone", "--no-hardlinks", "--", source.path, destination.path], in: destination.deletingLastPathComponent())
        let origin = try await runner.run(Command("/usr/bin/git", ["remote", "get-url", "origin"]), directory: source, environment: environment)
        if origin.code == 0 {
            _ = try await git(["remote", "set-url", "origin", origin.stdout.trimmingCharacters(in: .whitespacesAndNewlines)], in: destination)
        } else { _ = try await git(["remote", "remove", "origin"], in: destination) }
        if !patch.isEmpty { _ = try await git(["apply", "--binary", "-"], in: destination, input: patch) }
        for path in untracked.split(separator: "\0").map(String.init) {
            let from = source.appendingPathComponent(path)
            let to = destination.appendingPathComponent(path)
            guard from.resolvingSymlinksInPath().path.hasPrefix(source.path + "/"), to.resolvingSymlinksInPath().path.hasPrefix(destination.path + "/") else {
                throw ResearchError("An untracked file points outside its checkout and was not copied: \(path)")
            }
            try FileManager.default.createDirectory(at: to.deletingLastPathComponent(), withIntermediateDirectories: true)
            try FileManager.default.copyItem(at: from, to: to)
        }
    }
}
