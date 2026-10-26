import Foundation

/// Bundled discovery and starter instructions. Account state and research results
/// belong to the user's workspace, never to this definition.
public struct BuiltinProject: Decodable, Identifiable, Sendable {
    public let id: String
    public let name: String
    public let summary: String
    public let category: String
    public let website: URL
    public let leaderboard: URL
    public let promptFile: String
    public let suggestedPrompts: [String]

    public var project: ResearchProject {
        ResearchProject(repository: "", name: name, summary: summary, category: category, builtinID: id)
    }

    public func instructions(resources: URL) throws -> String {
        try validate()
        let common = try BuiltinProjects.readPrompt("StarterPrompts/OfficialJudge.md", resources: resources)
        let specific = try BuiltinProjects.readPrompt(promptFile, resources: resources)
        return common + "\n\n" + specific + "\n"
    }

    /// Seeds a local starter once. Existing instructions and research files are
    /// retained, including edits made by the user or a previous model session.
    public func prepare(workspaceRoot: String, resources: URL) throws -> URL {
        let body = try instructions(resources: resources)
        let root = try Workspace.validate(workspaceRoot)
        var candidate = root
        for component in project.managedWorkspaceComponent.split(separator: "/") {
            candidate.appendPathComponent(String(component))
            try BuiltinProjects.rejectSymlink(candidate)
        }
        let directory = try Workspace.directory(root.path, project.managedWorkspaceComponent)
        let instruction = directory.appendingPathComponent("auto-research.md")
        try BuiltinProjects.rejectSymlink(instruction)
        if FileManager.default.fileExists(atPath: instruction.path) {
            let attributes = try FileManager.default.attributesOfItem(atPath: instruction.path)
            guard attributes[.type] as? FileAttributeType == .typeRegular else {
                throw ResearchError("The starter instruction path is not a regular file. Its contents were preserved: \(instruction.path)")
            }
        } else {
            try Data(body.utf8).write(to: instruction, options: .withoutOverwriting)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: instruction.path)
        }
        return URL(fileURLWithPath: directory.path, isDirectory: true)
    }

    fileprivate func validate() throws {
        guard id.range(of: "\\A[a-z0-9][a-z0-9_-]*\\z", options: .regularExpression) != nil else {
            throw ResearchError("A built-in project has an invalid identifier.")
        }
        guard [name, summary, category].allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
              !suggestedPrompts.isEmpty,
              suggestedPrompts.allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else {
            throw ResearchError("The built-in project \(id) is missing its description or starter prompts.")
        }
        for url in [website, leaderboard] {
            guard url.scheme?.lowercased() == "https", let host = url.host, !host.isEmpty,
                  url.user == nil, url.password == nil else {
                throw ResearchError("The built-in project \(id) must use HTTPS public links.")
            }
        }
    }
}

public enum BuiltinProjects {
    public static func load(resources: URL) throws -> [BuiltinProject] {
        let file = try resource("BuiltinProjects.json", resources: resources)
        let definitions = try JSONDecoder().decode([BuiltinProject].self, from: Data(contentsOf: file))
        var seen = Set<String>()
        for definition in definitions {
            guard seen.insert(definition.id).inserted else {
                throw ResearchError("The built-in project catalog contains a duplicate identifier: \(definition.id)")
            }
            _ = try definition.instructions(resources: resources)
        }
        return definitions
    }

    /// Fresh public entries win; saved custom projects and projects with session
    /// history remain accessible even when removed from a remote catalog.
    public static func merge(fetched: [ResearchProject], existing: [ResearchProject], builtins: [BuiltinProject], sessionProjectIDs: Set<String>) -> [ResearchProject] {
        var seen = Set<String>()
        let retained = existing.filter { $0.isCustom || sessionProjectIDs.contains($0.id) }
        return (builtins.map(\.project) + fetched + retained).filter { seen.insert($0.id).inserted }
    }

    fileprivate static func readPrompt(_ path: String, resources: URL) throws -> String {
        let file = try resource(path, resources: resources)
        let body = try String(contentsOf: file, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !body.isEmpty else { throw ResearchError("A built-in starter instruction file is empty: \(path)") }
        return body
    }

    private static func resource(_ path: String, resources: URL) throws -> URL {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        guard !path.isEmpty, !path.hasPrefix("/"), !path.contains("\\"),
              parts.allSatisfy({ !$0.isEmpty && $0 != "." && $0 != ".." }) else {
            throw ResearchError("A built-in starter resource has an invalid relative path.")
        }
        let root = resources.standardizedFileURL.resolvingSymlinksInPath()
        let file = root.appendingPathComponent(path).standardizedFileURL.resolvingSymlinksInPath()
        guard file.path.hasPrefix(root.path + "/") else {
            throw ResearchError("A built-in starter resource points outside the application resources.")
        }
        guard let attributes = try? FileManager.default.attributesOfItem(atPath: file.path),
              attributes[.type] as? FileAttributeType == .typeRegular else {
            throw ResearchError("A built-in starter resource is missing or is not a regular file: \(path)")
        }
        return file
    }

    fileprivate static func rejectSymlink(_ path: URL) throws {
        if let attributes = try? FileManager.default.attributesOfItem(atPath: path.path),
           attributes[.type] as? FileAttributeType == .typeSymbolicLink {
            throw ResearchError("A starter workspace path is a symbolic link. It was preserved: \(path.path)")
        }
    }
}
