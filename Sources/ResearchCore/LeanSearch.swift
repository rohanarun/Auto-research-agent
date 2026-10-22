import Foundation

public struct ResearchSubtask: Codable, Identifiable, Sendable {
    public let id: String
    public let title: String
    public let symbol: String
    public let summary: String
    public let request: String
}

public enum LeanSearch {
    public static func subtasks(resources: URL) throws -> [ResearchSubtask] {
        try JSONDecoder().decode([ResearchSubtask].self, from: Data(contentsOf: resources.appendingPathComponent("LeanSearch/subtasks.json")))
    }

    public static func files(resources: URL) throws -> [String: Data] {
        let root = resources.appendingPathComponent("LeanSearch")
        let names = ["cli.py", "core.py", "qrh.py", "server.py", "remote.py", "SKILL.md", "qrh-track-a.json", "generic-plan.json", "subtasks.json"]
        return try Dictionary(uniqueKeysWithValues: names.map { ($0, try Data(contentsOf: root.appendingPathComponent($0))) })
    }

    public static func install(resources: URL, directory: URL) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        for (name, data) in try files(resources: resources) {
            try data.write(to: directory.appendingPathComponent(name), options: .atomic)
        }
    }

    public static func instruction(directory: String, connection: String? = nil) -> String {
        var text = """
        CPU search and Lean verification tools are available at \(directory)/cli.py.
        For mathematical search with Lean, read \(directory)/SKILL.md. The model chooses
        the experiment, hypotheses and next action using current source evidence.
        Search and exact arithmetic do not establish a Lean theorem. Preserve failed
        candidates and require the correct terminal theorem, axiom audit and Comparator
        for a full formal claim. Generalize through reviewed plans, not fixed targets.
        """
        if let connection {
            text += "\nA shared Lean server connection is available at \(connection). Read its health/profiles through the CLI. Prefer submitting candidates there over duplicating Lean builds on search nodes. Never print its token or include it in prompts, logs or committed files."
        }
        return text
    }
}
