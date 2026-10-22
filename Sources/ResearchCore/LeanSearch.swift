import Foundation

public struct ResearchSubtask: Codable, Identifiable, Sendable {
    public let id: String
    public let title: String
    public let symbol: String
    public let summary: String
    public let request: String
    public let overviewTitle: String
    public let overview: ResearchSubtaskOverview
    public let execution: String?
    public let requiresCPUSearch: Bool?
}

public struct CPUSearchConfiguration: Codable {
    public var python: String
    public var script: String
    public var plan: String
    public var summary: String?
}

public struct CPUSearchResult: Codable {
    public let status: String
    public let summary: String
}

public struct ResearchSubtaskOverview: Codable, Equatable, Sendable {
    public let objective: String
    public let method: String
    public let meaning: String
}

public enum LeanSearch {
    public static func cpuConfiguration(directory: URL) -> CPUSearchConfiguration? {
        let url = directory.appendingPathComponent(".auto-research/cpu-search.json")
        guard let data = try? Data(contentsOf: url),
              let config = try? JSONDecoder().decode(CPUSearchConfiguration.self, from: data),
              let summary = config.summary,
              [config.python, config.script, config.plan, summary].allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }),
              [config.script, config.plan].allSatisfy({ path in
                  let file = directory.appendingPathComponent(path)
                  return (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true
                      && FileManager.default.isReadableFile(atPath: file.path)
              }) else { return nil }
        return config
    }

    public static func overviewURL(directory: URL, sessionID: UUID) -> URL {
        directory.appendingPathComponent(".auto-research-tools/lean-search/runs/\(sessionID.uuidString)/overview.json")
    }

    public static func readOverview(at url: URL) -> ResearchSubtaskOverview? {
        guard let size = try? url.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= 16_384,
              let data = try? Data(contentsOf: url),
              let overview = try? JSONDecoder().decode(ResearchSubtaskOverview.self, from: data),
              [overview.objective, overview.method, overview.meaning].allSatisfy({ !$0.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }) else { return nil }
        return overview
    }

    public static func overviewInstruction(at url: URL) -> String {
        """
        The app shows a plain-language experiment overview above this subtask's chat.
        After reading the repository and choosing the experiment, write \(url.path)
        as JSON with three nonempty string fields: objective, method, meaning.
        Explain the exact parameters or objects being searched and the target improvement
        in objective; the candidate checks and bounded experiment in method; and what
        success would establish, with unresolved proof obligations, in meaning.
        Use brief accessible prose, define symbols, and avoid implementation paths or jargon.
        Update it when the experiment changes and at completion, retaining failed results
        and distinguishing conditional evidence from Lean or Comparator acceptance.
        These fields describe the model's plan and findings; they do not control execution.
        """
    }

    public static func subtasks(resources: URL) throws -> [ResearchSubtask] {
        try JSONDecoder().decode([ResearchSubtask].self, from: Data(contentsOf: resources.appendingPathComponent("LeanSearch/subtasks.json")))
    }

    public static func files(resources: URL) throws -> [String: Data] {
        let root = resources.appendingPathComponent("LeanSearch")
        let names = ["cli.py", "core.py", "qrh.py", "server.py", "remote.py", "SKILL.md", "qrh-track-a.json", "generic-plan.json", "subtasks.json", "cpu_search.py", "qrh-cpu-plan.json"]
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
