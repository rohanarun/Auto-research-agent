import Foundation

public struct ResearchMethodOptions: Codable, Equatable, Sendable {
    public var rsiTop: Bool
    public var forkAndFlush: Bool
    public var branches: Int
    public var explorationMinutes: Double
    public var exploitationMinutes: Double
    public var rounds: Int
    public var enabled: Bool { rsiTop || forkAndFlush }
    public func validate() throws {
        guard enabled else { return }
        guard branches >= 2, rounds >= 1,
              explorationMinutes.isFinite, explorationMinutes > 0,
              exploitationMinutes.isFinite, exploitationMinutes > 0 else {
            throw ResearchError("Research methods need at least two branches, one round, and positive exploration and continuation windows.")
        }
    }
}

/// Portable tools are installed beside the run. Scientific decisions and metric
/// definitions are supplied by the model through the editable method policy.
public enum ResearchMethods {
    public static func files(options: ResearchMethodOptions, provider: Provider, settings: Settings,
                             policy: String, resources: URL, disabledMCPServers: [String] = [],
                             enabledMCPServers: [String] = [], sharedContext: SharedContextOptions? = nil) throws -> [String: Data] {
        try options.validate()
        guard provider == .codex || provider == .claude else {
            throw ResearchError("Optional research methods currently support Codex and Claude. Use a normal \(provider.title) run or choose a supported provider for these methods.")
        }
        if let sharedContext { try sharedContext.validate() }
        let command = CLI.chat(provider: provider, settings: settings, sessionID: nil,
                               prompt: "", policy: policy, disabledMCPServers: disabledMCPServers,
                               enabledMCPServers: enabledMCPServers)
        let config: [String: Any] = [
            "options": try JSONSerialization.jsonObject(with: JSONEncoder().encode(options)),
            "shareContext": sharedContext?.enabled ?? false,
            "sharingIntervalMinutes": sharedContext?.intervalMinutes ?? 0,
            "provider": provider.rawValue, "executable": command.executable,
            "arguments": (provider == .codex ? ["--no-daemon"] : []) + command.arguments, "policy": policy
        ]
        return ["ResearchMethods.py": try Data(contentsOf: resources.appendingPathComponent("ResearchMethods.py")),
                "ResearchMethodsPolicy.md": try Data(contentsOf: resources.appendingPathComponent("ResearchMethodsPolicy.md")),
                "config.json": try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])]
    }
    public static func instruction(directory: String) -> String {
        """
        Optional research methods are enabled for this run. Read \(directory)/ResearchMethodsPolicy.md
        and \(directory)/config.json now. Follow that policy automatically: design and freeze the task's
        evaluator/metrics, save the complete plan to \(directory)/plan.json, then finish your turn with
        RESEARCH_METHOD_PLAN: ready
        on its own final line. The app will execute ResearchMethods.py outside your CLI tool sandbox.
        Do not launch the runner yourself. Its branches use the
        configured model with fresh contexts. Do not merely describe the method or simulate branches.
        The method runner owns parallelism; do not launch a second parallel group. Keep all artifacts.
        """
    }
    public static func ready(_ output: String, provider: Provider) -> Bool {
        let messages = output.split(separator: "\n").flatMap { EventDecoder.decode(String($0), provider: provider) }.compactMap { event -> String? in
            if case .message(_, "assistant", let text) = event { return text }; return nil
        }
        return messages.last?.split(separator: "\n").last?.trimmingCharacters(in: .whitespacesAndNewlines) == "RESEARCH_METHOD_PLAN: ready"
    }
    public static func summary(_ output: String) throws -> String {
        guard let line = output.split(separator: "\n").last,
              let result = try JSONSerialization.jsonObject(with: Data(line.utf8)) as? [String: Any],
              result["status"] as? String == "COMPLETED", let leader = result["leader"] as? [String: Any],
              let workspace = leader["workspace"] as? String, let evaluation = leader["evaluation"] as? [String: Any],
              evaluation["valid"] as? Bool == true, let score = evaluation["score"] as? String else {
            throw ResearchError("Method execution did not return a verified leader. All branch artifacts are preserved.")
        }
        let pending = (result["pending_workspaces"] as? [[String: Any]])?.count ?? 0
        let reason = result["stop_reason"] as? String ?? "configured_batch_complete"
        let receipt = (result["result_file"] as? String).map { " Batch receipt: \($0)." } ?? ""
        return "Research methods completed this batch. Evaluator score: \(score). Selected workspace: \(workspace). Original checkout and other branches preserved. Batch completion does not establish that the research objective is finished. Boundary: \(reason); \(pending) unfinished workspaces need review.\(receipt)"
    }
}
