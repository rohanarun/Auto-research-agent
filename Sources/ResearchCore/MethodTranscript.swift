import Foundation

/// Worker events belong to the coordinator's transcript, not its CLI lifecycle.
/// In particular, a worker completing or failing cannot complete the whole run.
public struct MethodTranscriptMessage {
    public let key: String
    public let role: String
    public let text: String
    public let speaker: String
}

public enum MethodTranscript {
    public static func decode(_ line: String) -> [MethodTranscriptMessage]? {
        guard let object = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any],
              object["type"] as? String == "research_method_event",
              let branch = object["branch"] as? String,
              let provider = Provider(rawValue: object["provider"] as? String ?? ""),
              var event = object["event"] as? [String: Any] else { return nil }
        let fallback = object["key"] as? String ?? branch
        // Provider IDs identify updates; log offsets identify events without IDs.
        for field in ["item", "message"] {
            if var value = event[field] as? [String: Any], value["id"] == nil {
                value["id"] = fallback; event[field] = value
            }
        }
        guard let data = try? JSONSerialization.data(withJSONObject: event) else { return [] }
        let parts = branch.split(separator: "-").map(String.init)
        func value(_ field: String) -> String? {
            guard let at = parts.firstIndex(of: field), parts.indices.contains(at + 1) else { return nil }
            return parts[at + 1]
        }
        let phase = parts.contains("explore") ? "Exploration" : parts.contains("continue") ? "Continuation" : "Research"
        let label = [value("branch").map { "Worker " + $0 }, provider.title, phase,
                     value("round").map { "Round " + $0 }, value("epoch").map { "Checkpoint " + $0 }].compactMap { $0 }.joined(separator: " · ")
        return EventDecoder.decode(String(decoding: data, as: UTF8.self), provider: provider).compactMap { decoded in
            switch decoded {
            case .message(let key, let role, let text):
                return MethodTranscriptMessage(key: branch + "/" + key, role: role, text: text, speaker: label)
            case .completed:
                return MethodTranscriptMessage(key: fallback, role: "status", text: "Worker turn finished; checkpoint verification follows.", speaker: label)
            case .failure(let text):
                return MethodTranscriptMessage(key: fallback, role: "error", text: text, speaker: label)
            case .session:
                return MethodTranscriptMessage(key: fallback, role: "status", text: "Worker connected.", speaker: label)
            }
        }
    }
}

public struct CloudMethodAttachment: Codable, Sendable {
    public let sessionID: UUID
    public let sandboxID: String
    public let operationID: String
    public let controlDirectory: String
    public init(sessionID: UUID, sandboxID: String, operationID: String, controlDirectory: String) {
        self.sessionID = sessionID; self.sandboxID = sandboxID
        self.operationID = operationID; self.controlDirectory = controlDirectory
    }
}
