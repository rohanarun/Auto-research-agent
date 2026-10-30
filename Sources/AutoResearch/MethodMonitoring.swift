import Foundation
import ResearchCore

extension AppStore {
    func consumeMethodLine(_ line: String, sessionID id: UUID, operation: String) {
        guard let index = sessions.firstIndex(where: { $0.id == id }) else { return }
        if let messages = MethodTranscript.decode(line) {
            for item in messages {
                let key = "method/" + operation + "/" + item.key
                if let at = sessions[index].messages.firstIndex(where: { $0.sourceKey == key }) {
                    sessions[index].messages[at].text = item.text
                } else {
                    var message = ChatMessage(role: item.role, text: item.text)
                    message.sourceKey = key; message.speaker = item.speaker
                    sessions[index].messages.append(message)
                }
            }
            save(); return
        }
        if let value = (try? JSONSerialization.jsonObject(with: Data(line.utf8))) as? [String: Any] {
            if value["status"] as? String == "COMPLETED" { return } // Summarized by the execution owner.
            if let phase = value["phase"] as? String, let round = value["round"] as? Int {
                append(id, role: "status", text: "Round \(round) · \(phase) finished. Evaluator score: \(value["score"] ?? "unavailable").")
                return
            }
        }
        if !line.isEmpty { append(id, role: "tool", text: line) }
    }

    func methodAttachmentURL(_ id: UUID) throws -> URL {
        try Workspace.directory(settings.workspaceRoot, "cloud-method-attachments").appendingPathComponent(id.uuidString + ".json")
    }

    func restoreMethodMonitoring() {
        let directory = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("cloud-method-attachments")
        guard let files = try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil) else { return }
        for file in files where file.pathExtension == "json" {
            do {
                let attachment = try JSONDecoder().decode(CloudMethodAttachment.self, from: Data(contentsOf: file))
                guard let session = sessions.first(where: { $0.id == attachment.sessionID }),
                      session.cloud?.sandboxID == attachment.sandboxID,
                      UUID(uuidString: attachment.operationID) != nil,
                      attachment.controlDirectory.hasPrefix("/workspace/checkout/SHARED_CONTEXT/methods-"),
                      !attachment.controlDirectory.contains(".."),
                      runners[session.id] == nil else { continue }
                resumeMethodMonitoring(attachment, session: session)
            } catch { self.error = "Saved research monitor could not be restored: " + error.localizedDescription }
        }
    }

    private func resumeMethodMonitoring(_ attachment: CloudMethodAttachment, session: ResearchSession) {
        guard let cloud = session.cloud else { return }
        guard cloud.provider == daytona.sandboxProvider else {
            error = "Select \(cloud.provider.title) in Connections & settings → Cloud to reconnect saved research.";return
        }
        let id = session.id
        // Reserve the saved session immediately; no second launch can race attachment.
        runners[id] = ProcessRunner(); state(id, "Reconnecting")
        workerTasks[id] = Task {
            do {
                let client = try cloudClient()
                _ = try await client.inspect(cloud.sandboxID)
                // Updating the polling bridge does not replace the live method process.
                try await client.upload(cloud.sandboxID, files: [
                    "/opt/autoresearch/bridge.py": try Data(contentsOf: resourceDirectory.appendingPathComponent("DaytonaWorker.py")),
                    "/opt/autoresearch/MethodTelemetry.py": try Data(contentsOf: resourceDirectory.appendingPathComponent("MethodTelemetry.py"))])
                let backend = DaytonaRunner(client: client, cloud: cloud, configuration: daytona,
                                            operationID: attachment.operationID, resumeExisting: true)
                let runner = ProcessRunner(backend: backend); runners[id] = runner
                state(id, "Running")
                append(id, role: "status", text: "Reconnected to the existing cloud research run. Worker messages are read from its saved logs; no agents were restarted.")
                let root = try Workspace.directory(settings.workspaceRoot, "sessions/" + id.uuidString)
                let result = try await runner.run(Command("autoresearch-methods", [attachment.controlDirectory]), directory: root, onLine: { line, _ in
                    Task { @MainActor in self.consumeMethodLine(line, sessionID: id, operation: attachment.operationID) }
                })
                if stopped.contains(id) { state(id, "Stopped") }
                else if result.code == 0 {
                    append(id, role: "assistant", text: try ResearchMethods.summary(result.stdout)); state(id, "Completed")
                } else {
                    append(id, role: "error", text: "Research execution ended with code \(result.code). All checkpoints and logs remain in the cloud sandbox.")
                    state(id, "Needs attention")
                }
                cloudDrains[id] = true
                try FileManager.default.removeItem(at: methodAttachmentURL(id))
            } catch {
                state(id, "Needs attention")
                append(id, role: "error", text: "Research monitoring needs attention. The saved remote operation was not restarted or cancelled. " + error.localizedDescription)
            }
            runners.removeValue(forKey: id); persist()
            if keepResearching.contains(id), sessions.first(where: { $0.id == id })?.state == "Completed",
               let project = projects.first(where: { $0.id == session.projectID }) {
                let next = (try? String(contentsOf: resourceDirectory.appendingPathComponent("ContinuePolicy.md"), encoding: .utf8)) ?? ""
                if !next.isEmpty { launchCloud(id, project: project, request: next) }
            }
        }
    }
}
