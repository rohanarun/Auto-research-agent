import Foundation
import Darwin

public struct Command: Sendable {
    public var executable: String
    public var arguments: [String]
    public var input: String?
    public init(_ executable: String, _ arguments: [String], input: String? = nil) {
        self.executable = executable; self.arguments = arguments; self.input = input
    }
}

public enum CLI {
    public static func chat(provider: Provider, settings: Settings, sessionID: String?, prompt: String, policy: String, disabledMCPServers: [String] = [], enabledMCPServers: [String] = []) -> Command {
        if provider == .codex {
            var args = ["exec", "--disable", "plugins", "-c", "approval_policy=\"never\"", "-c", "sandbox_mode=\"\(settings.codexSandbox)\""]
            for name in disabledMCPServers {
                args += ["-c", "mcp_servers.\(name).enabled=false"]
            }
            for name in enabledMCPServers {
                args += ["-c", "mcp_servers.\(name).enabled=true"]
            }
            if let sessionID { args += ["resume", sessionID] }
            args += ["--json", "--skip-git-repo-check"]
            if !settings.codexModel.isEmpty { args += ["--model", settings.codexModel] }
            args += ["-"]
            return Command(settings.codexPath, args, input: "Harness instructions:\n\(policy)\n\nUser request:\n\(prompt)")
        }
        var args = ["--print", "--verbose", "--output-format", "stream-json", "--permission-mode", settings.claudePermissionMode, "--append-system-prompt", policy]
        if let sessionID { args += ["--resume", sessionID] }
        if !settings.claudeModel.isEmpty { args += ["--model", settings.claudeModel] }
        return Command(settings.claudePath, args, input: prompt)
    }
    public static func configuredMCPServers(_ data: Data) throws -> [String] {
        guard let servers = try JSONSerialization.jsonObject(with: data) as? [[String: Any]] else { throw ResearchError("Codex returned an unreadable MCP configuration. No research process was started.") }
        let names = servers.compactMap { $0["name"] as? String }.sorted()
        guard names.allSatisfy({ !$0.contains(".") }) else { throw ResearchError("A Codex MCP server name contains a dot, which this CLI cannot address with a per-run override. Rename that server in Codex before starting research.") }
        return names
    }
    public static func mcpDiscovery(_ executable: String) -> Command {
        Command(executable, ["--disable", "plugins", "mcp", "list", "--json"])
    }
    public static func auth(_ provider: Provider, settings: Settings, login: Bool) -> Command {
        Command(settings.executable(provider), provider == .codex ? (login ? ["login"] : ["login", "status"]) : ["auth", login ? "login" : "status"])
    }
}

public struct ProcessResult: Sendable {
    public var code: Int32
    public var stdout: String
    public var stderr: String
}

public protocol ProcessBackend: AnyObject, Sendable {
    func run(_ command: Command, onLine: @escaping (String, Bool) -> Void) async throws -> ProcessResult
    func stop()
    func sendInput(_ text: String)
}

public final class ProcessRunner: @unchecked Sendable {
    private let lock = NSLock()
    private var process: Process?
    private var input: FileHandle?
    private var cancelled = false
    private let backend: (any ProcessBackend)?
    public init(backend: (any ProcessBackend)? = nil) { self.backend = backend }

    public func sendInput(_ text: String) {
        if let backend { backend.sendInput(text); return }
        lock.lock(); let handle = input; lock.unlock()
        try? handle?.write(contentsOf: Data((text + "\n").utf8))
    }
    public func stop() {
        if let backend { backend.stop(); return }
        lock.lock(); cancelled = true; let running = process; lock.unlock()
        guard let running, running.isRunning else { return }
        let parent = running.processIdentifier
        // Only signal this run's descendants, never other CLI sessions on the machine.
        let scan = Process(); let pipe = Pipe()
        scan.executableURL = URL(fileURLWithPath: "/bin/ps"); scan.arguments = ["-axo", "pid=,ppid="]; scan.standardOutput = pipe
        var descendants: [Int32] = []
        if (try? scan.run()) != nil {
            let rows = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).split(separator: "\n").compactMap { row -> (Int32, Int32)? in
                let parts = row.split(whereSeparator: { $0.isWhitespace }); guard parts.count == 2, let pid = Int32(parts[0]), let ppid = Int32(parts[1]) else { return nil }; return (pid, ppid)
            }
            scan.waitUntilExit()
            func walk(_ pid: Int32) { for child in rows where child.1 == pid { walk(child.0); descendants.append(child.0) } }
            walk(parent)
        }
        for pid in descendants { kill(pid, SIGTERM) }
        running.terminate()
    }

    public func run(_ command: Command, directory: URL, environment: [String: String] = [:], interactive: Bool = false, onLine: @escaping (String, Bool) -> Void = { _, _ in }) async throws -> ProcessResult {
        if let backend { return try await backend.run(command, onLine: onLine) }
        return try await withCheckedThrowingContinuation { continuation in
            DispatchQueue.global(qos: .userInitiated).async {
                let task = Process(); let out = Pipe(); let err = Pipe(); let stdin = Pipe()
                task.executableURL = URL(fileURLWithPath: command.executable)
                task.arguments = command.arguments; task.currentDirectoryURL = directory
                var env = ProcessInfo.processInfo.environment
                environment.forEach { env[$0.key] = $0.value }
                // Avoid inheriting nested-session guards from a parent coding agent.
                env.removeValue(forKey: "CLAUDECODE")
                env["NO_COLOR"] = "1"; task.environment = env
                task.standardOutput = out; task.standardError = err; task.standardInput = stdin
                self.lock.lock()
                if self.cancelled { self.lock.unlock(); continuation.resume(throwing: CancellationError()); return }
                self.process = task; self.input = stdin.fileHandleForWriting
                do { try task.run(); self.lock.unlock() }
                catch { self.process = nil; self.input = nil; self.lock.unlock(); continuation.resume(throwing: error); return }
                let group = DispatchGroup(); let dataLock = NSLock()
                var output = Data(); var errors = Data()
                for (handle, isError) in [(out.fileHandleForReading, false), (err.fileHandleForReading, true)] {
                    group.enter()
                    DispatchQueue.global(qos: .userInitiated).async {
                        var pending = Data()
                        while true {
                            let data = handle.availableData
                            if data.isEmpty { break }
                            dataLock.lock(); if isError { errors.append(data) } else { output.append(data) }; dataLock.unlock()
                            pending.append(data)
                            while let end = pending.firstIndex(of: 10) {
                                let line = String(decoding: pending[..<end], as: UTF8.self)
                                pending.removeSubrange(...end)
                                onLine(line, isError)
                            }
                        }
                        if !pending.isEmpty { onLine(String(decoding: pending, as: UTF8.self), isError) }
                        group.leave()
                    }
                }
                if let text = command.input { try? stdin.fileHandleForWriting.write(contentsOf: Data(text.utf8)) }
                if !interactive { try? stdin.fileHandleForWriting.close() }
                task.waitUntilExit(); group.wait()
                try? stdin.fileHandleForWriting.close()
                self.lock.lock(); self.process = nil; self.input = nil; self.lock.unlock()
                continuation.resume(returning: ProcessResult(code: task.terminationStatus, stdout: String(decoding: output, as: UTF8.self), stderr: String(decoding: errors, as: UTF8.self)))
            }
        }
    }
}

public enum CLIEvent: Equatable {
    case session(String)
    case message(id: String, role: String, text: String)
    case failure(String)
    case completed
}

public enum EventDecoder {
    public static func sharedContextReady(_ text: String) -> Bool {
        text.split(separator: "\n").last?.trimmingCharacters(in: .whitespacesAndNewlines) == "SHARED_CONTEXT_STATUS: ready"
    }
    public static func continuationDecision(_ text: String) -> String? {
        text.split(separator: "\n").reversed().compactMap { line -> String? in
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard trimmed.hasPrefix("RESEARCH_STATUS: ") else { return nil }
            let decision = String(trimmed.dropFirst("RESEARCH_STATUS: ".count))
            return ["continue", "needs_input", "finished"].contains(decision) ? decision : nil
        }.first
    }
    public static func decode(_ line: String, provider: Provider) -> [CLIEvent] {
        guard let data = line.data(using: .utf8), let object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] else { return [] }
        let type = object["type"] as? String ?? ""
        if provider == .codex {
            if type == "thread.started", let id = object["thread_id"] as? String { return [.session(id)] }
            if type == "turn.completed" { return [.completed] }
            if type == "error" || type == "turn.failed" {
                let error = object["error"] as? [String: Any]
                return [.failure(error?["message"] as? String ?? object["message"] as? String ?? "Codex reported a failed turn.")]
            }
            guard let item = object["item"] as? [String: Any] else { return [] }
            let id = item["id"] as? String ?? UUID().uuidString
            let kind = item["type"] as? String ?? ""
            if kind == "agent_message", let text = item["text"] as? String { return [.message(id: id, role: "assistant", text: text)] }
            if kind == "command_execution" {
                let command = item["command"] as? String ?? "Command"
                let output = item["aggregated_output"] as? String ?? ""
                let status = item["status"] as? String ?? "running"
                return [.message(id: id, role: "tool", text: "\(command)\n\(status)\(output.isEmpty ? "" : "\n" + output)")]
            }
            if kind == "file_change" || kind == "mcp_tool_call" || kind == "web_search" {
                let payload = (try? JSONSerialization.data(withJSONObject: item, options: [.prettyPrinted, .sortedKeys])).map { String(decoding: $0, as: UTF8.self) } ?? kind
                return [.message(id: id, role: "tool", text: payload)]
            }
            return []
        }
        var events: [CLIEvent] = []
        if let id = object["session_id"] as? String { events.append(.session(id)) }
        if type == "assistant", let message = object["message"] as? [String: Any], let content = message["content"] as? [[String: Any]] {
            let id = message["id"] as? String ?? UUID().uuidString
            for (index, block) in content.enumerated() {
                if let text = block["text"] as? String { events.append(.message(id: "\(id)-\(index)", role: "assistant", text: text)) }
                if block["type"] as? String == "tool_use" {
                    let input = (try? JSONSerialization.data(withJSONObject: block["input"] ?? [:], options: [.prettyPrinted, .sortedKeys])).map { String(decoding: $0, as: UTF8.self) } ?? ""
                    events.append(.message(id: block["id"] as? String ?? "\(id)-\(index)", role: "tool", text: "\(block["name"] as? String ?? "Tool")\n\(input)"))
                }
            }
        }
        if type == "user", let message = object["message"] as? [String: Any], let content = message["content"] as? [[String: Any]] {
            for block in content where block["type"] as? String == "tool_result" {
                let text = block["content"] as? String ?? String(describing: block["content"] ?? "")
                events.append(.message(id: "result-\(block["tool_use_id"] as? String ?? UUID().uuidString)", role: "tool", text: text))
            }
        }
        if type == "result" {
            if object["is_error"] as? Bool == true { events.append(.failure((object["errors"] as? [String])?.joined(separator: "\n") ?? object["result"] as? String ?? "Claude reported a failed turn.")) }
            else { events.append(.completed) }
            if let denied = object["permission_denials"] as? [[String: Any]], !denied.isEmpty {
                events.append(.failure("Claude denied \(denied.count) tool request(s). Review permission mode in Connections or continue this session in Terminal."))
            }
        }
        return events
    }
}
