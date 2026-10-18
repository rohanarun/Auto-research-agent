import Foundation

/// Polls an idempotently named remote operation; an uncertain HTTP response never
/// causes a second model invocation. Each runner owns only its operation.
public final class DaytonaRunner: ProcessBackend, @unchecked Sendable {
    let client: DaytonaClient
    let cloud: CloudSession
    let configuration: DaytonaConfiguration
    let login: Bool
    let operationID: String?
    let resumeExisting: Bool
    private let lock = NSLock()
    private var cancelled = false
    public init(client: DaytonaClient, cloud: CloudSession, configuration: DaytonaConfiguration, login: Bool = false, operationID: String? = nil, resumeExisting: Bool = false) {
        self.client = client;self.cloud = cloud;self.configuration = configuration;self.login = login
        self.operationID = operationID; self.resumeExisting = resumeExisting
    }
    public func stop() { lock.lock();cancelled = true;lock.unlock() }
    public func sendInput(_ text: String) {} // Device login is completed in the browser.
    private func stopping() -> Bool { lock.lock();defer { lock.unlock() };return cancelled }
    public func run(_ command: Command, onLine: @escaping (String, Bool) -> Void) async throws -> ProcessResult {
        if stopping() || Task.isCancelled { throw CancellationError() }
        let operation = operationID ?? UUID().uuidString
        let methods = command.executable == "autoresearch-methods"
        var args = command.arguments
        if !login && !methods { args = ["-c", "model_reasoning_effort=" + DaytonaClient.quote(configuration.reasoningEffort), "-c", "features.memories=false", "-c", "features.multi_agent=false"] + args }
        guard !resumeExisting || (methods && operationID != nil) else { throw ResearchError("Only a saved method operation can be reattached.") }
        do {
        if !resumeExisting {
        _ = try await client.call(cloud.sandboxID, action: "launch", payload: ["id":operation,"state_id":cloud.stateID,"previous_state_id":cloud.previousStateID as Any? ?? NSNull(),"login":login,"arguments":args,"input":command.input ?? "", "research_methods": methods])
        }
        var offsets = [0,0];var raw = [Data(),Data()];var pending = [Data(),Data()];var stopSent = false
        var methodCursor: [String: Any] = [:]
        var telemetryError: String?
        while true {
            if (stopping() || Task.isCancelled) && !stopSent {
                let receipt = try await client.call(cloud.sandboxID,action:"stop",payload:["id":operation])
                if let cleanup = receipt["cleanup"] as? [String:Any],let remaining = cleanup["remaining_pids"] as? [Int],!remaining.isEmpty { throw ResearchError("Cloud cancellation did not drain every owned process. The sandbox and evidence were preserved.") }
                stopSent = true
            }
            var payload: [String: Any] = ["id":operation,"stdout_offset":offsets[0],"stderr_offset":offsets[1],"limit":configuration.logChunkBytes]
            if methods { payload["method_cursor"] = methodCursor }
            let result = try await client.call(cloud.sandboxID, action:"poll",payload:payload)
            if methods {
                let warning = result["method_error"] as? String
                if let warning, warning != telemetryError { onLine(warning, true) }
                telemetryError = warning
                for event in result["method_events"] as? [[String: Any]] ?? [] {
                    let bytes = try JSONSerialization.data(withJSONObject: event)
                    onLine(String(decoding: bytes, as: UTF8.self), false)
                }
                methodCursor = result["method_cursor"] as? [String: Any] ?? methodCursor
            }
            for (i,name) in ["stdout","stderr"].enumerated() {
                guard let encoded = result[name] as? String,let data = Data(base64Encoded:encoded),let next = result[name + "_offset"] as? Int,next == offsets[i] + data.count else { throw ResearchError("Cloud log offsets could not be reconciled. Inspect the retained operation before continuing.") }
                offsets[i] = next;raw[i].append(data);pending[i].append(data)
                while let end = pending[i].firstIndex(of:10) { onLine(String(decoding:pending[i][..<end],as:UTF8.self),i == 1);pending[i].removeSubrange(...end) }
            }
            let status = result["status"] as? [String:Any] ?? [:]
            if ["ended","failed","not_started"].contains(status["phase"] as? String ?? ""),result["stdout_eof"] as? Bool == true,result["stderr_eof"] as? Bool == true, (!methods || result["method_eof"] as? Bool == true) {
                for i in 0...1 where !pending[i].isEmpty { onLine(String(decoding:pending[i],as:UTF8.self),i == 1) }
                if status["phase"] as? String == "failed" { throw ResearchError(status["message"] as? String ?? "The cloud process failed; its evidence is retained.") }
                return ProcessResult(code:Int32(status["exit_code"] as? Int ?? 1),stdout:String(decoding:raw[0],as:UTF8.self),stderr:String(decoding:raw[1],as:UTF8.self))
            }
            // Cancellation is handled above so the owned process is drained first.
            try? await Task.sleep(for:.seconds(configuration.pollSeconds))
        }
        } catch {
            // Observation errors must not kill a detached research run. The saved
            // attachment can resume polling the same operation without relaunching it.
            if methods && !stopping() && !Task.isCancelled { throw error }
            // The launch receipt may have been lost after remote acceptance.
            // Stop that same operation even when local polling fails; never
            // create a replacement or leave cleanup assuming it was drained.
            do {
                let receipt = try await client.call(cloud.sandboxID,action:"stop",payload:["id":operation])
                guard ["ended","failed","not_started"].contains(receipt["phase"] as? String ?? ""),let cleanup = receipt["cleanup"] as? [String:Any],let remaining = cleanup["remaining_pids"] as? [Int],remaining.isEmpty else {
                    throw ResearchError("Cloud process drain was not confirmed. Owned sandbox \(cloud.sandboxID), operation \(operation), remains preserved.")
                }
            } catch let drainError {
                throw ResearchError("\(error.localizedDescription) \(drainError.localizedDescription)")
            }
            throw error
        }
    }
}
