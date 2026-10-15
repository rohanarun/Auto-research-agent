import Foundation

public struct DaytonaConfiguration: Codable, Equatable, Sendable {
    public var enabled = false
    public var installID = UUID().uuidString
    public var connectionSandboxID: String?
    public var snapshot: String
    public var codexPackage: String
    public var reasoningEffort: String
    public var pollSeconds: Double
    public var logChunkBytes: Int
    public var idleStopMinutes = 60
    public init(snapshot: String, codexPackage: String, reasoningEffort: String, pollSeconds: Double, logChunkBytes: Int) {
        self.snapshot = snapshot;self.codexPackage = codexPackage;self.reasoningEffort = reasoningEffort
        self.pollSeconds = pollSeconds;self.logChunkBytes = logChunkBytes
    }
    private enum CodingKeys: String, CodingKey { case enabled,installID,connectionSandboxID,snapshot,codexPackage,reasoningEffort,pollSeconds,logChunkBytes,idleStopMinutes }
    public init(from decoder: Decoder) throws {
        let d = try decoder.container(keyedBy:CodingKeys.self)
        enabled = try d.decodeIfPresent(Bool.self,forKey:.enabled) ?? false
        installID = try d.decodeIfPresent(String.self,forKey:.installID) ?? UUID().uuidString
        connectionSandboxID = try d.decodeIfPresent(String.self,forKey:.connectionSandboxID)
        snapshot = try d.decode(String.self,forKey:.snapshot);codexPackage = try d.decode(String.self,forKey:.codexPackage)
        reasoningEffort = try d.decode(String.self,forKey:.reasoningEffort);pollSeconds = try d.decode(Double.self,forKey:.pollSeconds)
        logChunkBytes = try d.decode(Int.self,forKey:.logChunkBytes)
        idleStopMinutes = try d.decodeIfPresent(Int.self,forKey:.idleStopMinutes) ?? 60
        guard pollSeconds.isFinite,pollSeconds > 0,logChunkBytes > 0,idleStopMinutes > 0 else { throw ResearchError("Invalid Daytona polling configuration.") }
    }
}

public struct DaytonaSandbox: Codable, Sendable {
    public let id: String
    public let name: String?
    public let state: String
    public let cpu: Int?
    public let memory: Int?
    public let disk: Int?
    public let labels: [String: String]?
}

public struct CloudSession: Codable, Sendable {
    public var sandboxID: String
    public var stateID: String
    public var previousStateID: String?
    public init(sandboxID: String, stateID: String, previousStateID: String? = nil) {
        self.sandboxID = sandboxID;self.stateID = stateID;self.previousStateID = previousStateID
    }
}

/// API credentials are confined to the management origin. Sandbox control uses
/// the sandbox's separate preview credential, kept only in memory.
public actor DaytonaClient {
    private let key: String
    private let session: URLSession
    private let apiURL: URL
    private let installID: String
    private let readRetryInterval: TimeInterval
    private var owned: Set<String>
    private var previews: [String: Preview] = [:]
    private struct Preview: Decodable { let url: URL;let token: String }
    private struct Execution: Decodable { let exitCode: Int;let result: String }
    private final class RedirectPolicy: NSObject, URLSessionTaskDelegate, @unchecked Sendable {
        let host: String?
        let requiredPrefix: String?
        init(host: String?,requiredPrefix: String?) { self.host = host;self.requiredPrefix = requiredPrefix }
        func urlSession(_ session: URLSession,task: URLSessionTask,willPerformHTTPRedirection response: HTTPURLResponse,newRequest request: URLRequest,completionHandler: @escaping (URLRequest?) -> Void) {
            completionHandler(request.url?.scheme == "https" && request.url?.host == host && (requiredPrefix.map { request.url?.path.hasPrefix($0) == true } ?? true) ? request : nil)
        }
    }
    public init(apiKey: String, installID: String, knownOwnedIDs: Set<String> = [], apiURL: URL, session: URLSession = .shared, readRetryInterval: TimeInterval = 2) throws {
        guard apiURL.scheme == "https",apiURL.host == "app.daytona.io",apiURL.path == "/api",!apiKey.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ResearchError("Enter a Daytona API key for the official Daytona service.")
        }
        guard readRetryInterval.isFinite,readRetryInterval > 0 else { throw ResearchError("Invalid cloud read recovery interval.") }
        self.key = apiKey;self.installID = installID;self.owned = knownOwnedIDs;self.apiURL = apiURL;self.session = session;self.readRetryInterval = readRetryInterval
    }
    private func request(_ url: URL, method: String, body: [String: Any]?, management: Bool, token: String? = nil) async throws -> Data {
        var req = URLRequest(url: url);req.httpMethod = method;req.timeoutInterval = 90
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if management {
            guard url.host == apiURL.host,url.scheme == "https",url.path.hasPrefix(apiURL.path + "/") else { throw ResearchError("Refusing to send Daytona credentials outside its API.") }
            req.setValue("Bearer " + key, forHTTPHeaderField: "Authorization")
        } else if let token { req.setValue(token, forHTTPHeaderField: "X-Daytona-Preview-Token") }
        if let body { req.httpBody = try JSONSerialization.data(withJSONObject: body) }
        // A failed observation must not abandon a still-running model. Retry
        // only management GETs within the existing request timeout. Mutating
        // requests retain their original uncertainty/reconciliation behavior.
        let readDeadline = Date().addingTimeInterval(req.timeoutInterval)
        while true {
            do {
                if management,method == "GET" {
                    let remaining = readDeadline.timeIntervalSinceNow
                    guard remaining > 0 else { throw ResearchError("Daytona management reads remained unavailable through the request timeout; remote work is preserved for reconciliation.") }
                    req.timeoutInterval = remaining
                }
                let (data,response) = try await session.data(for: req,delegate:RedirectPolicy(host:url.host,requiredPrefix:management ? apiURL.path + "/" : nil))
                let status = (response as? HTTPURLResponse)?.statusCode ?? 0
                if management,method == "GET",[408,429,500,502,503,504].contains(status),Date() < readDeadline {
                    let serverDelay = (response as? HTTPURLResponse)?.value(forHTTPHeaderField:"Retry-After").flatMap(Double.init)
                    let delay = max(readRetryInterval,serverDelay ?? readRetryInterval)
                    try await Task.sleep(for:.seconds(min(delay,max(0,readDeadline.timeIntervalSinceNow))));continue
                }
                guard (200..<300).contains(status) else {
                    throw ResearchError("Daytona returned HTTP \(status) for \(management ? "management" : "sandbox control"). Check the connection, quota and sandbox state; existing work is preserved.")
                }
                return data
            } catch let error as URLError where management && method == "GET" && [.timedOut,.networkConnectionLost,.cannotConnectToHost,.cannotFindHost,.notConnectedToInternet].contains(error.code) && Date() < readDeadline {
                try await Task.sleep(for:.seconds(min(readRetryInterval,max(0,readDeadline.timeIntervalSinceNow))))
            }
        }
    }
    private func api(_ path: String, method: String = "GET", body: [String: Any]? = nil) async throws -> Data {
        guard path.hasPrefix("/"),!path.contains("..") else { throw ResearchError("Invalid Daytona API path.") }
        return try await request(URL(string: apiURL.absoluteString + path)!, method: method, body: body, management: true)
    }
    public func discoverOwned() async throws -> [DaytonaSandbox] {
        struct Page: Decodable { let items: [DaytonaSandbox];let nextCursor: String? }
        var boxes: [DaytonaSandbox] = [];var cursor: String?;var seen = Set<String>()
        repeat {
            var query = URLComponents();query.queryItems = [URLQueryItem(name:"limit",value:"100")]
            if let cursor { query.queryItems!.append(URLQueryItem(name:"cursor",value:cursor)) }
            let page = try JSONDecoder().decode(Page.self,from:await api("/sandbox?" + (query.percentEncodedQuery ?? "")))
            for box in page.items where box.labels?["autoresearch-install"] == installID { owned.insert(box.id);boxes.append(box) }
            cursor = page.nextCursor
            if let cursor,!seen.insert(cursor).inserted { throw ResearchError("Daytona pagination repeated; existing sandboxes were preserved.") }
        } while cursor != nil && cursor != ""
        return boxes
    }
    public func create(role: String, snapshot: String, name: String, group: String? = nil, idleStopMinutes: Int = 60) async throws -> DaytonaSandbox {
        var labels = ["autoresearch-install":installID,"autoresearch-role":role]
        if let group { labels["autoresearch-group"] = group }
        do {
            let data = try await api("/sandbox",method: "POST",body: ["name":name,"snapshot":snapshot,"public":false,"autoStopInterval":idleStopMinutes,"autoPauseInterval":0,"autoDeleteInterval":-1,"labels":labels])
            let box = try JSONDecoder().decode(DaytonaSandbox.self,from:data);owned.insert(box.id);return box
        } catch {
            // Reconcile an uncertain POST by its unique name, never repeat it.
            let matches = (try? await discoverOwned())?.filter { $0.name == name } ?? []
            if matches.count == 1 { return matches[0] }
            throw ResearchError("Sandbox creation was not confirmed. Connect or Stop owned cloud compute to reconcile this installation's sandboxes before retrying. No model invocation was repeated.")
        }
    }

    public func inspect(_ id: String) async throws -> DaytonaSandbox {
        guard UUID(uuidString:id) != nil,owned.contains(id) else { throw ResearchError("This sandbox is not owned by this app installation.") }
        let box = try JSONDecoder().decode(DaytonaSandbox.self,from: await api("/sandbox/" + id))
        guard box.id == id,box.labels?["autoresearch-install"] == installID else { throw ResearchError("The sandbox ownership label does not match this installation. It was left untouched.") }
        return box
    }
    public func start(_ id: String) async throws {
        previews.removeValue(forKey:id)
        var box = try await inspect(id)
        while ["stopping","archiving"].contains(box.state) { try await Task.sleep(for: .seconds(1));box = try await inspect(id) }
        if ["stopped","archived"].contains(box.state) { _ = try await api("/sandbox/" + id + "/start",method:"POST");previews.removeValue(forKey:id) }
        while box.state != "started" {
            try Task.checkCancellation();try await Task.sleep(for:.seconds(1));box = try await inspect(id)
            if ["error","deleted","destroyed","build_failed"].contains(box.state) { throw ResearchError("The owned sandbox could not start: \(box.state).") }
        }
    }
    public func stop(_ id: String, archive: Bool) async throws {
        var box = try await inspect(id)
        if box.state == "archived" { return }
        while ["starting","restoring","pending_build","building"].contains(box.state) { try await Task.sleep(for:.seconds(1));box = try await inspect(id) }
        if box.state == "started" { _ = try await api("/sandbox/" + id + "/stop",method:"POST") }
        while !["stopped","archived"].contains(box.state) {
            if ["error","deleted","destroyed"].contains(box.state) { throw ResearchError("Could not confirm sandbox cleanup: \(box.state).") }
            try await Task.sleep(for:.seconds(1));box = try await inspect(id)
        }
        previews.removeValue(forKey:id)
        if archive && box.state != "archived" {
            _ = try await api("/sandbox/" + id + "/archive",method:"POST")
            while box.state != "archived" {
                try await Task.sleep(for:.seconds(1));box = try await inspect(id)
                if ["error","deleted","destroyed","build_failed"].contains(box.state) { throw ResearchError("Daytona archive could not be confirmed: \(box.state). The verified local recovery export is retained.") }
            }
        }
    }
    public func execute(_ id: String, command: String) async throws -> String {
        _ = try await inspect(id)
        if previews[id] == nil { previews[id] = try JSONDecoder().decode(Preview.self,from: await api("/sandbox/" + id + "/ports/2280/preview-url")) }
        let preview = previews[id]!
        guard preview.url.scheme == "https",preview.url.user == nil,preview.url.password == nil else { throw ResearchError("Daytona returned an invalid private control URL.") }
        let url = preview.url.appendingPathComponent("process/execute")
        let data = try await request(url,method:"POST",body:["command":command,"timeout":60],management:false,token:preview.token)
        let result = try JSONDecoder().decode(Execution.self,from:data)
        guard result.exitCode == 0 else { throw ResearchError("A sandbox control command failed. Inspect this run's retained diagnostics in Daytona; no successful work was repeated.") }
        return result.result
    }
    public func copyCloudAuthentication(from source: String, to target: String) async throws {
        _ = try await inspect(source); _ = try await inspect(target)
        if previews[target] == nil { previews[target] = try JSONDecoder().decode(Preview.self, from: await api("/sandbox/" + target + "/ports/2280/preview-url")) }
        let p = previews[target]!
        let receipt = try await call(source, action: "fork-auth", payload: ["url":p.url.absoluteString,"token":p.token])
        guard receipt["copied"] as? Bool == true else { throw ResearchError("Cloud sign-in could not be transferred to the owned worker.") }
    }
    public func upload(_ id: String, files: [String: Data]) async throws {
        let payload = files.mapValues { $0.base64EncodedString() }
        let code = "import sys,json,pathlib,base64,os;d=json.load(sys.stdin)\nfor name,value in d.items():\n p=pathlib.Path(name);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_name(p.name+'.incoming');q.write_bytes(base64.b64decode(value));q.chmod(0o600);q.replace(p)\nprint('uploaded')"
        let data = try JSONSerialization.data(withJSONObject:payload).base64EncodedString()
        _ = try await execute(id,command:"printf %s " + Self.quote(data) + " | base64 -d | sudo -n python3 -c " + Self.quote(code))
    }
    public func call(_ id: String, action: String, payload: [String: Any] = [:]) async throws -> [String: Any] {
        guard action.allSatisfy({ $0.isLetter || $0 == "-" }) else { throw ResearchError("Invalid sandbox action.") }
        let encoded = try JSONSerialization.data(withJSONObject:payload).base64EncodedString()
        let result = try await execute(id,command:"printf %s " + Self.quote(encoded) + " | base64 -d | sudo -n python3 /opt/autoresearch/bridge.py " + action)
        guard let object = try JSONSerialization.jsonObject(with:Data(result.utf8)) as? [String:Any] else { throw ResearchError("Unreadable sandbox response; reconcile its saved run before retrying.") }
        return object
    }
    public static func quote(_ value: String) -> String { "'" + value.replacingOccurrences(of:"'",with:"'\\''") + "'" }
}
