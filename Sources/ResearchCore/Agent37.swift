import Foundation

struct Agent37OutputTruncated: LocalizedError {
    var errorDescription: String? { "Agent37 control output was truncated. The retained operation was not repeated." }
}

extension DaytonaClient {
    private struct Instance: Decodable {
        let id: String
        let name: String?
        let status: String
        let metadata: [String: String]?
        let resources: Resources?
        private enum CodingKeys: String, CodingKey { case id,name,status,metadata,resources }
        init(from decoder: Decoder) throws {
            let values = try decoder.container(keyedBy:CodingKeys.self)
            id = try values.decode(String.self,forKey:.id);status = try values.decode(String.self,forKey:.status)
            name = try values.decodeIfPresent(String.self,forKey:.name)
            // Other apps may store arbitrary JSON metadata in this workspace.
            // Unreadable ownership metadata can never authorize an operation.
            metadata = try? values.decode([String:String].self,forKey:.metadata)
            resources = try values.decodeIfPresent(Resources.self,forKey:.resources)
        }
        struct Resources: Decodable { let cpu: Int?; let memory: Int?; let disk: Int? }
        var sandbox: DaytonaSandbox { DaytonaSandbox(id:id,name:name,state:status,cpu:resources?.cpu,memory:resources?.memory,disk:resources?.disk,labels:metadata) }
    }
    private func validAgent37ID(_ id: String) -> Bool {
        id.range(of:"^[a-z0-9]{10}$",options:.regularExpression) != nil
    }
    func agent37Discover() async throws -> [DaytonaSandbox] {
        struct Page: Decodable { let data: [Instance] }
        let instances = try JSONDecoder().decode(Page.self,from:await api("/instances")).data
        let boxes = instances.filter { $0.metadata?["autoresearch-install"] == installID && validAgent37ID($0.id) }.map(\.sandbox)
        owned.formUnion(boxes.map(\.id));return boxes
    }
    func agent37Create(role: String, template: String, name: String, group: String?, idleStopMinutes: Int) async throws -> DaytonaSandbox {
        guard !template.isEmpty,(5...1440).contains(idleStopMinutes) else { throw ResearchError("Agent37 requires a template and an idle timeout from 5 to 1440 minutes.") }
        var metadata = ["autoresearch-install":installID,"autoresearch-role":role]
        if let group { metadata["autoresearch-group"] = group }
        do {
            // All durable bridge, auth and checkout files live under this mount.
            let body: [String:Any] = ["template":template,"name":name,"metadata":metadata,"mount_path":"/workspace","auto_sleep":true,"idle_timeout_seconds":idleStopMinutes * 60,"public_ports":[Any]()]
            let box = try JSONDecoder().decode(Instance.self,from:await api("/instances",method:"POST",body:body)).sandbox
            guard validAgent37ID(box.id),box.labels?["autoresearch-install"] == installID else { throw ResearchError("Agent37 creation returned an invalid ownership receipt.") }
            owned.insert(box.id);return box
        } catch {
            let matches = (try? await agent37Discover())?.filter { $0.name == name } ?? []
            if matches.count == 1 { return matches[0] }
            throw ResearchError("Agent37 creation was not confirmed. Connect to reconcile owned instances before retrying. The create request was not repeated.")
        }
    }
    func agent37Inspect(_ id: String) async throws -> DaytonaSandbox {
        guard validAgent37ID(id),owned.contains(id) else { throw ResearchError("This Agent37 instance is not owned by this installation.") }
        let box = try JSONDecoder().decode(Instance.self,from:await api("/instances/" + id)).sandbox
        guard box.id == id,box.labels?["autoresearch-install"] == installID else { throw ResearchError("Agent37 ownership metadata changed; the instance was left untouched.") }
        return box
    }
    private func waitForAgent37(_ id: String, states: Set<String>) async throws {
        let deadline = Date().addingTimeInterval(300)
        while true {
            try Task.checkCancellation()
            let box = try await agent37Inspect(id)
            if states.contains(box.state) { return }
            guard !["failed","deleted","deleting"].contains(box.state),Date() < deadline else { throw ResearchError("Agent37 lifecycle did not reach the requested state. The instance and its files are preserved.") }
            try await Task.sleep(for:.seconds(readRetryInterval))
        }
    }
    func agent37Start(_ id: String) async throws {
        var box = try await agent37Inspect(id)
        if box.state == "stopping" { try await waitForAgent37(id,states:["stopped"]);box = try await agent37Inspect(id) }
        if ["stopped","sleeping"].contains(box.state) { _ = try await api("/instances/" + id + "/start",method:"POST") }
        try await waitForAgent37(id,states:["running"])
        // Agent37 rebuilds the container on start. Restore only links into the
        // persistent mount; refuse to replace any existing directory or file.
        let bootstrap = """
        import pathlib
        root=pathlib.Path('/workspace/.autoresearch');root.mkdir(mode=0o700,parents=True,exist_ok=True);root.chmod(0o700)
        for name,target in [('/research-private',root/'private'),('/opt/autoresearch',root/'tools')]:
         target.mkdir(mode=0o700,parents=True,exist_ok=True)
         p=pathlib.Path(name);p.parent.mkdir(parents=True,exist_ok=True)
         if p.is_symlink(): assert p.resolve()==target.resolve(),'Unexpected cloud storage link'
         elif p.exists(): raise RuntimeError('Refusing to replace existing cloud storage')
         else: p.symlink_to(target,target_is_directory=True)
        print('ready')
        """
        _ = try await agent37Execute(id,command:"python3 -c " + Self.quote(bootstrap))
    }
    func agent37Stop(_ id: String) async throws {
        let box = try await agent37Inspect(id)
        if box.state == "stopped" { return }
        if !["running","sleeping"].contains(box.state) { try await waitForAgent37(id,states:["running","sleeping","stopped"]) }
        if try await agent37Inspect(id).state != "stopped" { _ = try await api("/instances/" + id + "/stop",method:"POST") }
        // Agent37 has no archive operation. Retain its disk after verified export.
        try await waitForAgent37(id,states:["stopped"])
    }
    func agent37Execute(_ id: String, command: String) async throws -> String {
        _ = try await agent37Inspect(id)
        struct Execution: Decodable { let exit_code: Int;let stdout: String;let stderr: String;let truncated: Bool }
        // The documented control-plane exec endpoint confines the workspace key
        // to api.agent37.com. Detached bridge operations keep each call short.
        let result = try JSONDecoder().decode(Execution.self,from:await api("/instances/" + id + "/exec",method:"POST",body:["command":command,"user":"root"]))
        if result.truncated { throw Agent37OutputTruncated() }
        guard result.exit_code == 0 else { throw ResearchError("Agent37 control command failed or its output was truncated. Inspect retained diagnostics; no command was repeated.") }
        return result.stdout
    }
    func agent37CopyAuthentication(from source: String, to target: String) async throws {
        _ = try await agent37Inspect(source);_ = try await agent37Inspect(target)
        let token = UUID().uuidString
        let ready = try await call(source,action:"prepare-auth-transfer",payload:["token":token])
        guard let port = ready["port"] as? Int else { throw ResearchError("Cloud authentication transfer did not start.") }
        struct Signed: Decodable { let url: URL }
        let signed = try JSONDecoder().decode(Signed.self,from:await api("/instances/" + source + "/signed-url",method:"POST",body:["port":port,"ttl_seconds":60]))
        guard signed.url.scheme == "https",signed.url.host == source + "-" + String(port) + ".agent37.app",signed.url.user == nil,signed.url.password == nil else { throw ResearchError("Invalid Agent37 authentication transfer URL.") }
        let receipt = try await call(target,action:"receive-auth-transfer",payload:["url":signed.url.absoluteString,"token":token])
        guard receipt["copied"] as? Bool == true else { throw ResearchError("Cloud authentication transfer was not confirmed.") }
    }
}
