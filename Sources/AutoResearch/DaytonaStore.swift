import Foundation
import AppKit
import ResearchCore

extension AppStore {
    func saveDaytona() throws {
        let root = try Workspace.directory(settings.workspaceRoot,"cloud")
        try Workspace.write(daytona,to:root.deletingLastPathComponent().appendingPathComponent("daytona.json"))
        try Workspace.write(cloudOwnedIDs,to:root.deletingLastPathComponent().appendingPathComponent("daytona-owned.json"))
    }
    func cloudClient() throws -> DaytonaClient {
        _ = try Workspace.validate(settings.workspaceRoot)
        if let daytonaClient { return daytonaClient }
        let client = try DaytonaClient(apiKey:DaytonaKeychain.read(daytona.installID),installID:daytona.installID,knownOwnedIDs:cloudOwnedIDs,apiURL:URL(string:"https://app.daytona.io/api")!,readRetryInterval:daytona.pollSeconds)
        daytonaClient = client;return client
    }
    func saveDaytonaKey(_ value: String) {
        do { try DaytonaKeychain.save(value,account:daytona.installID);daytonaClient = nil;try saveDaytona();daytonaStatus = "API key saved in Keychain. Connect to prepare cloud Codex." } catch { self.error = error.localizedDescription }
    }
    func removeDaytonaKey() {
        guard !hasActiveRuns,!daytonaBusy else { return }
        DaytonaKeychain.remove(daytona.installID);daytonaClient = nil;daytona.enabled = false;try? saveDaytona();daytonaStatus = "Key removed. Existing sandboxes are preserved."
    }
    func setCloudExecution(_ enabled: Bool) {
        guard !hasActiveRuns else { return };daytona.enabled = enabled
        if enabled { provider = .codex }
        do { try saveDaytona() } catch { self.error = error.localizedDescription }
    }
    func createCloudBox(_ client: DaytonaClient, role: String, group: UUID?) async throws -> String {
        let box = try await client.create(role:role,snapshot:daytona.snapshot,name:"autoresearch-" + UUID().uuidString.lowercased(),group:group?.uuidString,idleStopMinutes:daytona.idleStopMinutes)
        cloudOwnedIDs.insert(box.id)
        // Persist ownership before setup so an interrupted preparation is recoverable.
        try saveDaytona();return box.id
    }
    func setupCloudBox(_ id: String, client: DaytonaClient, repository: String = "", source: URL? = nil) async throws {
        try await client.start(id)
        try await client.upload(id,files:["/opt/autoresearch/bridge.py":try Data(contentsOf:resourceDirectory.appendingPathComponent("DaytonaWorker.py"))])
        let existing = try await client.call(id,action:"setup-status",payload:["id":"setup"])
        if existing["phase"] as? String == "ready" { cloudRuntime = existing["runtime"] as? [String:String];return }
        if let source {
            let handle = try FileHandle(forReadingFrom:source);defer { try? handle.close() };var offset = 0
            while let data = try handle.read(upToCount:262_144),!data.isEmpty {
                _ = try await client.call(id,action:"source-chunk",payload:["offset":offset,"data":data.base64EncodedString()]);offset += data.count
            }
        }
        let operation = "setup"
        _ = try await client.call(id,action:"setup",payload:["id":operation,"codex_package":daytona.codexPackage,"repository":repository,"source_archive":source != nil,"runtime":cloudRuntime as Any? ?? NSNull()])
        while true {
            try Task.checkCancellation()
            let status = try await client.call(id,action:"setup-status",payload:["id":operation])
            if status["phase"] as? String == "ready" { cloudRuntime = status["runtime"] as? [String:String];return }
            if status["phase"] as? String == "failed" { throw ResearchError(status["message"] as? String ?? "Cloud setup failed; sandbox preserved.") }
            try await Task.sleep(for:.seconds(daytona.pollSeconds))
        }
    }
    func ensureCloudConnection() async throws -> (DaytonaClient,String) {
        let client = try cloudClient()
        cloudOwnedIDs.formUnion(try await client.discoverOwned().map(\.id));try saveDaytona()
        let id: String
        if let existing = daytona.connectionSandboxID { id = existing;try await client.start(id) }
        else {
            id = try await createCloudBox(client,role:"connection",group:nil);daytona.connectionSandboxID = id;try saveDaytona()
        }
        try await setupCloudBox(id,client:client)
        return (client,id)
    }
    func cancelCloudLogin() { daytonaLoginCancelled = true;daytonaLoginRunner?.stop() }
    func connectDaytona(signIn: Bool) {
        guard !daytonaBusy else { return };daytonaLoginCancelled = false;daytonaBusy = true;daytonaStatus = "Preparing private cloud connection…"
        Task {
            defer { daytonaBusy = false;daytonaLoginRunner = nil }
            do {
                let (client,id) = try await ensureCloudConnection()
                if signIn {
                    daytonaLoginOutput = "Complete the official Codex device sign-in below. This connects cloud model access separately from your Daytona key.\n"
                    let runner = ProcessRunner(backend:DaytonaRunner(client:client,cloud:CloudSession(sandboxID:id,stateID:"login"),configuration:daytona,login:true))
                    daytonaLoginRunner = runner
                    let result = try await runner.run(Command("codex",["login","--device-auth"]),directory:URL(fileURLWithPath:settings.workspaceRoot),onLine:{ line,_ in Task { @MainActor in self.daytonaLoginOutput += line + "\n" } })
                    if daytonaLoginCancelled { daytonaStatus = "Cloud sign-in cancelled";return }
                    guard result.code == 0 else { throw ResearchError("Cloud sign-in ended without completion. Request a new code using Sign in when ready.") }
                }
                let auth = try await client.call(id,action:"auth-status")
                daytonaStatus = auth["authenticated"] as? Bool == true ? "Cloud Codex connected" : "Daytona connected · cloud Codex sign-in required"
            } catch { daytonaStatus = "Needs attention";self.error = error.localizedDescription }
        }
    }
    func stopCloudCompute() {
        guard !hasActiveRuns,!daytonaBusy else { return };daytonaBusy = true
        Task {
            defer { daytonaBusy = false }
            do {
                let client = try cloudClient()
                cloudOwnedIDs.formUnion(try await client.discoverOwned().map(\.id));try saveDaytona()
                for id in cloudOwnedIDs.sorted() { try await client.stop(id,archive:false) }
                daytonaStatus = "Owned compute stopped. Files and sign-in are retained."
            } catch { self.error = error.localizedDescription }
        }
    }
    func cloudSource(_ project: ResearchProject, group: UUID) async throws -> URL? {
        let source = project.localPath.map(URL.init(fileURLWithPath:)) ?? URL(fileURLWithPath:settings.workspaceRoot).appendingPathComponent("projects/" + project.repository)
        guard FileManager.default.fileExists(atPath:source.path) else {
            guard project.localPath == nil else { throw ResearchError("The local source checkout is missing.") };return nil
        }
        _ = try Workspace.validate(source.path)
        guard !activePaths.values.contains(source.resolvingSymlinksInPath().path) else { throw ResearchError("Another local run is writing this source checkout. Stop it before taking a cloud snapshot.") }
        let runner = ProcessRunner();let env = try environment()
        let index = try await runner.run(Command("/usr/bin/git",["ls-files","--stage"]),directory:source,environment:env)
        guard index.code == 0,!index.stdout.split(separator:"\n").contains(where:{$0.hasPrefix("160000 ")}) else { throw ResearchError("Cloud source snapshots require a Git checkout without submodules.") }
        let result = try await runner.run(Command("/usr/bin/git",["ls-files","--cached","--others","--exclude-standard","-z"]),directory:source,environment:env)
        guard result.code == 0 else { throw ResearchError("Choose a Git checkout for cloud research.") }
        let paths = Set(result.stdout.split(separator:"\0").map(String.init)).sorted().filter { FileManager.default.fileExists(atPath:source.appendingPathComponent($0).path) }
        for path in paths {
            let file = source.appendingPathComponent(path)
            guard file.resolvingSymlinksInPath().path.hasPrefix(source.resolvingSymlinksInPath().path + "/"),!path.split(separator:"/").contains(".git"),file.lastPathComponent != "auth.json" else { throw ResearchError("A source entry escapes the checkout or contains provider credentials. It was not uploaded: \(path)") }
            if file.hasDirectoryPath { throw ResearchError("Cloud source snapshots do not support submodule checkouts.") }
        }
        let root = try Workspace.directory(settings.workspaceRoot,"cloud/" + group.uuidString)
        let archive = root.appendingPathComponent("source.tar.gz")
        let packed = try await runner.run(Command("/usr/bin/tar",["--disable-copyfile","-czf",archive.path,"--null","-T","-"],input:paths.joined(separator:"\0") + "\0"),directory:source,environment:env)
        guard packed.code == 0 else { throw ResearchError("Could not snapshot the local checkout; original files are unchanged.") }
        try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:archive.path);return archive
    }
    func prepareCloudWorker(client: DaytonaClient, connection: String, id: String, project: ResearchProject, source: URL?) async throws {
        try await setupCloudBox(id,client:client,repository:source == nil ? project.repository : "",source:source)
        try await client.copyCloudAuthentication(from:connection,to:id)
    }
    func recoverCloudBox(_ id: String, client: DaytonaClient) async throws {
        cloudCleanup.insert(id);defer { cloudCleanup.remove(id) }
        let root = try Workspace.directory(settings.workspaceRoot,"cloud/recovery/" + id + "/" + UUID().uuidString)
        let receipt = try await client.call(id,action:"pack")
        guard let sha = receipt["sha256"] as? String,let bytes = receipt["bytes"] as? Int,let count = receipt["entries"] as? Int else { throw ResearchError("Cloud export did not return a recovery receipt.") }
        for name in ["manifest.json","checkout-state.tar.gz"] {
            let destination = root.appendingPathComponent(name);FileManager.default.createFile(atPath:destination.path,contents:nil,attributes:[.posixPermissions:0o600])
            let output = try FileHandle(forWritingTo:destination);defer { try? output.close() };var offset = 0
            while true {
                let chunk = try await client.call(id,action:"export-chunk",payload:["file":name,"offset":offset,"limit":daytona.logChunkBytes])
                guard let encoded = chunk["data"] as? String,let data = Data(base64Encoded:encoded),let next = chunk["offset"] as? Int,next == offset + data.count else { throw ResearchError("Cloud recovery download could not be verified; original sandbox was preserved.") }
                try output.write(contentsOf:data);offset = next
                if chunk["eof"] as? Bool == true { break }
                guard !data.isEmpty else { throw ResearchError("Incomplete cloud recovery download.") }
            }
            if name.hasSuffix(".gz"),offset != bytes { throw ResearchError("Cloud archive size mismatch.") }
        }
        let verified = try DaytonaArchive.verify(archive:root.appendingPathComponent("checkout-state.tar.gz"),manifest:root.appendingPathComponent("manifest.json"),expectedSHA:sha)
        guard count == verified else { throw ResearchError("Cloud recovery entry count mismatch.") }
        try Workspace.write(["sandbox":id,"sha256":sha,"verified_entries":String(verified),"credentials_excluded":"true"],to:root.appendingPathComponent("verified.json"))
        try await client.stop(id,archive:true)
    }
}
