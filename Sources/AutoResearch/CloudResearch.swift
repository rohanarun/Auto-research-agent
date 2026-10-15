import Foundation
import ResearchCore

extension AppStore {
    func launchCloud(_ id: UUID, project: ResearchProject, request: String?) {
        guard runners[id] == nil,let initial = sessions.first(where:{$0.id == id}) else { return }
        guard initial.cloud.map({ !cloudCleanup.contains($0.sandboxID) }) ?? true else { error = "Wait for cloud recovery export to finish before resuming this chat.";return }
        guard initial.provider == .codex else { error = "Daytona currently runs Codex. Select Codex or switch execution back to This Mac.";return }
        cloudDrains[id] = false;runners[id] = ProcessRunner();stopped.remove(id);failed.remove(id);completed.remove(id);eventMessages[id] = [:];state(id,"Preparing")
        workerTasks[id] = Task {
            var client: DaytonaClient?;var boxID: String?;var nextDecision: String?
            do {
                let c = try cloudClient();client = c
                var cloud = initial.cloud
                if cloud == nil {
                    let (connected,connection) = try await ensureCloudConnection()
                    let auth = try await connected.call(connection,action:"auth-status")
                    guard auth["authenticated"] as? Bool == true else { throw ResearchError("Sign in to cloud Codex in Connections & settings → Daytona first.") }
                    let source = try await cloudSource(project,group:id)
                    let box = try await createCloudBox(c,role:"worker",group:id);boxID = box
                    try await prepareCloudWorker(client:c,connection:connection,id:box,project:project,source:source)
                    cloud = CloudSession(sandboxID:box,stateID:id.uuidString)
                    if let i = sessions.firstIndex(where:{$0.id == id}) { sessions[i].cloud = cloud;persist() }
                }
                guard let cloud else { throw ResearchError("Missing cloud worker.") };boxID = cloud.sandboxID
                try await c.start(cloud.sandboxID)
                guard !stopped.contains(id) else { throw CancellationError() }
                let runner = ProcessRunner(backend:DaytonaRunner(client:c,cloud:cloud,configuration:daytona));runners[id] = runner
                var prompt = request ?? "Start an autonomous research run in your assigned /workspace/checkout. Read applicable AGENTS.md and auto-research.md. If that entry point is absent, locate the repository's actual research guide, explain the source you choose, and follow it. Ask for input if no actionable objective exists."
                if let shared = initial.sharedAgent {
                    prompt += "\nAgent \(shared.index) of \(shared.agentCount ?? 1), generation \(shared.generation), in your own private checkout."
                    if !shared.isIndependent,let path = shared.contextFile {
                        prompt += "\n" + (try SharedContextFiles.cloudInjection(file:URL(fileURLWithPath:path),remoteRoot:"/workspace/checkout/SHARED_CONTEXT"))
                    }
                }
                let extra = initial.sharedAgent == nil ? "" : try String(contentsOf:resourceDirectory.appendingPathComponent(initial.sharedAgent?.isIndependent == true ? "IndependentAgentPolicy.md" : "SharedAgentPolicy.md"),encoding:.utf8)
                let cloudPolicy = policy + "\n\n" + extra + "\n\nExecution environment: your private Linux Daytona sandbox, assigned checkout /workspace/checkout. All commands and generated files stay in this checkout. The external-SSD rule applies to the native Mac controller, whose mount was validated there; do not look for /Volumes inside Linux. No host account files, prior conversations, peer checkouts or host MCP configuration are supplied."
                var cloudSettings = settings;cloudSettings.codexPath = "/opt/autoresearch/bin/codex"
                let command = CLI.chat(provider:.codex,settings:cloudSettings,sessionID:initial.providerSessionID,prompt:prompt,policy:cloudPolicy)
                append(id,role:"user",text:prompt)
                let root = try Workspace.directory(settings.workspaceRoot,"sessions/" + id.uuidString)
                let logURL = root.appendingPathComponent(UUID().uuidString + ".jsonl")
                FileManager.default.createFile(atPath:logURL.path,contents:nil,attributes:[.posixPermissions:0o600])
                let log = try FileHandle(forWritingTo:logURL);let lock = NSLock();defer { try? log.close() }
                state(id,"Running")
                let result = try await runner.run(command,directory:root,onLine:{ line,stderr in
                    lock.lock();try? log.write(contentsOf:Data((line + "\n").utf8));lock.unlock()
                    Task { @MainActor in self.consume(line,stderr:stderr,provider:.codex,sessionID:id) }
                })
                cloudDrains[id] = true
                for line in result.stdout.split(separator:"\n") {
                    consume(String(line),stderr:false,provider:.codex,sessionID:id)
                    for event in EventDecoder.decode(String(line),provider:.codex) {
                        if case .message(_,"assistant",let text) = event { nextDecision = EventDecoder.continuationDecision(text) }
                    }
                }
                if stopped.contains(id) { state(id,"Stopped") }
                else if result.code != 0 || failed.contains(id) || !completed.contains(id) { state(id,"Needs attention");append(id,role:"error",text:"Cloud CLI did not complete successfully (\(result.code)). Its files and logs are preserved.") }
                else { state(id,"Completed") }
            } catch {
                state(id,error is CancellationError ? "Stopped" : "Needs attention")
                if !(error is CancellationError) { append(id,role:"error",text:error.localizedDescription + (stopped.contains(id) ? " Cloud process drain was not confirmed; no successor may be started." : "")) }
            }
            runners.removeValue(forKey:id);persist()
            let finished = sessions.first(where:{$0.id == id})
            if finished?.sharedAgent != nil {
                if finished?.state == "Needs attention",finished?.sharedAgent?.isIndependent != true { sharedCycles[project.id]?.stop(failure:"A cloud agent needs attention. Its sandbox was preserved.") }
            } else if keepResearching.contains(id),finished?.state == "Completed",nextDecision == "continue" {
                let next = (try? String(contentsOf:resourceDirectory.appendingPathComponent("ContinuePolicy.md"),encoding:.utf8)) ?? ""
                if !next.isEmpty { launchCloud(id,project:project,request:next);return }
            } else if let client,let boxID {
                keepResearching.remove(id)
                await cleanupCloudWorkers([boxID],client:client)
            }
        }
    }
    func cleanupCloudWorkers(_ ids: [String],client: DaytonaClient) async {
        // Run outside cancelled rotation tasks so cancellation still saves and stops compute.
        await withTaskGroup(of:Void.self) { tasks in
            for id in Set(ids) { tasks.addTask { await self.cleanupCloudWorker(id,client:client) } }
            await tasks.waitForAll()
        }
    }
    private func cleanupCloudWorker(_ id: String,client: DaytonaClient) async {
        do { try await recoverCloudBox(id,client:client) }
        catch {
            let message = error.localizedDescription
            // A failed export never destroys work. Stop compute, preserve filesystem.
            do { try await client.stop(id,archive:false) } catch { self.error = "Cloud cleanup needs attention: " + error.localizedDescription }
            self.error = "Cloud files retained in sandbox \(id). Recovery export needs attention: " + message
        }
    }

    func startCloudResearch(_ project: ResearchProject,request: String?) {
        guard !running(project),provider == .codex else { error = "Stop existing research and select Codex before starting Daytona agents.";return }
        let options = sharedConfiguration(project.id)
        do { try options.validate() } catch { self.error = error.localizedDescription;return }
        let group = UUID();var boxes: [String] = [];var client: DaytonaClient?
        sharedRuns[project.id] = SharedRunStatus(groupID:group,sharesContext:options.enabled)
        let hooks = SharedContextCycle.Hooks(prepare: {
            let (c,connection) = try await self.ensureCloudConnection();client = c
            let auth = try await c.call(connection,action:"auth-status")
            guard auth["authenticated"] as? Bool == true else { throw ResearchError("Sign in to cloud Codex in Connections & settings → Daytona first.") }
            let source = try await self.cloudSource(project,group:group)
            for _ in 0..<options.agentCount { try Task.checkCancellation();boxes.append(try await self.createCloudBox(c,role:"worker",group:group)) }
            try await withThrowingTaskGroup(of:Void.self) { tasks in
                for box in boxes { tasks.addTask { try await self.prepareCloudWorker(client:c,connection:connection,id:box,project:project,source:source) } }
                try await tasks.waitForAll()
            }
        },launch: { generation,context,previous in
            var ids: [UUID] = []
            for (offset,box) in boxes.enumerated() {
                var session = ResearchSession(projectID:project.id,title:"Agent \(offset + 1) · \(options.enabled ? "round \(generation)" : "independent") · Daytona",provider:.codex)
                let old = previous.indices.contains(offset) ? previous[offset] : nil
                session.cloud = CloudSession(sandboxID:box,stateID:session.id.uuidString,previousStateID:old?.uuidString)
                session.sharedAgent = SharedAgent(groupID:group,index:offset + 1,generation:generation,previousSessionID:old,contextFile:context?.path,sharesContext:options.enabled,agentCount:options.agentCount)
                session.messages.append(ChatMessage(role:"status",text:"Private Daytona sandbox \(box). Fresh provider state; only cloud sign-in and this group's published checkpoints are supplied."))
                self.sessions.append(session);ids.append(session.id)
            }
            guard self.persist() else { throw ResearchError("Could not save cloud session lineage. No replacement agents were started.") }
            for id in ids { self.launchCloud(id,project:project,request:request) };return ids
        },drain: { ids in
            let tasks = ids.compactMap { self.workerTasks[$0] }
            for id in ids { self.stopWorker(id) }
            for task in tasks { await task.value }
            self.persist()
        },capture: { ids,generation,previous in
            guard let client else { throw ResearchError("Missing cloud connection.") }
            let folder = try Workspace.directory(self.settings.workspaceRoot,"shared-context/\(group.uuidString)/round-\(generation)")
            var agents: [SharedContextManifest.Agent] = []
            for id in ids {
                guard let s = self.sessions.first(where:{$0.id == id}),let a = s.sharedAgent,let cloud = s.cloud,s.state != "Needs attention",self.cloudDrains[id] == true else { throw ResearchError("A cloud worker did not drain cleanly. No replacement was started.") }
                let snapshot = try await client.call(cloud.sandboxID,action:"checkpoint")
                func save(_ encoded: String?,name: String) throws -> String? {
                    guard let encoded else { return nil };guard let data = Data(base64Encoded:encoded) else { throw ResearchError("Invalid cloud checkpoint bytes.") }
                    let file = folder.appendingPathComponent(name);try FileManager.default.createDirectory(at:file.deletingLastPathComponent(),withIntermediateDirectories:true);try data.write(to:file,options:.atomic);try FileManager.default.setAttributes([.posixPermissions:0o600],ofItemAtPath:file.path);return file.path
                }
                let checkpoint = try save(snapshot["checkpoint"] as? String,name:"agent-\(a.index)-checkpoint.md")
                let notes = try save(snapshot["notes"] as? String,name:"agent-\(a.index)-notes.md")
                var topics: [String] = []
                for (name,encoded) in snapshot["topics"] as? [String:String] ?? [:] {
                    guard !name.hasPrefix("/"),!name.split(separator:"/").contains("..") else { throw ResearchError("Cloud topic path escaped the snapshot.") }
                    if let path = try save(encoded,name:"agent-\(a.index)-topics/" + name) { topics.append(path) }
                }
                let transcript = folder.appendingPathComponent("agent-\(a.index).json");try Workspace.write(s,to:transcript)
                agents.append(.init(number:a.index,sessionID:id,transcript:transcript.path,checkout:"Daytona \(cloud.sandboxID):/workspace/checkout",notes:notes,contextFiles:topics.sorted(),checkpoint:checkpoint))
            }
            let manifest = folder.appendingPathComponent("manifest.json");try Workspace.write(SharedContextManifest(generation:generation,project:project.repository,previousContext:previous?.path,agents:agents),to:manifest);return manifest
        },merge: { manifest in
            guard let client else { throw ResearchError("Missing cloud connection.") }
            let context = try SharedContextFiles.publish(manifest:manifest,policy:String(contentsOf:self.resourceDirectory.appendingPathComponent("SharedContextPolicy.md"),encoding:.utf8))
            let folder = manifest.deletingLastPathComponent()
            let files = FileManager.default.enumerator(at:folder,includingPropertiesForKeys:[.isRegularFileKey])!.allObjects as! [URL]
            var bundle: [String:String] = [:]
            for file in files where (try file.resourceValues(forKeys:[.isRegularFileKey])).isRegularFile == true { bundle[String(file.path.dropFirst(folder.path.count + 1))] = try Data(contentsOf:file).base64EncodedString() }
            for box in boxes { _ = try await client.call(box,action:"put-context",payload:["round":folder.lastPathComponent,"files":bundle]) }
            return context
        },retire: { ids in for id in ids { self.state(id,"Replaced") };self.persist() },cancelOperation: {},update: { phase,generation,ids,message in
            self.sharedRuns[project.id] = SharedRunStatus(groupID:group,sharesContext:options.enabled,generation:generation,phase:phase,sessionIDs:ids,message:message)
            if [.failed,.stopped,.completed].contains(phase) {
                if let message { self.error = message }
                if let client { self.cloudCleanup.formUnion(boxes);Task { await self.cleanupCloudWorkers(boxes,client:client) } };self.persist()
            }
        },waitForCompletion: { ids in
            for id in ids { await self.workerTasks[id]?.value }
            if self.sessions.contains(where:{ids.contains($0.id) && $0.state == "Needs attention"}) { throw ResearchError("A cloud agent needs attention. All completed research is preserved.") }
        })
        select(project);selectSharedGroup(group,projectID:project.id)
        let cycle = SharedContextCycle(options:options,hooks:hooks);sharedCycles[project.id] = cycle;cycle.start()
    }
}
