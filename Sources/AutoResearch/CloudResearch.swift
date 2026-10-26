import Foundation
import ResearchCore

extension AppStore {
    func launchCloud(_ id: UUID, project: ResearchProject, request: String?) {
        guard project.builtinID == nil else { error = "Built-in challenge starters run on this Mac and use the official remote judge. Start them from the project page."; return }
        guard runners[id] == nil,let initial = sessions.first(where:{$0.id == id}) else { return }
        do { try gatewayPreflight(initial, cloud: true) }
        catch { append(id, role: "error", text: error.localizedDescription); state(id, "Needs attention"); return }
        let savedMonitor = URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("cloud-method-attachments/" + id.uuidString + ".json")
        if FileManager.default.fileExists(atPath: savedMonitor.path) {
            restoreMethodMonitoring()
            if runners[id] == nil { error = "The saved cloud operation needs inspection before a replacement can be launched." }
            return
        }
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
                    try await cloudPrepared?(project.id,id,[box])
                }
                guard let cloud else { throw ResearchError("Missing cloud worker.") };boxID = cloud.sandboxID
                try await c.start(cloud.sandboxID)
                guard !stopped.contains(id) else { throw CancellationError() }
                let runner = ProcessRunner(backend:DaytonaRunner(client:c,cloud:cloud,configuration:daytona));runners[id] = runner
                var prompt = request ?? "Start an autonomous research run in your assigned /workspace/checkout. Read applicable AGENTS.md and auto-research.md. If that entry point is absent, locate the repository's actual research guide, explain the source you choose, and follow it. Ask for input if no actionable objective exists."
                let methodOptions = sharedConfiguration(project.id).methods
                if let methods = methodOptions, methods.enabled {
                    prompt += "\nYou are the coordinator for \(methods.branches) method branches collaborating on this same objective. Previous group indices are historical evidence, not the current team assignment. Write the common plan; the owned method runner launches the branches. Do not start another group or nested fan-out."
                }
                if let shared = initial.sharedAgent {
                    if methodOptions?.enabled != true { prompt += "\nAgent \(shared.index) of \(shared.agentCount ?? 1), generation \(shared.generation), in your own private checkout." }
                    if !shared.isIndependent,let path = shared.contextFile {
                        prompt += "\n" + (try SharedContextFiles.cloudInjection(file:URL(fileURLWithPath:path),remoteRoot:"/workspace/checkout/SHARED_CONTEXT"))
                    }
                }
                let extra = initial.sharedAgent == nil || methodOptions?.enabled == true ? "" : try String(contentsOf:resourceDirectory.appendingPathComponent(initial.sharedAgent?.isIndependent == true ? "IndependentAgentPolicy.md" : "SharedAgentPolicy.md"),encoding:.utf8)
                var cloudPolicy = policy + "\n\n" + extra + "\n\nExecution environment: your private Linux Daytona sandbox, assigned checkout /workspace/checkout. All commands and generated files stay in this checkout. The external-SSD rule applies to the native Mac controller, whose mount was validated there; do not look for /Volumes inside Linux. No host account files, prior conversations, peer checkouts or host MCP configuration are supplied."
                var cloudSettings = settings;cloudSettings.codexPath = "/opt/autoresearch/bin/codex"
                var methodDirectory: String?
                var methodFiles: [String: Data] = [:]
                if let methods = sharedConfiguration(project.id).methods, methods.enabled {
                    var files = try ResearchMethods.files(options: methods, provider: .codex, settings: cloudSettings,
                                                         policy: cloudPolicy, resources: resourceDirectory, sharedContext: sharedConfiguration(project.id))
                    var config = try JSONSerialization.jsonObject(with: files["config.json"]!) as! [String: Any]
                    config["arguments"] = ["--no-daemon", "-c", "model_reasoning_effort=" + DaytonaClient.quote(daytona.reasoningEffort), "-c", "features.memories=false", "-c", "features.multi_agent=false"] + (config["arguments"] as! [String]).filter { $0 != "--no-daemon" }
                    files["config.json"] = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
                    let name = "methods-" + UUID().uuidString
                    // Existing sandboxes may still have the previous bridge version.
                    try await c.upload(cloud.sandboxID, files: ["/opt/autoresearch/bridge.py": try Data(contentsOf: resourceDirectory.appendingPathComponent("DaytonaWorker.py")), "/opt/autoresearch/MethodTelemetry.py": try Data(contentsOf: resourceDirectory.appendingPathComponent("MethodTelemetry.py"))])
                    _ = try await c.call(cloud.sandboxID, action: "put-context", payload: ["round": name, "files": files.mapValues { $0.base64EncodedString() }])
                    methodDirectory = "/workspace/checkout/SHARED_CONTEXT/" + name; methodFiles = files
                    cloudPolicy += "\n\n" + ResearchMethods.instruction(directory: "/workspace/checkout/SHARED_CONTEXT/" + name)
                    append(id, role: "status", text: "Optional research methods installed in this Daytona sandbox. Branch workspaces and logs remain inside /workspace/checkout.")
                }
                prompt = try claimFleetFollowups(id, prompt: prompt)
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
                if let folder = methodDirectory, !stopped.contains(id), result.code == 0, !failed.contains(id), completed.contains(id),
                   (ResearchMethods.ready(result.stdout, provider: .codex) || !["finished", "needs_input"].contains(nextDecision ?? "")) {
                    guard ResearchMethods.ready(result.stdout, provider: .codex) else { throw ResearchError("The model has not prepared a research-method plan. No branches were launched; review its response.") }
                    _ = try await c.call(cloud.sandboxID, action: "put-context", payload: ["round": URL(fileURLWithPath: folder).lastPathComponent, "files": methodFiles.mapValues { $0.base64EncodedString() }])
                    cloudDrains[id] = false
                    append(id, role: "status", text: "Plan ready. Running isolated method branches in this Daytona sandbox.")
                    let operation = UUID().uuidString
                    let attachment = CloudMethodAttachment(sessionID: id, sandboxID: cloud.sandboxID, operationID: operation, controlDirectory: folder)
                    try JSONEncoder().encode(attachment).write(to: methodAttachmentURL(id), options: .atomic)
                    let methodRunner = ProcessRunner(backend: DaytonaRunner(client: c, cloud: cloud, configuration: daytona, operationID: operation))
                    runners[id] = methodRunner
                    let execution = try await methodRunner.run(Command("autoresearch-methods", [folder]), directory: root, onLine: { line, _ in
                        Task { @MainActor in self.consumeMethodLine(line, sessionID: id, operation: operation) }
                    })
                    cloudDrains[id] = true
                    try FileManager.default.removeItem(at: methodAttachmentURL(id))
                    guard execution.code == 0 else { throw ResearchError("Research-method execution stopped or failed. All cloud branches and logs are preserved. \(execution.stderr)") }
                    append(id, role: "assistant", text: try ResearchMethods.summary(execution.stdout))
                    nextDecision = keepResearching.contains(id) ? "continue" : nil
                }
                if stopped.contains(id) { state(id,"Stopped") }
                else if result.code != 0 || failed.contains(id) || !completed.contains(id) { state(id,"Needs attention");append(id,role:"error",text:"Cloud CLI did not complete successfully (\(result.code)). Its files and logs are preserved.") }
                else if nextDecision == "needs_input" { state(id,"Needs attention") }
                else { state(id,"Completed") }
            } catch {
                state(id,error is CancellationError ? "Stopped" : "Needs attention")
                if !(error is CancellationError) { append(id,role:"error",text:error.localizedDescription + (stopped.contains(id) ? " Cloud process drain was not confirmed; no successor may be started." : "")) }
            }
            runners.removeValue(forKey:id);persist()
            if finishFleetTurn(id) { return }
            let finished = sessions.first(where:{$0.id == id})
            if finished?.sharedAgent != nil && !sharedConfiguration(project.id).usesMethods {
                if finished?.state == "Needs attention",finished?.sharedAgent?.isIndependent != true { sharedCycles[project.id]?.stop(failure:"A cloud agent needs attention. Its sandbox was preserved.") }
            } else if keepResearching.contains(id),finished?.state == "Completed",nextDecision == "continue" {
                let next = (try? String(contentsOf:resourceDirectory.appendingPathComponent("ContinuePolicy.md"),encoding:.utf8)) ?? ""
                if !next.isEmpty { launchCloud(id,project:project,request:next);return }
            } else if let client,let boxID {
                keepResearching.remove(id)
                if cloudDrains[id] == true,finished?.state != "Needs attention" { await cleanupCloudWorkers([boxID],client:client) }
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

    func resumeCloudResearch(_ project: ResearchProject, groupID: UUID, request: String? = nil) {
        guard !running(project) else { error = "Stop the current research before resuming the saved group."; return }
        do {
            let plan = try CloudGroupResume(sessions: sessions, projectID: project.id, groupID: groupID,
                                            options: sharedConfiguration(project.id))
            guard plan.sessions.allSatisfy({ !cloudCleanup.contains($0.cloud!.sandboxID) }) else {
                throw ResearchError("Wait for cloud recovery export before resuming this group.")
            }
            _ = try Workspace.validate(settings.workspaceRoot)
            let prompt = try request ?? String(contentsOf: resourceDirectory.appendingPathComponent("ContinuePolicy.md"), encoding: .utf8)
            runCloudResearch(project, request: prompt, resume: plan)
        } catch { self.error = error.localizedDescription }
    }

    func startCloudResearch(_ project: ResearchProject,request: String?) {
        guard !running(project),provider == .codex else { error = "Stop existing research and select Codex before starting Daytona agents.";return }
        runCloudResearch(project, request: request, resume: nil)
    }

    private func runCloudResearch(_ project: ResearchProject, request: String?, resume: CloudGroupResume?) {
        guard gatewayConfigurationReadable, !gateway.applies(to: .codex) else { error = "OpenRouter runs on This Mac only. Repair gateway settings, switch execution to This Mac, or turn Gateway off before starting Daytona research."; return }
        guard project.builtinID == nil else { error = "Built-in challenge starters use a local coordinator. Start research from the project page."; return }
        let options = resume?.options ?? sharedConfiguration(project.id)
        do { try options.validate() } catch { self.error = error.localizedDescription;return }
        let group = resume?.groupID ?? UUID()
        var boxes: [String] = resume?.sessions.compactMap { $0.cloud?.sandboxID } ?? []
        var client: DaytonaClient?
        var resumedInitial = false
        var preparationSucceeded = false
        sharedRuns[project.id] = SharedRunStatus(groupID:group,sharesContext:options.enabled)
        let hooks = SharedContextCycle.Hooks(prepare: {
            if let resume {
                let c = try self.cloudClient(); client = c
                let probe = try String(contentsOf: self.resourceDirectory.appendingPathComponent("CloudResumeProbe.py"), encoding: .utf8)
                // A start preserves an existing stopped sandbox; no setup, clone, auth fork or creation.
                // Validate every worker before any resumed model invocation is submitted.
                for session in resume.sessions {
                    try Task.checkCancellation()
                    let cloud = session.cloud!
                    try await c.start(cloud.sandboxID)
                    let command = "sudo -n python3 -B -c " + DaytonaClient.quote(probe) + " " + DaytonaClient.quote(cloud.stateID) + " " + DaytonaClient.quote(session.providerSessionID!)
                    let output = try await c.execute(cloud.sandboxID, command: command)
                    let receipt = try JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any]
                    guard receipt?["status"] as? String == "READY_TO_RESUME_EXISTING_STATE" else {
                        throw ResearchError("Agent \(session.sharedAgent!.index) is active, has uncertain cleanup, or lacks its saved provider state. No group worker was launched. Inspect the preserved sandbox.")
                    }
                }
                preparationSucceeded = true
                return
            }
            let (c,connection) = try await self.ensureCloudConnection();client = c
            let auth = try await c.call(connection,action:"auth-status")
            guard auth["authenticated"] as? Bool == true else { throw ResearchError("Sign in to cloud Codex in Connections & settings → Daytona first.") }
            let source = try await self.cloudSource(project,group:group)
            for _ in 0..<options.agentCount { try Task.checkCancellation();boxes.append(try await self.createCloudBox(c,role:"worker",group:group)) }
            try await withThrowingTaskGroup(of:Void.self) { tasks in
                for box in boxes { tasks.addTask { try await self.prepareCloudWorker(client:c,connection:connection,id:box,project:project,source:source) } }
                try await tasks.waitForAll()
            }
            try await self.cloudPrepared?(project.id,group,boxes)
            preparationSucceeded = true
        },launch: { generation,context,previous in
            if let resume, !resumedInitial {
                guard previous == resume.sessions.map(\.id), generation == resume.generation else {
                    throw ResearchError("Saved cloud group lineage changed before resume.")
                }
                for saved in resume.sessions {
                    guard let i = self.sessions.firstIndex(where: { $0.id == saved.id }),
                          self.sessions[i].providerSessionID == saved.providerSessionID,
                          self.sessions[i].cloud?.sandboxID == saved.cloud?.sandboxID,
                          self.sessions[i].cloud?.stateID == saved.cloud?.stateID else {
                        throw ResearchError("Saved cloud worker changed before resume; no replacement was created.")
                    }
                }
                for saved in resume.sessions {
                    if let i = self.sessions.firstIndex(where: { $0.id == saved.id }) { self.sessions[i].closedAt = nil }
                }
                guard self.persist() else { throw ResearchError("Could not persist resumed lineage; no workers were launched.") }
                resumedInitial = true
                for saved in resume.sessions { self.launchCloud(saved.id, project: project, request: request) }
                return previous
            }
            var ids: [UUID] = []
            for (offset,box) in boxes.enumerated() {
                var session = ResearchSession(projectID:project.id,title:"Agent \(offset + 1) · \(options.enabled ? "round \(generation)" : "independent") · Daytona",provider:.codex)
                let old = previous.indices.contains(offset) ? previous[offset] : nil
                session.cloud = CloudSession(sandboxID:box,stateID:session.id.uuidString,previousStateID:old?.uuidString)
                session.sharedAgent = SharedAgent(groupID:group,index:offset + 1,generation:generation,previousSessionID:old,contextFile:context?.path,sharesContext:options.enabled,agentCount:options.agentCount)
                session.messages.append(ChatMessage(role:"status",text:"Private Daytona sandbox \(box). Fresh provider state; only cloud sign-in and this group's published checkpoints are supplied."))
                self.inheritFleetDraft(session)
                self.sessions.append(session);ids.append(session.id)
            }
            guard self.persist() else { throw ResearchError("Could not save cloud session lineage. No replacement agents were started.") }
            for id in ids { self.launchCloud(id,project:project,request:request) };return ids
        },drain: { ids in
            let rotating = self.sharedRuns[project.id]?.phase == .capturing && self.sharedCycles[project.id]?.active == true
            if rotating { self.fleetDrainingForRotation.formUnion(ids) }
            defer { self.fleetDrainingForRotation.subtract(ids) }
            let tasks = ids.compactMap { self.workerTasks[$0] }
            for id in ids { self.stopWorker(id) }
            for task in tasks { await task.value }
            self.persist()
        },capture: { ids,generation,previous in
            guard let client else { throw ResearchError("Missing cloud connection.") }
            let roundPath = "shared-context/\(group.uuidString)/round-\(generation)"
            let existingRound = URL(fileURLWithPath: self.settings.workspaceRoot).appendingPathComponent(roundPath)
            let snapshotPath = FileManager.default.fileExists(atPath: existingRound.path)
                ? roundPath + "-" + UUID().uuidString : roundPath
            let folder = try Workspace.directory(self.settings.workspaceRoot, snapshotPath)
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
                if phase != .completed { self.suspendFleetGroup(group) }
                if let message { self.error = message }
                let drained = boxes.filter { box in
                    let sessions = self.sessions.filter { $0.cloud?.sandboxID == box }
                    return !sessions.isEmpty && sessions.allSatisfy { self.cloudDrains[$0.id] == true && $0.state != "Needs attention" }
                }
                if let client, preparationSucceeded { self.cloudCleanup.formUnion(drained);Task { await self.cleanupCloudWorkers(drained,client:client) } };self.persist()
            }
        },waitForCompletion: { ids in
            await self.waitForFleetTurns(ids)
            if self.sessions.contains(where:{ids.contains($0.id) && $0.state == "Needs attention"}) { throw ResearchError("A cloud agent needs attention. All completed research is preserved.") }
        })
        select(project);selectSharedGroup(group,projectID:project.id)
        let cycle = SharedContextCycle(options:options,hooks:hooks);sharedCycles[project.id] = cycle
        cycle.start(resuming: resume.map { SharedContextCycle.ResumeState(generation: $0.generation, sessionIDs: $0.sessions.map(\.id), context: $0.context) })
    }
}
