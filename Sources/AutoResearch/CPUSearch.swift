import AppKit
import ResearchCore

extension AppStore {
    func cpuSearchConfiguration(_ project: ResearchProject) -> CPUSearchConfiguration? {
        LeanSearch.cpuConfiguration(directory: cpuSearchDirectory(project))
    }

    func availableSubtasks(_ project: ResearchProject) -> [ResearchSubtask] {
        researchSubtasks.filter { subtaskAvailable($0, project: project) }
    }

    func subtaskAvailable(_ task: ResearchSubtask, project: ResearchProject) -> Bool {
        !(task.execution == "script" || task.requiresCPUSearch == true) || cpuSearchConfiguration(project) != nil
    }

    var isCPUSubtask: Bool {
        session?.scripted == true || researchSubtasks.contains { $0.id == session?.researchSubtaskID && $0.execution == "script" }
    }

    func cpuSearchDirectory(_ project: ResearchProject, session: ResearchSession? = nil) -> URL {
        URL(fileURLWithPath: session?.workingDirectory ?? project.localPath ?? URL(fileURLWithPath: settings.workspaceRoot).appendingPathComponent("projects/\(project.repository)").path)
    }

    func revealCPUSearchPlan() {
        guard let project else { return }
        let directory = cpuSearchDirectory(project, session: session)
        let configURL = directory.appendingPathComponent(".auto-research/cpu-search.json")
        if let data = try? Data(contentsOf: configURL), let config = try? JSONDecoder().decode(CPUSearchConfiguration.self, from: data) {
            NSWorkspace.shared.open(directory.appendingPathComponent(config.plan))
        } else {
            error = "Configure .auto-research/cpu-search.json in this repository with python, script and plan paths. No model fallback is used. See docs/lean-search.md for the script protocol."
        }
    }

    func launchCPUSearch(_ id: UUID, project: ResearchProject) {
        guard runners[id] == nil, let current = sessions.first(where: { $0.id == id }) else { return }
        let target = cpuSearchDirectory(project, session: current).resolvingSymlinksInPath()
        guard !activePaths.values.contains(target.path), !sharedActive(project.id) else {
            append(id, role: "error", text: "This checkout has active research. Stop it before running the CPU script.")
            state(id, "Needs attention"); return
        }
        activePaths[id] = target.path
        let runner = ProcessRunner(); runners[id] = runner; stopped.remove(id)
        state(id, "Preparing")
        workerTasks[id] = Task {
            defer { runners.removeValue(forKey: id); activePaths.removeValue(forKey: id); persist() }
            do {
                let directory = try Workspace.validate(target.path)
                let configURL = directory.appendingPathComponent(".auto-research/cpu-search.json")
                guard let config = LeanSearch.cpuConfiguration(directory: directory) else {
                    throw ResearchError("CPU search needs an installed script, plan and search summary in \(configURL.path). No model was launched. Configure this repository before running.")
                }
                if let index = sessions.firstIndex(where: { $0.id == id }) {
                    sessions[index].workingDirectory = directory.path; sessions[index].scripted = true
                }
                try LeanSearch.install(resources: resourceDirectory, directory: directory.appendingPathComponent(".auto-research-tools/lean-search"))
                let script = directory.appendingPathComponent(config.script).standardizedFileURL
                let plan = directory.appendingPathComponent(config.plan).standardizedFileURL
                guard FileManager.default.fileExists(atPath: script.path), FileManager.default.fileExists(atPath: plan.path), !config.python.isEmpty else {
                    throw ResearchError("Configured Python script or plan is missing. No model was launched.")
                }
                let overview = LeanSearch.overviewURL(directory: directory, sessionID: id)
                try FileManager.default.createDirectory(at: overview.deletingLastPathComponent(), withIntermediateDirectories: true)
                let output = overview.deletingLastPathComponent().appendingPathComponent("result-" + UUID().uuidString)
                let logs = try Workspace.directory(settings.workspaceRoot, "sessions/\(id.uuidString)")
                let logURL = logs.appendingPathComponent(UUID().uuidString + ".log")
                FileManager.default.createFile(atPath: logURL.path, contents: nil)
                let log = try FileHandle(forWritingTo: logURL); let lock = NSLock()
                defer { try? log.close() }
                let command = Command("/usr/bin/env", [config.python, "-u", "-B", script.path, "--plan", plan.path, "--output", output.path, "--overview", overview.path])
                try Workspace.write(["executable": command.executable, "arguments": command.arguments.joined(separator: "\n"), "config": configURL.path], to: logs.appendingPathComponent("command.json"))
                append(id, role: "status", text: "Running generated CPU script. No model is used.\nScript: \(script.path)\nPlan: \(plan.path)\nResults: \(output.path)")
                state(id, "Running")
                var scriptEnvironment = try environment()
                if let connection = leanServer.connectionPath { scriptEnvironment["LEAN_VERIFIER_CONNECTION"] = connection }
                let result = try await runner.run(command, directory: directory, environment: scriptEnvironment, onLine: { line, stderr in
                    lock.lock(); try? log.write(contentsOf: Data((line + "\n").utf8)); lock.unlock()
                    Task { @MainActor in self.append(id, role: stderr ? "error" : "tool", text: line) }
                })
                try Workspace.write(["exit_code": String(result.code), "output": output.path, "model_calls": "0"], to: logs.appendingPathComponent("process.json"))
                if stopped.contains(id) { state(id, "Stopped"); return }
                guard result.code == 0 else { throw ResearchError("CPU script exited with status \(result.code). \(result.stderr) Results and logs are preserved.") }
                let receipt = try JSONDecoder().decode(CPUSearchResult.self, from: Data(contentsOf: output.appendingPathComponent("cpu-result.json")))
                append(id, role: "assistant", text: receipt.summary + "\n\nResults: \(output.path)")
                state(id, receipt.status == "COMPLETED" ? "Completed" : "Needs attention")
            } catch {
                if stopped.contains(id) || error is CancellationError { state(id, "Stopped") }
                else { append(id, role: "error", text: error.localizedDescription); state(id, "Needs attention") }
            }
        }
    }
}
