import XCTest
import Combine
import ResearchCore
@testable import AutoResearch

/// Live native adapter for one controlled pair. Explicit opt-in only. The
/// coordinator owns the common release, external oracle and terminal decision.
/// This driver never accepts a model's success assertion as a benchmark result.
final class MatchedNativeDriverTests: XCTestCase {
    func testProductionLiteralPublisherFixture() throws {
        guard let folder = ProcessInfo.processInfo.environment["RESEARCH_LITERAL_PARITY_FIXTURE"] else {
            throw XCTSkip("Optional local cross-language publisher fixture; no provider calls")
        }
        let root = try Workspace.validate(folder)
        let manifest = root.appendingPathComponent("manifest.json")
        let policy = try String(contentsOf:root.appendingPathComponent("policy.txt"),encoding:.utf8)
        let published = try SharedContextFiles.publish(manifest:manifest,policy:policy)
        let expected = try Data(contentsOf:root.appendingPathComponent("python-expected.md"))
        XCTAssertEqual(try Data(contentsOf:published),expected)
        let injection = try SharedContextFiles.cloudInjection(file:published,remoteRoot:"/workspace/checkout/SHARED_CONTEXT")
        XCTAssertEqual(Data(injection.utf8),try Data(contentsOf:root.appendingPathComponent("python-injection.txt")))
    }

    @MainActor func testControlledNativeArm() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["RESEARCH_MATCHED_NATIVE"] == "1",
              let rawRoot = environment["RESEARCH_MATCHED_ARM_ROOT"],
              let keyFile = environment["RESEARCH_DAYTONA_KEY_FILE"] else {
            throw XCTSkip("Explicit controlled live native adapter; no default model calls")
        }
        let root = try Workspace.validate(rawRoot)
        func read(_ name: String) throws -> [String:Any] {
            try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent(name))) as! [String:Any]
        }
        func write(_ name: String,_ value: [String:Any]) throws {
            try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent(name),options:.atomic)
        }
        let execution = try read("execution.json")
        let pair = try read("pair.json")
        let pairID = try XCTUnwrap(pair["pair_id"] as? String)
        let count = try XCTUnwrap(pair["agents_per_arm"] as? Int)
        let configuration = try read("control-settings.json")
        let review = try read("launch-review.json")
        guard review["fresh_permission_preflight_passed"] as? Bool == true,
              review["capacity_available"] as? Bool == true,
              review["source_freeze_passed"] as? Bool == true,
              review["baseline_is_discovery_failure"] as? Bool == true else {
            throw ResearchError("Controlled live launch prerequisites are not verified")
        }
        guard !FileManager.default.fileExists(atPath:root.appendingPathComponent("driver-start.json").path) else {
            throw ResearchError("Native adapter already started; reconcile its owned work instead of relaunching")
        }
        try write("driver-start.json",["pid":ProcessInfo.processInfo.processIdentifier,"unix":Date().timeIntervalSince1970,"pair_id":pairID])
        let app = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let resources = app.appendingPathComponent("Resources")
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:resources.appendingPathComponent("Defaults.json")))
        settings.workspaceRoot = try Workspace.directory(root.path,"runtime").path
        settings.codexModel = try XCTUnwrap(configuration["model"] as? String)
        settings.codexSandbox = try XCTUnwrap(configuration["sandbox"] as? String)
        let store = AppStore(resourceDirectory:resources,defaults:settings)
        // Runtime is fresh and local to this arm; no saved GUI state is loaded.
        guard store.sessions.isEmpty,store.cloudOwnedIDs.isEmpty else { throw ResearchError("Native arm runtime is not fresh") }
        store.daytona.enabled = true
        store.daytona.installID = try XCTUnwrap(execution["install_id"] as? String)
        store.daytona.connectionSandboxID = try XCTUnwrap(execution["connection_sandbox_id"] as? String)
        store.daytona.snapshot = try XCTUnwrap(execution["snapshot"] as? String)
        store.daytona.codexPackage = try XCTUnwrap(configuration["codex_package"] as? String)
        store.daytona.reasoningEffort = try XCTUnwrap(configuration["reasoning_effort"] as? String)
        store.daytona.pollSeconds = try XCTUnwrap(execution["poll_seconds"] as? Double)
        let connection = store.daytona.connectionSandboxID!
        store.cloudOwnedIDs.insert(connection)
        let client = try DaytonaClient(apiKey:String(contentsOfFile:keyFile,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines),installID:store.daytona.installID,knownOwnedIDs:[connection],apiURL:URL(string:"https://app.daytona.io/api")!)
        store.daytonaClient = client
        let source = try Workspace.validate(try XCTUnwrap(execution["source_path"] as? String))
        let request = try String(contentsOf:root.appendingPathComponent("request.txt"),encoding:.utf8)
        var project = ResearchProject(repository:"",name:pairID,localPath:source.path,isCustom:true)
        project.id = pairID
        store.projects = [project]
        let poll = store.daytona.pollSeconds
        func stopped() -> Bool { FileManager.default.fileExists(atPath:root.appendingPathComponent("stop.json").path) }
        var released = false
        var workerIDs: [String] = []
        var phases: [[String:Any]] = []
        var subscriptions = Set<AnyCancellable>()
        store.$sharedRuns.sink { values in
            guard let value = values[pairID] else { return }
            let previous = phases.last
            if previous?["phase"] as? String != value.phase.rawValue || previous?["generation"] as? Int != value.generation {
                phases.append(["phase":value.phase.rawValue,"generation":value.generation,"unix":Date().timeIntervalSince1970,"uptime":ProcessInfo.processInfo.systemUptime])
                try? write("native-phases.json",["events":phases])
            }
        }.store(in:&subscriptions)
        store.cloudPrepared = { projectID,group,boxes in
            guard projectID == pairID,boxes.count == count,Set(boxes).count == count else { throw ResearchError("Prepared native worker set does not match the pair") }
            workerIDs = boxes
            // The coordinator runs the immutable lineage/runtime/state probe on
            // each returned box, and arms the external observer before release.
            try write("prepared-workers.json",["pair_id":pairID,"group_id":group.uuidString,"worker_ids":boxes,"model_invocations":0,"unix":Date().timeIntervalSince1970])
            while !FileManager.default.fileExists(atPath:root.appendingPathComponent("release.json").path) {
                try Task.checkCancellation()
                if stopped() { throw CancellationError() }
                try await Task.sleep(for:.seconds(poll))
            }
            let release = try read("release.json")
            guard release["pair_id"] as? String == pairID,
                  let deadline = release["deadline_unix"] as? Double,
                  deadline > Date().timeIntervalSince1970,
                  release["both_observers_armed"] as? Bool == true,
                  release["all_workers_verified_fresh"] as? Bool == true,
                  release["launch_controls_passed"] as? Bool == true,
                  !stopped() else { throw ResearchError("Invalid or cancelled common release") }
            released = true
            try write("release-received.json",["pair_id":pairID,"unix":Date().timeIntervalSince1970,"release":release])
        }
        store.cloudRecoveryReady = { id in
            guard workerIDs.contains(id) else { throw ResearchError("Recovery worker is outside this arm") }
            while !FileManager.default.fileExists(atPath:root.appendingPathComponent("allow-recovery.json").path) {
                try Task.checkCancellation();try await Task.sleep(for:.seconds(poll))
            }
        }
        func stopNative() {
            store.stopSharedResearch(pairID)
            for session in store.sessions where session.projectID == pairID { store.stopWorker(session.id) }
        }
        do {
            if count == 1 && pair["mode"] as? String != "fast" {
                let session = ResearchSession(projectID:pairID,title:pairID,provider:.codex)
                store.sessions.append(session);store.launchCloud(session.id,project:project,request:request)
            } else {
                store.setSharedConfiguration(.init(enabled:pair["mode"] as? String == "fast",agentCount:count,intervalMinutes:Double(try XCTUnwrap(pair["interval_seconds"] as? Int))/60,parallelEnabled:true),for:pairID)
                store.startCloudResearch(project,request:request)
            }
            var requestedStop = false
            while true {
                if stopped() { requestedStop = true;stopNative() }
                try Workspace.write(store.sessions,to:root.appendingPathComponent("native-sessions.json"))
                if let error = store.error { throw ResearchError(error) }
                if store.sessions.contains(where:{$0.state == "Needs attention"}) { throw ResearchError("Actual native agent failed; preserve exact session and do not retry") }
                let active = !store.runners.isEmpty || store.sharedCycles[pairID]?.active == true
                if released && !active {
                    guard store.sessions.allSatisfy({ store.cloudDrains[$0.id] == true }) else {
                        throw ResearchError("Native completion lacks confirmed model process drain")
                    }
                    try write("model-drained.json",["pair_id":pairID,"worker_ids":workerIDs,"unix":Date().timeIntervalSince1970,"stop_requested":requestedStop,"all_owned_model_processes_drained":true])
                    if requestedStop { break }
                    // Normal model completion is not an external discovery
                    // result. Preserve files while the observer drains checks.
                }
                if requestedStop && !active { break }
                try await Task.sleep(for:.seconds(poll))
            }
            guard store.sessions.allSatisfy({ store.cloudDrains[$0.id] == true }) else {
                throw ResearchError("A native model process has not confirmed drain")
            }
            try write("driver-drained.json",["pair_id":pairID,"worker_ids":workerIDs,"unix":Date().timeIntervalSince1970,"discovery_outcome_owned_by_external_coordinator":true])
            // Archive admission is a separate coordinator decision; a caller
            // that has not preserved candidates cannot trigger it via stop.
            while !store.cloudCleanup.isEmpty { try await Task.sleep(for:.seconds(poll)) }
            for task in store.workerTasks.values { await task.value }
            try write("driver-complete.json",["pair_id":pairID,"worker_ids":workerIDs,"unix":Date().timeIntervalSince1970])
        } catch {
            stopNative()
            for cycle in store.sharedCycles.values { await cycle.waitUntilStopped() }
            // Do not wait forever for archive admission after a failed pair.
            // Preserve originals by stopping only this arm's owned workers.
            // A preparation failure can precede cloudPrepared. Reconcile only
            // this project's actual group UUIDs, never all installation boxes.
            var groups = Set(store.sessions.filter {$0.projectID == pairID}.map {$0.id.uuidString})
            if let group = store.sharedRuns[pairID]?.groupID { groups.insert(group.uuidString) }
            let owned = (try? await client.discoverOwned()) ?? []
            workerIDs = Array(Set(workerIDs + owned.filter {
                $0.labels?["autoresearch-role"] == "worker" && groups.contains($0.labels?["autoresearch-group"] ?? "")
            }.map(\.id))).sorted()
            var states: [String:String] = [:]
            for id in workerIDs {
                do { try await client.stop(id,archive:false);states[id] = try await client.inspect(id).state }
                catch { states[id] = "unconfirmed" }
            }
            try? Workspace.write(store.sessions,to:root.appendingPathComponent("native-sessions.json"))
            try? write("driver-failure.json",["pair_id":pairID,"message":error.localizedDescription,"worker_ids":workerIDs,"worker_states":states,"unix":Date().timeIntervalSince1970])
            throw error
        }
        _ = subscriptions
    }
}
