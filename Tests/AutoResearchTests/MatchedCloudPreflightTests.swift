import XCTest
import ResearchCore
@testable import AutoResearch

/// One actual native invocation, explicitly enabled only for a fresh paired
/// transport/policy preflight. No research result is counted here.
final class MatchedCloudPreflightTests: XCTestCase {
    @MainActor func testFreshNativePreflight() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["RESEARCH_MATCHED_PREFLIGHT"] == "1",let path = env["RESEARCH_MATCHED_PREFLIGHT_ROOT"],let keyPath = env["RESEARCH_DAYTONA_KEY_FILE"] else { throw XCTSkip("Actual cloud preflight requires explicit opt-in") }
        let root = try Workspace.validate(path)
        func object(_ name: String) throws -> [String:Any] { try JSONSerialization.jsonObject(with:Data(contentsOf:root.appendingPathComponent(name))) as! [String:Any] }
        func write(_ name: String,_ value: [String:Any]) throws { try JSONSerialization.data(withJSONObject:value,options:[.prettyPrinted,.sortedKeys]).write(to:root.appendingPathComponent(name),options:.atomic) }
        guard !FileManager.default.fileExists(atPath:root.appendingPathComponent("native-launch.json").path) else { throw ResearchError("Preflight already launched; reconcile existing operation") }
        let prepared = try object("prepared.json"),configuration = try object("control-settings.json")
        guard prepared["fresh_workers_verified"] as? Bool == true,prepared["new_cloud_login_authenticated"] as? Bool == true else { throw ResearchError("Fresh authenticated preflight workers are required") }
        let native = (prepared["workers"] as! [String:[String:Any]])["native"]!
        let worker = native["id"] as! String
        let app = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let resources = app.appendingPathComponent("Resources")
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:resources.appendingPathComponent("Defaults.json")))
        settings.workspaceRoot = try Workspace.directory(root.path,"native-runtime").path
        settings.codexModel = configuration["model"] as! String;settings.codexSandbox = configuration["sandbox"] as! String
        let store = AppStore(resourceDirectory:resources,defaults:settings)
        guard store.sessions.isEmpty else { throw ResearchError("Preflight runtime has prior sessions") }
        store.daytona.enabled = true;store.daytona.installID = prepared["install_id"] as! String
        store.daytona.reasoningEffort = configuration["reasoning_effort"] as! String
        store.daytona.codexPackage = configuration["codex_package"] as! String
        store.cloudOwnedIDs.insert(worker)
        store.daytonaClient = try DaytonaClient(apiKey:String(contentsOfFile:keyPath,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines),installID:store.daytona.installID,knownOwnedIDs:[worker],apiURL:URL(string:"https://app.daytona.io/api")!)
        store.cloudRecoveryReady = { _ in
            while !FileManager.default.fileExists(atPath:root.appendingPathComponent("allow-native-recovery.json").path) { try await Task.sleep(for:.seconds(1)) }
        }
        let project = ResearchProject(repository:"",name:"Matched runtime preflight",localPath:root.path,isCustom:true)
        var session = ResearchSession(projectID:project.id,title:project.name,provider:.codex)
        session.cloud = CloudSession(sandboxID:worker,stateID:session.id.uuidString)
        store.sessions.append(session)
        try write("native-launch.json",["sandbox_id":worker,"state_id":session.id.uuidString,"pid":ProcessInfo.processInfo.processIdentifier,"unix":Date().timeIntervalSince1970,"research_groups_started":0])
        store.launchCloud(session.id,project:project,request:try String(contentsOf:root.appendingPathComponent("request.txt"),encoding:.utf8))
        while store.runners[session.id] != nil { try await Task.sleep(for:.seconds(1)) }
        let finished = try XCTUnwrap(store.sessions.first)
        try write("native-model-ended.json",["state":finished.state,"actual_provider_id":finished.providerSessionID as Any? ?? NSNull(),"drained":store.cloudDrains[session.id] == true,"unix":Date().timeIntervalSince1970])
        try Workspace.write(store.sessions,to:root.appendingPathComponent("native-sessions.json"))
        await store.workerTasks[session.id]?.value
        guard finished.state == "Completed",store.cloudDrains[session.id] == true else { throw ResearchError("Actual native preflight did not complete; original evidence retained") }
        try write("native-complete.json",["unix":Date().timeIntervalSince1970,"research_groups_started":0])
    }
}
