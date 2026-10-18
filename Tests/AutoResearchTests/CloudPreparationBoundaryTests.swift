import XCTest
import ResearchCore
@testable import AutoResearch

/// Exercises real preparation and launch paths with every HTTP request handled
/// locally. No cloud sandbox, account or model is used by these tests.
private final class PreparationProtocol: URLProtocol, @unchecked Sendable {
    static let lock = NSLock()
    static var install = ""
    static var boxes: [String:[String:Any]] = [:]
    static var launches: [[String:Any]] = []
    static var checkpoints = 0
    static var packs = 0
    static func reset(_ id: String) {
        lock.lock();defer { lock.unlock() }
        install = id;boxes = [:];launches = [];checkpoints = 0;packs = 0
    }
    static func counts() -> (Int,Int) {
        lock.lock();defer { lock.unlock() };return (launches.count,checkpoints)
    }
    static func packCount() -> Int { lock.lock();defer { lock.unlock() };return packs }
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    private func body() throws -> [String:Any] {
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream,data.isEmpty {
            stream.open();defer { stream.close() };var buffer = [UInt8](repeating:0,count:4096)
            while stream.hasBytesAvailable { let n = stream.read(&buffer,maxLength:buffer.count);if n <= 0 { break };data.append(buffer,count:n) }
        }
        return data.isEmpty ? [:] : try JSONSerialization.jsonObject(with:data) as! [String:Any]
    }
    override func startLoading() {
        do {
            let input = try body()
            Self.lock.lock();defer { Self.lock.unlock() }
            let object: [String:Any]
            if request.url!.host == "app.daytona.io" {
                let parts = request.url!.path.split(separator:"/")
                if parts.last == "sandbox",request.httpMethod == "POST" {
                    let id = UUID().uuidString
                    let box: [String:Any] = ["id":id,"state":"started","labels":input["labels"]!,"cpu":2,"memory":4,"disk":8]
                    Self.boxes[id] = box;object = box
                } else if parts.last == "sandbox" { object = ["items":Array(Self.boxes.values)] }
                else {
                    let id = String(parts[2]);guard let box = Self.boxes[id] else { throw ResearchError("Unknown fixture sandbox") }
                    if parts.last == "preview-url" { object = ["url":"https://" + id.lowercased() + ".preparation.invalid/","token":"local-fixture"] }
                    else if parts.last == "stop" { var stopped = box;stopped["state"] = "stopped";Self.boxes[id] = stopped;object = [:] }
                    else { object = box }
                }
            } else {
                guard request.url!.host!.hasSuffix(".preparation.invalid") else { throw ResearchError("Unexpected fixture origin") }
                let command = input["command"] as! String
                var response: [String:Any] = [:]
                if !command.contains("/opt/autoresearch/bridge.py ") { object = ["exitCode":0,"result":"uploaded"] }
                else {
                    let action = command.components(separatedBy:"/opt/autoresearch/bridge.py ")[1]
                    let encoded = command.components(separatedBy:"'")[1]
                    let payload = try JSONSerialization.jsonObject(with:Data(base64Encoded:encoded)!) as! [String:Any]
                    switch action {
                    case "setup-status": response = ["phase":"ready","runtime":["release":"fixture-only"]]
                    case "auth-status": response = ["authenticated":true]
                    case "fork-auth": response = ["copied":true]
                    case "launch": Self.launches.append(payload);response = ["phase":"requested"]
                    case "checkpoint": Self.checkpoints += 1;response = [:]
                    case "pack": Self.packs += 1;throw ResearchError("Fixture export deliberately ends before archive download")
                    case "poll":
                        // Terminal failure keeps fixture workers from triggering
                        // automatic recovery. This is not a research outcome.
                        response = ["stdout":"","stderr":"","stdout_offset":0,"stderr_offset":0,"stdout_eof":true,"stderr_eof":true,"status":["phase":"ended","exit_code":1]]
                    default: throw ResearchError("Unexpected fixture action: " + action)
                    }
                    object = ["exitCode":0,"result":String(decoding:try JSONSerialization.data(withJSONObject:response),as:UTF8.self)]
                }
            }
            client?.urlProtocol(self,didReceive:HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:nil,headerFields:nil)!,cacheStoragePolicy:.notAllowed)
            client?.urlProtocol(self,didLoad:try JSONSerialization.data(withJSONObject:object));client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self,didFailWithError:error) }
    }
    override func stopLoading() {}
}

final class CloudPreparationBoundaryTests: XCTestCase {
    @MainActor private func fixture() async throws -> (AppStore,ResearchProject,URL) {
        let app = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let root = try Workspace.directory(app.path,"tmp/cloud-preparation-" + UUID().uuidString)
        let source = try Workspace.directory(root.path,"source")
        try "Fixture input".write(to:source.appendingPathComponent("input.txt"),atomically:true,encoding:.utf8)
        for args in [["init","-q"],["add","."]] {
            let result = try await ProcessRunner().run(Command("/usr/bin/git",args),directory:source)
            XCTAssertEqual(result.code,0)
        }
        let resources = app.appendingPathComponent("Resources")
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:resources.appendingPathComponent("Defaults.json")))
        settings.workspaceRoot = try Workspace.directory(root.path,"runtime").path
        let store = AppStore(resourceDirectory:resources,defaults:settings)
        store.daytona.enabled = true;store.daytona.pollSeconds = 0.01
        PreparationProtocol.reset(store.daytona.installID)
        let config = URLSessionConfiguration.ephemeral;config.protocolClasses = [PreparationProtocol.self]
        store.daytonaClient = try DaytonaClient(apiKey:"fixture",installID:store.daytona.installID,apiURL:URL(string:"https://app.daytona.io/api")!,session:URLSession(configuration:config))
        let project = ResearchProject(repository:"",name:"Preparation fixture",localPath:source.path,isCustom:true)
        store.projects = [project]
        return (store,project,root)
    }
    @MainActor private func wait(_ condition: @escaping () -> Bool) async throws {
        let deadline = Date().addingTimeInterval(10)
        while !condition(),Date() < deadline { try await Task.sleep(for:.milliseconds(10)) }
        XCTAssertTrue(condition(),"Local preparation fixture timed out")
    }
    @MainActor func testSingleAgentCannotInvokeCLIBeforePreparedGateRelease() async throws {
        let (store,project,root) = try await fixture()
        defer { store.shutdown();try? FileManager.default.removeItem(at:root) }
        var prepared = false,release = false
        store.cloudPrepared = { projectID,_,boxes in
            XCTAssertEqual(projectID,project.id);XCTAssertEqual(boxes.count,1);prepared = true
            while !release { try await Task.sleep(for:.milliseconds(10)) }
        }
        let session = ResearchSession(projectID:project.id,title:"Fixture",provider:.codex)
        store.sessions.append(session);store.launchCloud(session.id,project:project,request:"Fixture request")
        try await wait { prepared }
        XCTAssertEqual(PreparationProtocol.counts().0,0)
        XCTAssertEqual(store.sessions[0].state,"Preparing")
        release = true
        await store.workerTasks[session.id]?.value
        XCTAssertEqual(PreparationProtocol.counts().0,1)
    }
    @MainActor func testSharedTimerAndAllCLIsWaitForRelease() async throws {
        let (store,project,root) = try await fixture()
        defer { store.shutdown();try? FileManager.default.removeItem(at:root) }
        var prepared = false,release = false
        store.cloudPrepared = { _,_,boxes in
            XCTAssertEqual(boxes.count,2);prepared = true
            while !release { try await Task.sleep(for:.milliseconds(10)) }
        }
        store.setSharedConfiguration(.init(enabled:true,agentCount:2,intervalMinutes:0.01,parallelEnabled:true),for:project.id)
        store.startCloudResearch(project,request:"Fixture request")
        try await wait { prepared }
        // Hold longer than a full sharing interval. No model or checkpoint
        // operation may occur while the other arm is still preparing.
        try await Task.sleep(for:.milliseconds(750))
        XCTAssertEqual(PreparationProtocol.counts().0,0);XCTAssertEqual(PreparationProtocol.counts().1,0)
        XCTAssertEqual(store.sharedRuns[project.id]?.phase,.preparing)
        XCTAssertTrue(store.sessions.isEmpty)
        release = true
        try await wait { PreparationProtocol.counts().0 == 2 }
        await store.sharedCycles[project.id]?.waitUntilStopped()
        XCTAssertEqual(PreparationProtocol.counts().0,2)
    }
    @MainActor func testCancelPreparedParallelGroupNeverStartsAWorker() async throws {
        let (store,project,root) = try await fixture()
        defer { store.shutdown();try? FileManager.default.removeItem(at:root) }
        var prepared = false
        store.cloudPrepared = { _,_,_ in
            prepared = true
            while true { try await Task.sleep(for:.milliseconds(10)) }
        }
        store.setSharedConfiguration(.init(enabled:false,agentCount:2,intervalMinutes:0,parallelEnabled:true),for:project.id)
        store.startCloudResearch(project,request:"Fixture request")
        try await wait { prepared }
        store.stopSharedResearch(project.id)
        await store.sharedCycles[project.id]?.waitUntilStopped()
        XCTAssertEqual(PreparationProtocol.counts().0,0);XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertEqual(store.sharedRuns[project.id]?.phase,.stopped)
    }
    @MainActor func testExternalVerificationWaitsBeforeAnyRecoveryHTTPRequest() async throws {
        let (store,_,root) = try await fixture()
        defer { store.shutdown();try? FileManager.default.removeItem(at:root) }
        let client = try store.cloudClient()
        let box = try await store.createCloudBox(client,role:"worker",group:nil)
        var waiting = false,release = false
        store.cloudRecoveryReady = { id in
            XCTAssertEqual(id,box);waiting = true
            while !release { try await Task.sleep(for:.milliseconds(10)) }
        }
        let task = Task { try await store.recoverCloudBox(box,client:client) }
        try await wait { waiting }
        XCTAssertEqual(PreparationProtocol.packCount(),0)
        release = true
        do { try await task.value;XCTFail("Fixture pack must fail after admission") } catch { }
        XCTAssertEqual(PreparationProtocol.packCount(),1)
        XCTAssertFalse(store.cloudCleanup.contains(box))
        XCTAssertEqual(PreparationProtocol.counts().0,0)
    }
}
