import XCTest
import CryptoKit
@testable import ResearchCore
@testable import AutoResearch

/// Local-only capture of the real AppStore -> CLI -> DaytonaRunner request.
/// No HTTP request leaves URLProtocol and no model process is started.
private final class ParityCaptureProtocol: URLProtocol, @unchecked Sendable {
    static var install = ""
    static var box = ""
    static var launches: [[String:Any]] = []
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let object: [String:Any]
            if request.url?.host == "app.daytona.io" {
                if request.url!.path.hasSuffix("preview-url") { object = ["url":"https://parity.invalid/","token":"local-fixture"] }
                else { object = ["id":Self.box,"state":"started","labels":["autoresearch-install":Self.install]] }
            } else {
                guard request.url?.host == "parity.invalid" else { throw ResearchError("Unexpected fixture origin") }
                var data = request.httpBody ?? Data()
                if let stream = request.httpBodyStream, data.isEmpty {
                    stream.open();defer { stream.close() }
                    var buffer = [UInt8](repeating:0,count:4096)
                    while stream.hasBytesAvailable { let n = stream.read(&buffer,maxLength:buffer.count);if n <= 0 { break };data.append(buffer,count:n) }
                }
                let body = try JSONSerialization.jsonObject(with:data) as! [String:Any]
                let command = body["command"] as! String
                let payload = try JSONSerialization.jsonObject(with:Data(base64Encoded:command.components(separatedBy:"'")[1])!) as! [String:Any]
                let result: [String:Any]
                if command.contains("bridge.py launch") { Self.launches.append(payload);result = ["phase":"requested"] }
                else {
                    // Deliberate local-only terminal status avoids production
                    // archive operations. This is a capture fixture, not a trial.
                    result = ["stdout":"","stderr":"","stdout_offset":0,"stderr_offset":0,"stdout_eof":true,"stderr_eof":true,"status":["phase":"ended","exit_code":1]]
                }
                object = ["exitCode":0,"result":String(decoding:try JSONSerialization.data(withJSONObject:result),as:UTF8.self)]
            }
            client?.urlProtocol(self,didReceive:HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:nil,headerFields:nil)!,cacheStoragePolicy:.notAllowed)
            client?.urlProtocol(self,didLoad:try JSONSerialization.data(withJSONObject:object));client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self,didFailWithError:error) }
    }
    override func stopLoading() {}
}

final class BenchmarkPromptParityTests: XCTestCase {
    @MainActor func testExportActualNativeRequestsForMatchedCLIControlWithoutInference() async throws {
        guard let destination = ProcessInfo.processInfo.environment["RESEARCH_PARITY_EXPORT_ROOT"] else { throw XCTSkip("Explicit local-only matched benchmark export") }
        let output = try Workspace.validate(destination)
        let app = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let resources = app.appendingPathComponent("Resources")
        let contract = try JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("control-settings.json"))) as! [String:Any]
        let protocolObject = try JSONSerialization.jsonObject(with:Data(contentsOf:output.appendingPathComponent("protocol.json"))) as! [String:Any]
        let request = try String(contentsOf:output.appendingPathComponent("reference/task.base.md"),encoding:.utf8) + "\n\n" + String(contentsOf:output.appendingPathComponent("reference/endpoint-correction.md"),encoding:.utf8) + "\n\n" + String(contentsOf:output.appendingPathComponent("reference/common-platform.md"),encoding:.utf8)
        var settings = try JSONDecoder().decode(Settings.self,from:Data(contentsOf:resources.appendingPathComponent("Defaults.json")))
        let runtime = try Workspace.directory(output.path,"parity-fixtures/" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:runtime) }
        settings.workspaceRoot = runtime.path;settings.codexModel = contract["model"] as! String;settings.codexSandbox = contract["sandbox"] as! String
        let store = AppStore(resourceDirectory:resources,defaults:settings)
        store.daytona.enabled = true;store.daytona.reasoningEffort = contract["reasoning_effort"] as! String
        store.daytona.codexPackage = contract["codex_package"] as! String
        let policyBytes = try Data(contentsOf:resources.appendingPathComponent("ResearchPolicy.md"))
        store.policy = String(decoding:policyBytes,as:UTF8.self)
        var records: [[String:Any]] = []
        let cells: [(Bool,Int,Int)] = (protocolObject["agent_counts_without_sharing"] as! [Int]).map { (false,$0,0) } + (protocolObject["sharing_cadence_seconds"] as! [Int]).flatMap { interval in (protocolObject["agent_counts_with_sharing"] as! [Int]).map { (true,$0,interval) } }
        for (sharing,count,interval) in cells {
            for generation in 1...(sharing ? 2 : 1) {
                let context = runtime.appendingPathComponent("shared/round-1/shared-context.md")
                if generation > 1 {
                    try FileManager.default.createDirectory(at:context.deletingLastPathComponent(),withIntermediateDirectories:true)
                    // A fixed replay payload tests byte-preserving reinjection;
                    // actual research checkpoints will differ between trials.
                    try "LOCAL TRANSPORT FIXTURE ONLY\nAgent checkpoint: an incomplete experiment; no research result.\n".write(to:context,atomically:true,encoding:.utf8)
                }
                for index in 1...count {
                    let project = ResearchProject(repository:"",name:"Local parity fixture",localPath:runtime.path,isCustom:true)
                    var session = ResearchSession(projectID:project.id,title:"Local parity fixture",provider:.codex)
                    let box = UUID().uuidString
                    session.cloud = CloudSession(sandboxID:box,stateID:session.id.uuidString,previousStateID:generation > 1 ? UUID().uuidString : nil)
                    if count > 1 || sharing { session.sharedAgent = SharedAgent(groupID:UUID(),index:index,generation:generation,previousSessionID:nil,contextFile:generation > 1 ? context.path : nil,sharesContext:sharing,agentCount:count) }
                    store.sessions = [session]
                    let config = URLSessionConfiguration.ephemeral;config.protocolClasses = [ParityCaptureProtocol.self]
                    ParityCaptureProtocol.install = store.daytona.installID;ParityCaptureProtocol.box = box;ParityCaptureProtocol.launches = []
                    store.daytonaClient = try DaytonaClient(apiKey:"local-fixture",installID:store.daytona.installID,knownOwnedIDs:[box],apiURL:URL(string:"https://app.daytona.io/api")!,session:URLSession(configuration:config))
                    store.launchCloud(session.id,project:project,request:request)
                    await store.workerTasks[session.id]?.value
                    XCTAssertEqual(ParityCaptureProtocol.launches.count,1)
                    let captured = try XCTUnwrap(ParityCaptureProtocol.launches.first)
                    let input = try XCTUnwrap(captured["input"] as? String)
                    let args = try XCTUnwrap(captured["arguments"] as? [String])
                    XCTAssertFalse(args.contains("resume"));XCTAssertTrue(args.contains("features.memories=false"));XCTAssertTrue(args.contains("features.multi_agent=false"))
                    XCTAssertTrue(input.contains("Baseline" ) || input.contains("baseline"))
                    XCTAssertTrue(input.contains("ANY legal construction"))
                    XCTAssertTrue(input.contains("strictly greater than the supplied baseline kappa"))
                    XCTAssertFalse(input.contains("advance through ALL THREE"))
                    XCTAssertFalse(input.contains("endpoint remains first frozen-checker valid=true"))
                    records.append(["sharing":sharing,"agents":count,"interval_seconds":interval,"agent_index":index,"generation":generation,"arguments":args,"input":input,"input_sha256":SHA256.hash(data:Data(input.utf8)).map { String(format:"%02x",$0) }.joined(),"source":"actual production launchCloud -> CLI.chat -> DaytonaRunner captured locally","fixture_context":generation > 1])
                }
            }
        }
        XCTAssertEqual(records.count,cells.reduce(0) { $0 + $1.1 * ($1.0 ? 2 : 1) })
        let result: [String:Any] = ["status":"local_capture_only_not_actual_provider_parity","model_calls":0,"requests":records,"request_count":records.count,"actual_provider_context_audit_required":true,"research_policy_sha256":SHA256.hash(data:policyBytes).map { String(format:"%02x",$0) }.joined()]
        try JSONSerialization.data(withJSONObject:result,options:[.prettyPrinted,.sortedKeys]).write(to:output.appendingPathComponent("native-request-contracts.json"),options:.atomic)
    }
}
