import XCTest
@testable import ResearchCore

private final class Agent37Protocol: URLProtocol, @unchecked Sendable {
    static var handler: ((URLRequest) throws -> (Int,[String:Any]))?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do {
            let (status,body) = try Self.handler!(request)
            client?.urlProtocol(self,didReceive:HTTPURLResponse(url:request.url!,statusCode:status,httpVersion:nil,headerFields:nil)!,cacheStoragePolicy:.notAllowed)
            client?.urlProtocol(self,didLoad:try JSONSerialization.data(withJSONObject:body));client?.urlProtocolDidFinishLoading(self)
        } catch { client?.urlProtocol(self,didFailWithError:error) }
    }
    override func stopLoading() {}
}
final class Agent37ClientTests: XCTestCase {
    let id = "ab12cd34ef"
    func client(owned: Set<String> = []) throws -> DaytonaClient {
        let config = URLSessionConfiguration.ephemeral;config.protocolClasses = [Agent37Protocol.self]
        return try DaytonaClient(apiKey:"fixture-key",installID:"test-install",knownOwnedIDs:owned,apiURL:SandboxProvider.agent37.apiURL,session:URLSession(configuration:config),readRetryInterval:0.001,provider:.agent37)
    }
    func instance(state: String = "running", installation: String = "test-install", name: String = "fixture") -> [String:Any] {
        ["id":id,"name":name,"status":state,"metadata":["autoresearch-install":installation],"resources":["cpu":2,"memory":4,"disk":4]]
    }
    func body(_ request: URLRequest) throws -> [String:Any] {
        var data = request.httpBody ?? Data()
        if let stream = request.httpBodyStream,data.isEmpty {
            stream.open();defer { stream.close() };var buffer = [UInt8](repeating:0,count:4096)
            while stream.hasBytesAvailable { let n = stream.read(&buffer,maxLength:buffer.count);if n <= 0 { break };data.append(buffer,count:n) }
        }
        return data.isEmpty ? [:] : try JSONSerialization.jsonObject(with:data) as! [String:Any]
    }
    override func tearDown() { Agent37Protocol.handler = nil;super.tearDown() }
    func testCreateStartUploadRunStopUsesDocumentedContract() async throws {
        var state = "running";var commands: [String] = [];var paths: [String] = []
        Agent37Protocol.handler = { request in
            XCTAssertEqual(request.url?.host,"api.agent37.com")
            XCTAssertEqual(request.value(forHTTPHeaderField:"Authorization"),"Bearer fixture-key")
            XCTAssertNil(request.value(forHTTPHeaderField:"X-Daytona-Preview-Token"))
            let path = request.url!.path;paths.append(path)
            let body = try self.body(request)
            if path == "/v1/instances",request.httpMethod == "POST" {
                XCTAssertEqual(body["template"] as? String,"custom-template@2")
                XCTAssertEqual(body["mount_path"] as? String,"/workspace")
                XCTAssertEqual(body["idle_timeout_seconds"] as? Int,3600)
                XCTAssertEqual(body["auto_sleep"] as? Bool,true)
                XCTAssertEqual((body["metadata"] as? [String:String])?["autoresearch-group"],"group")
                XCTAssertEqual((body["public_ports"] as? [Any])?.count,0)
            }
            if path.hasSuffix("/stop") { state = "stopped" }
            if path.hasSuffix("/exec") {
                XCTAssertEqual(body["user"] as? String,"root")
                let command = try XCTUnwrap(body["command"] as? String);commands.append(command)
                var output = "ready"
                if command.contains("bridge.py launch") { output = "{}" }
                if command.contains("bridge.py poll") {
                    output = String(decoding:try JSONSerialization.data(withJSONObject:["stdout":Data("hello\n".utf8).base64EncodedString(),"stderr":"","stdout_offset":6,"stderr_offset":0,"stdout_eof":true,"stderr_eof":true,"status":["phase":"ended","exit_code":0]]),as:UTF8.self)
                }
                return (200,["exit_code":0,"stdout":output,"stderr":"","truncated":false])
            }
            return (200,self.instance(state:state))
        }
        let c = try client()
        let box = try await c.create(role:"worker",snapshot:"custom-template@2",name:"fixture",group:"group")
        XCTAssertEqual(box.id,id)
        try await c.start(id)
        try await c.upload(id,files:["/opt/autoresearch/bridge.py":Data("bridge".utf8)])
        let config = DaytonaConfiguration(snapshot:"unused",codexPackage:"unused",reasoningEffort:"high",pollSeconds:0.001,logChunkBytes:100)
        let runner = DaytonaRunner(client:c,cloud:CloudSession(sandboxID:id,stateID:"state",sandboxProvider:.agent37),configuration:config)
        let result = try await runner.run(Command("codex",[]),onLine:{ _,_ in })
        XCTAssertEqual(result.code,0);XCTAssertEqual(result.stdout,"hello\n")
        try await c.stop(id,archive:true)
        XCTAssertFalse(paths.contains { $0.hasSuffix("/archive") || $0.contains("/sandbox/") })
        XCTAssertTrue(commands.contains { $0.contains("/workspace/.autoresearch") })
        XCTAssertFalse(commands.contains { $0.contains("sudo -n") || $0.contains("fixture-key") })
        XCTAssertEqual(commands.filter { $0.contains("bridge.py launch") }.count,1)
    }
    func testLostCreateReceiptReconcilesWithoutAnotherPost() async throws {
        var posts = 0
        Agent37Protocol.handler = { req in
            if req.httpMethod == "POST" { posts += 1;throw URLError(.networkConnectionLost) }
            return (200,["data":[self.instance(name:"unique-create"),self.instance(installation:"other")]])
        }
        let box = try await client().create(role:"worker",snapshot:"agent37-base",name:"unique-create")
        XCTAssertEqual(box.id,id);XCTAssertEqual(posts,1)
    }
    func testOwnershipValidationAndTruncationFailClosed() async throws {
        var calls = 0;var installation = "other";var truncated = false;var exitCode = 0
        Agent37Protocol.handler = { req in
            calls += 1
            if req.url!.path.hasSuffix("/exec") { return (200,["exit_code":exitCode,"stdout":"partial","stderr":"secret diagnostic","truncated":truncated]) }
            return (200,self.instance(installation:installation))
        }
        let c = try client(owned:[id])
        for invalid in ["../../escape","unknown123","AB12CD34EF"] {
            do { _ = try await c.execute(invalid,command:"true");XCTFail("unowned ID accepted") } catch {}
        }
        XCTAssertEqual(calls,0)
        do { _ = try await c.execute(id,command:"true");XCTFail("foreign metadata accepted") } catch {}
        XCTAssertEqual(calls,1)
        installation = "test-install";truncated = true
        do { _ = try await c.execute(id,command:"true");XCTFail("truncated output accepted") } catch { XCTAssertFalse(error.localizedDescription.contains("secret diagnostic")) }
        truncated = false;exitCode = 1
        do { _ = try await c.execute(id,command:"false");XCTFail("failed command accepted") } catch {}
    }
    func testSavedSessionsAndSettingsKeepDaytonaCompatibility() throws {
        let legacy = Data(#"{"sandboxID":"old","stateID":"state"}"#.utf8)
        XCTAssertEqual(try JSONDecoder().decode(CloudSession.self,from:legacy).provider,.daytona)
        let new = CloudSession(sandboxID:id,stateID:"state",sandboxProvider:.agent37)
        XCTAssertEqual(try JSONDecoder().decode(CloudSession.self,from:JSONEncoder().encode(new)).provider,.agent37)
        XCTAssertThrowsError(try DaytonaClient(apiKey:"key",installID:"test",apiURL:SandboxProvider.daytona.apiURL,provider:.agent37))
        XCTAssertThrowsError(try DaytonaClient(apiKey:"key",installID:"test",apiURL:URL(string:"https://evil.example/v1")!,provider:.agent37))
    }
    func testDiscoveryIgnoresForeignArbitraryMetadata() async throws {
        Agent37Protocol.handler = { _ in
            var other = self.instance(installation:"other")
            other["metadata"] = ["nested":["value":1],"count":2]
            return (200,["data":[other,self.instance()]])
        }
        let boxes = try await client().discoverOwned()
        XCTAssertEqual(boxes.map(\.id),[id])
    }
    func testTransientReadsRetryButUncertainCommandsNeverRepeat() async throws {
        var reads = 0;var executions = 0
        Agent37Protocol.handler = { request in
            if request.httpMethod == "GET" {
                reads += 1
                return reads == 1 ? (503,[:]) : (200,self.instance())
            }
            executions += 1;throw URLError(.networkConnectionLost)
        }
        let c = try client(owned:[id])
        do { _ = try await c.execute(id,command:"do-not-repeat");XCTFail("Uncertain command must be reported") } catch {}
        XCTAssertEqual(reads,2);XCTAssertEqual(executions,1)
    }
    func testAuthenticationTransferUsesShortLivedSignedPortWithoutAPIKeyInCommands() async throws {
        let target = "cd34ef56ab";var transferToken: String?;var actions: [String] = []
        Agent37Protocol.handler = { request in
            let path = request.url!.path
            if request.httpMethod == "GET" {
                var box = self.instance();box["id"] = path.components(separatedBy:"/").last!;return (200,box)
            }
            let body = try self.body(request)
            if path.hasSuffix("/signed-url") {
                XCTAssertEqual(body["ttl_seconds"] as? Int,60);XCTAssertEqual(body["port"] as? Int,43210)
                return (200,["url":"https://ab12cd34ef-43210.agent37.app/?a37_token=temporary"])
            }
            let command = try XCTUnwrap(body["command"] as? String)
            XCTAssertFalse(command.contains("fixture-key"))
            let encoded = command.components(separatedBy:"'")[1]
            let payload = try JSONSerialization.jsonObject(with:Data(base64Encoded:encoded)!) as! [String:Any]
            let action = command.components(separatedBy:"bridge.py ")[1];actions.append(action)
            let receipt: [String:Any]
            if action == "prepare-auth-transfer" {
                transferToken = payload["token"] as? String;XCTAssertNotNil(UUID(uuidString:transferToken!));receipt = ["port":43210]
            } else {
                XCTAssertEqual(action,"receive-auth-transfer");XCTAssertEqual(payload["token"] as? String,transferToken)
                XCTAssertEqual(payload["url"] as? String,"https://ab12cd34ef-43210.agent37.app/?a37_token=temporary")
                receipt = ["copied":true]
            }
            return (200,["exit_code":0,"stdout":String(decoding:try JSONSerialization.data(withJSONObject:receipt),as:UTF8.self),"stderr":"","truncated":false])
        }
        try await client(owned:[id,target]).copyCloudAuthentication(from:id,to:target)
        XCTAssertEqual(actions,["prepare-auth-transfer","receive-auth-transfer"])
    }
    func testTruncatedReadShrinksBudgetWithoutAdvancingOffsetsOrRepeatingLaunch() async throws {
        var budgets: [Int] = [];var mutations = 0
        Agent37Protocol.handler = { request in
            if request.httpMethod == "GET" { return (200,self.instance()) }
            let command = try XCTUnwrap(self.body(request)["command"] as? String)
            let encoded = command.components(separatedBy:"'")[1]
            let payload = try JSONSerialization.jsonObject(with:Data(base64Encoded:encoded)!) as! [String:Any]
            if command.contains("bridge.py poll") {
                budgets.append(payload["limit"] as! Int)
                XCTAssertEqual(payload["stdout_offset"] as? Int,42)
                return (200,["exit_code":0,"stdout":"{}","stderr":"","truncated":budgets.count < 3])
            }
            mutations += 1
            return (200,["exit_code":0,"stdout":"partial","stderr":"","truncated":true])
        }
        let c = try client(owned:[id])
        _ = try await c.call(id,action:"poll",payload:["id":"existing","stdout_offset":42,"limit":262144])
        XCTAssertEqual(budgets,[262144,131072,65536])
        do { _ = try await c.call(id,action:"launch",payload:["id":"new"]);XCTFail("Truncated mutation receipt must fail without retry") } catch {}
        XCTAssertEqual(mutations,1)
    }
    func testLiveAgent37PersistentBridge() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["RESEARCH_LIVE_AGENT37_TEST"] == "1",let key = env["AGENT37_API_KEY"],!key.isEmpty else { throw XCTSkip("Live Agent37 compute requires explicit opt-in and AGENT37_API_KEY.") }
        let c = try DaytonaClient(apiKey:key,installID:"smoke-" + UUID().uuidString,apiURL:SandboxProvider.agent37.apiURL,provider:.agent37)
        let box = try await c.create(role:"smoke",snapshot:env["AGENT37_TEST_TEMPLATE"] ?? "agent37-base",name:"autoresearch-smoke-" + UUID().uuidString)
        do {
            try await c.start(box.id)
            let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            try await c.upload(box.id,files:["/opt/autoresearch/bridge.py":Data(contentsOf:root.appendingPathComponent("Resources/DaytonaWorker.py")),"/workspace/checkout/smoke.txt":Data("persistent bridge".utf8)])
            let initial = try await c.call(box.id,action:"setup-status",payload:["id":"setup"])
            XCTAssertEqual(initial["phase"] as? String,"requested")
            try await c.stop(box.id,archive:false);try await c.start(box.id)
            let text = try await c.execute(box.id,command:"cat /workspace/checkout/smoke.txt")
            XCTAssertEqual(text,"persistent bridge")
            let receipt = try await c.call(box.id,action:"pack")
            XCTAssertNotNil(receipt["sha256"] as? String)
            try await c.stop(box.id,archive:true)
            let final = try await c.inspect(box.id);XCTAssertEqual(final.state,"stopped")
            print("Agent37 smoke instance stopped with disk retained: \(box.id)")
        } catch {
            try? await c.stop(box.id,archive:false)
            throw error
        }
    }
}
