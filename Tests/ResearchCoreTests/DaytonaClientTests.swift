import XCTest
import CryptoKit
@testable import ResearchCore

private final class DaytonaProtocol: URLProtocol, @unchecked Sendable {
    static var handler: ((URLRequest) throws -> Data)?
    static var statusCode = 200
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do { Self.statusCode = 200;let data = try Self.handler!(request);client?.urlProtocol(self,didReceive:HTTPURLResponse(url:request.url!,statusCode:Self.statusCode,httpVersion:nil,headerFields:nil)!,cacheStoragePolicy:.notAllowed);client?.urlProtocol(self,didLoad:data);client?.urlProtocolDidFinishLoading(self) }
        catch { client?.urlProtocol(self,didFailWithError:error) }
    }
    override func stopLoading() {}
}
final class DaytonaClientTests: XCTestCase {
    func testTransientManagementReadRecoversWithoutRepeatingLaunch() async throws {
        try await exerciseRunnerFailure(status:502,expectRecovery:true)
    }
    func testTerminalPollingFailureStopsTheSameOwnedOperation() async throws {
        try await exerciseRunnerFailure(status:401,expectRecovery:false)
    }
    private func exerciseRunnerFailure(status: Int,expectRecovery: Bool) async throws {
        let configuration = URLSessionConfiguration.ephemeral;configuration.protocolClasses = [DaytonaProtocol.self]
        let id = UUID().uuidString;let install = UUID().uuidString;var launches: [String] = [];var stops: [String] = [];var failedRead = false
        DaytonaProtocol.handler = { request in
            let object: [String:Any]
            if request.url?.host == "app.daytona.io" {
                if !launches.isEmpty,!failedRead { failedRead = true;DaytonaProtocol.statusCode = status;return Data("{}".utf8) }
                if request.url!.path.hasSuffix("preview-url") { object = ["url":"https://owned.example/","token":"preview-only"] }
                else { object = ["id":id,"state":"started","labels":["autoresearch-install":install]] }
            } else {
                var body = request.httpBody ?? Data()
                if body.isEmpty,let stream = request.httpBodyStream { stream.open();defer { stream.close() };var buffer = [UInt8](repeating:0,count:4096);while stream.hasBytesAvailable { let n = stream.read(&buffer,maxLength:buffer.count);if n <= 0 { break };body.append(buffer,count:n) } }
                let command = (try JSONSerialization.jsonObject(with:body) as! [String:Any])["command"] as! String
                let encoded = command.components(separatedBy:"'")[1]
                let payload = try JSONSerialization.jsonObject(with:Data(base64Encoded:encoded)!) as! [String:Any]
                let operation = payload["id"] as! String
                let receipt: [String:Any]
                if command.contains("bridge.py launch") { launches.append(operation);receipt = ["phase":"requested"] }
                else if command.contains("bridge.py stop") { stops.append(operation);receipt = ["phase":"ended","exit_code":-15,"cleanup":["remaining_pids":[Int]()]] }
                else { receipt = ["stdout":"","stderr":"","stdout_offset":0,"stderr_offset":0,"stdout_eof":true,"stderr_eof":true,"status":["phase":"ended","exit_code":0,"cleanup":["remaining_pids":[Int]()]]] }
                object = ["exitCode":0,"result":String(decoding:try JSONSerialization.data(withJSONObject:receipt),as:UTF8.self)]
            }
            return try JSONSerialization.data(withJSONObject:object)
        }
        defer { DaytonaProtocol.handler = nil }
        let client = try DaytonaClient(apiKey:"fixture",installID:install,knownOwnedIDs:[id],apiURL:URL(string:"https://app.daytona.io/api")!,session:URLSession(configuration:configuration),readRetryInterval:0.001)
        let options = DaytonaConfiguration(snapshot:"unused",codexPackage:"unused",reasoningEffort:"high",pollSeconds:0.001,logChunkBytes:100)
        let runner = DaytonaRunner(client:client,cloud:CloudSession(sandboxID:id,stateID:UUID().uuidString),configuration:options)
        do {
            let result = try await runner.run(Command("codex",[]),onLine:{_,_ in })
            XCTAssertTrue(expectRecovery);XCTAssertEqual(result.code,0)
        } catch { XCTAssertFalse(expectRecovery);XCTAssertTrue(error.localizedDescription.contains("401"),error.localizedDescription) }
        XCTAssertEqual(launches.count,1);XCTAssertTrue(failedRead)
        if expectRecovery { XCTAssertEqual(stops.count,0) }
        else { XCTAssertEqual(stops,launches,"Failure cleanup must stop the exact accepted operation without relaunch") }
    }
    func testExecutionPreferenceNeverMovesAnExistingChatToAnotherBackend() {
        var chat = ResearchSession(projectID:"owner/repo",title:"New",provider:.codex)
        XCTAssertFalse(chat.usesCloud(preferCloud:false));XCTAssertTrue(chat.usesCloud(preferCloud:true))
        chat.workingDirectory = "/existing/local/checkout"
        XCTAssertFalse(chat.usesCloud(preferCloud:true))
        chat.workingDirectory = nil;chat.providerSessionID = "existing-local-provider-id"
        XCTAssertFalse(chat.usesCloud(preferCloud:true))
        chat.cloud = CloudSession(sandboxID:UUID().uuidString,stateID:UUID().uuidString)
        XCTAssertTrue(chat.usesCloud(preferCloud:false))
    }
    func testRetainedCloudArchiveWhenExplicitlyProvided() throws {
        guard let folder = ProcessInfo.processInfo.environment["RESEARCH_DAYTONA_ARCHIVE_FIXTURE"] else { throw XCTSkip("A retained cloud archive is opt-in.") }
        let p = URL(fileURLWithPath:folder);let archive = p.appendingPathComponent("checkout-state.tar.gz")
        _ = try DaytonaArchive.verify(archive:archive,manifest:p.appendingPathComponent("manifest.json"),expectedSHA:DaytonaArchive.sha256(archive))
    }
    func testKeyNeverLeavesManagementOriginAndOwnershipIsRequired() async throws {
        let config = URLSessionConfiguration.ephemeral;config.protocolClasses = [DaytonaProtocol.self]
        let session = URLSession(configuration:config);let id = UUID().uuidString;let install = UUID().uuidString;var paths: [String] = []
        DaytonaProtocol.handler = { req in
            paths.append(req.url!.host! + req.url!.path)
            let object: [String:Any]
            if req.url!.host == "app.daytona.io" {
                XCTAssertEqual(req.value(forHTTPHeaderField:"Authorization"),"Bearer fixture-key")
                XCTAssertNil(req.value(forHTTPHeaderField:"X-Daytona-Preview-Token"))
                if req.url!.path.hasSuffix("preview-url") { object = ["url":"https://private-sandbox.example/","token":"scoped-preview"] }
                else { object = ["id":id,"state":"started","labels":["autoresearch-install":install]] }
            } else {
                XCTAssertNil(req.value(forHTTPHeaderField:"Authorization"));XCTAssertEqual(req.value(forHTTPHeaderField:"X-Daytona-Preview-Token"),"scoped-preview")
                XCTAssertFalse(String(decoding:req.httpBody ?? Data(),as:UTF8.self).contains("fixture-key"))
                object = ["exitCode":0,"result":"ok"]
            }
            return try JSONSerialization.data(withJSONObject:object)
        }
        defer { DaytonaProtocol.handler = nil;session.invalidateAndCancel() }
        let client = try DaytonaClient(apiKey:"fixture-key",installID:install,knownOwnedIDs:[id],apiURL:URL(string:"https://app.daytona.io/api")!,session:session)
        let output = try await client.execute(id,command:"printf ok");XCTAssertEqual(output,"ok");XCTAssertEqual(paths.count,3)
        do { _ = try await client.inspect(UUID().uuidString);XCTFail("Unowned sandbox accepted") } catch { XCTAssertEqual(paths.count,3) }
        XCTAssertThrowsError(try DaytonaClient(apiKey:"fixture-key",installID:install,apiURL:URL(string:"https://other.example/api")!))
    }
    func testChangedOwnershipCannotStopAnotherSandbox() async throws {
        let config = URLSessionConfiguration.ephemeral;config.protocolClasses = [DaytonaProtocol.self];let id = UUID().uuidString
        DaytonaProtocol.handler = { req in
            XCTAssertEqual(req.httpMethod,"GET")
            return try JSONSerialization.data(withJSONObject:["id":id,"state":"started","labels":["autoresearch-install":"different-install"]])
        }
        defer { DaytonaProtocol.handler = nil }
        let c = try DaytonaClient(apiKey:"fixture",installID:"ours",knownOwnedIDs:[id],apiURL:URL(string:"https://app.daytona.io/api")!,session:URLSession(configuration:config))
        do { try await c.stop(id,archive:true);XCTFail("Foreign sandbox stopped") } catch { XCTAssertTrue(error.localizedDescription.contains("ownership")) }
    }
    func testRecoveryArchiveChecksEveryEntryAndRejectsCredentialsOrAlteration() async throws {
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = try Workspace.directory(root.path,"tmp/archive-verification-" + UUID().uuidString);defer { try? FileManager.default.removeItem(at:folder) }
        let payload = Data("kept evidence".utf8);try payload.write(to:folder.appendingPathComponent("evidence.txt"))
        let archive = folder.appendingPathComponent("state.tar.gz")
        let result = try await ProcessRunner().run(Command("/usr/bin/tar",["--disable-copyfile","-czf",archive.path,"evidence.txt"]),directory:folder);XCTAssertEqual(result.code,0)
        let manifest = folder.appendingPathComponent("manifest.json")
        let digest = SHA256.hash(data:payload).map{String(format:"%02x",$0)}.joined()
        func save(hash: String) throws { try JSONSerialization.data(withJSONObject:["all_provider_credentials_excluded":true,"entries":["evidence.txt":["kind":"file","size":payload.count,"sha256":hash]]]).write(to:manifest) }
        try save(hash:digest);let sha = try DaytonaArchive.sha256(archive)
        XCTAssertEqual(try DaytonaArchive.verify(archive:archive,manifest:manifest,expectedSHA:sha),1)
        try save(hash:"altered");XCTAssertThrowsError(try DaytonaArchive.verify(archive:archive,manifest:manifest,expectedSHA:sha))
        try payload.write(to:folder.appendingPathComponent("auth.json"))
        let extra = try await ProcessRunner().run(Command("/usr/bin/tar",["--disable-copyfile","-czf",archive.path,"evidence.txt","auth.json"]),directory:folder);XCTAssertEqual(extra.code,0)
        try save(hash:digest);XCTAssertThrowsError(try DaytonaArchive.verify(archive:archive,manifest:manifest,expectedSHA:DaytonaArchive.sha256(archive)))
    }
}
