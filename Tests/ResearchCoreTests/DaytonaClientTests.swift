import XCTest
import CryptoKit
@testable import ResearchCore

private final class DaytonaProtocol: URLProtocol, @unchecked Sendable {
    static var handler: ((URLRequest) throws -> Data)?
    override class func canInit(with request: URLRequest) -> Bool { true }
    override class func canonicalRequest(for request: URLRequest) -> URLRequest { request }
    override func startLoading() {
        do { let data = try Self.handler!(request);client?.urlProtocol(self,didReceive:HTTPURLResponse(url:request.url!,statusCode:200,httpVersion:nil,headerFields:nil)!,cacheStoragePolicy:.notAllowed);client?.urlProtocol(self,didLoad:data);client?.urlProtocolDidFinishLoading(self) }
        catch { client?.urlProtocol(self,didFailWithError:error) }
    }
    override func stopLoading() {}
}
final class DaytonaClientTests: XCTestCase {
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
