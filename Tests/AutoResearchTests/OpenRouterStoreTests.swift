import XCTest
import ResearchCore
@testable import AutoResearch

private final class FixtureGatewayCredentials: OpenRouterCredentialStore {
    var value: String?
    var reads = 0
    func containsKey() -> Bool { value != nil }
    func read() throws -> String { reads += 1; guard let value else { throw ResearchError("Fixture key missing") }; return value }
    func save(_ value: String) throws { self.value = value }
    func remove() throws { value = nil }
}

final class OpenRouterStoreTests: XCTestCase {
    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private func assertNoCredential(_ credential: String, under folder: URL) throws {
        let files = FileManager.default.enumerator(at: folder, includingPropertiesForKeys: [.isRegularFileKey])!
        for case let file as URL in files where (try? file.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile) == true {
            XCTAssertFalse(String(decoding: try Data(contentsOf: file), as: UTF8.self).contains(credential), "Credential persisted at \(file.path)")
        }
    }
    @MainActor private func fixture() throws -> (AppStore, URL, FixtureGatewayCredentials) {
        let folder = try Workspace.directory(root.path, "tmp/gateway-test-" + UUID().uuidString)
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.example.json")))
        settings.workspaceRoot = folder.path
        let cli = folder.appendingPathComponent("fixture-cli")
        try #"""
        #!/usr/bin/python3
        import json,os,sys
        from pathlib import Path
        if 'mcp' in sys.argv:
            assert not os.environ.get('OPENROUTER_API_KEY')
            print('[]');sys.exit(0)
        sys.stdin.read()
        secret=os.environ.get('OPENROUTER_API_KEY','missing')
        assert secret != 'missing'
        Path('args.json').write_text(json.dumps(sys.argv))
        print(secret,file=sys.stderr)
        print(json.dumps({'type':'thread.started','thread_id':'fixture-gateway-session'}))
        print(json.dumps({'type':'item.completed','item':{'type':'agent_message','id':'answer','text':secret+'\nRESEARCH_STATUS: finished'}}))
        print(json.dumps({'type':'turn.completed'}))
        """#.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cli.path)
        settings.codexPath = cli.path
        let credentials = FixtureGatewayCredentials()
        let store = AppStore(resourceDirectory: root.appendingPathComponent("Resources"), defaults: settings, gatewayCredentials: credentials)
        return (store, folder, credentials)
    }
    @MainActor func testSettingsPersistWithoutKeyAndActiveRunsRejectKeyChanges() throws {
        let (store, folder, credentials) = try fixture()
        XCTAssertFalse(store.gateway.enabled)
        XCTAssertTrue(store.saveGatewayKey("fixture-secret"))
        store.gateway = .init(enabled: true, codexModel: "fixture/model")
        XCTAssertTrue(store.saveGatewayConfiguration())
        let data = try Data(contentsOf: folder.appendingPathComponent("openrouter.json"))
        XCTAssertFalse(String(decoding: data, as: UTF8.self).contains("fixture-secret"))
        XCTAssertEqual(credentials.reads, 0)
        store.runners[UUID()] = ProcessRunner()
        XCTAssertFalse(store.saveGatewayKey("replacement"))
        XCTAssertFalse(store.removeGatewayKey())
        store.gateway.enabled = false
        XCTAssertFalse(store.saveGatewayConfiguration())
        XCTAssertTrue(store.gateway.enabled)
        XCTAssertEqual(credentials.value, "fixture-secret")
    }
    @MainActor func testMissingModelKeyAndCorruptConfigurationFailBeforeLaunch() throws {
        let (store, folder, _) = try fixture()
        store.gateway = .init(enabled: true, codexModel: "fixture/model")
        XCTAssertFalse(store.saveGatewayConfiguration())
        XCTAssertTrue(store.saveGatewayKey("fixture-secret"))
        store.gateway = .init(enabled: true)
        XCTAssertFalse(store.saveGatewayConfiguration())
        try "unreadable".write(to: folder.appendingPathComponent("openrouter.json"), atomically: true, encoding: .utf8)
        store.loadGatewayConfiguration()
        XCTAssertFalse(store.gatewayConfigurationReadable)
        XCTAssertThrowsError(try store.gatewayPreflight(ResearchSession(projectID: "p", title: "x", provider: .codex), cloud: false))
    }
    @MainActor func testFakeCLIRedactsBothStreamsBeforeChatAndLogPersistence() async throws {
        let (store, folder, credentials) = try fixture()
        XCTAssertTrue(store.saveGatewayKey("fixture-openrouter-secret"))
        store.gateway = .init(enabled: true, codexModel: "fixture/model")
        XCTAssertTrue(store.saveGatewayConfiguration())
        let checkout = try Workspace.directory(folder.path, "project")
        let project = ResearchProject(repository: "", localPath: checkout.path, isCustom: true)
        let session = ResearchSession(projectID: project.id, title: "Fixture", provider: .codex)
        store.projects.append(project); store.sessions.append(session)
        store.launch(session.id, project: project, request: "Fixture only")
        await store.workerTasks[session.id]?.value
        let saved = try XCTUnwrap(store.sessions.first { $0.id == session.id })
        XCTAssertEqual(saved.state, "Completed", saved.messages.map(\.text).joined(separator: "\n"))
        XCTAssertEqual(saved.connectionRoute, "openrouter:codex")
        XCTAssertEqual(saved.connectionModel, "fixture/model")
        XCTAssertEqual(credentials.reads, 1)
        XCTAssertTrue(saved.messages.map(\.text).joined().contains("[REDACTED]"))
        XCTAssertFalse(saved.messages.map(\.text).joined().contains("fixture-openrouter-secret"))
        try assertNoCredential("fixture-openrouter-secret", under: folder)
        store.gateway.enabled = false
        XCTAssertThrowsError(try store.gatewayPreflight(saved, cloud: false))
    }
    @MainActor func testCloudGatewayRejectedBeforeCredentialRead() throws {
        let (store, _, credentials) = try fixture()
        store.gateway = .init(enabled: true, codexModel: "fixture/model")
        let session = ResearchSession(projectID: "fixture", title: "Fixture", provider: .codex)
        XCTAssertThrowsError(try store.gatewayPreflight(session, cloud: true))
        XCTAssertEqual(credentials.reads, 0)
    }
}
