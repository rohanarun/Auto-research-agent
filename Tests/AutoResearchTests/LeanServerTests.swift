import XCTest
@testable import AutoResearch
import ResearchCore

@MainActor final class LeanServerTests: XCTestCase {
    func testAppOwnedServerStartsAndDrains() async throws {
        guard ProcessInfo.processInfo.environment["AUTORESEARCH_LEAN_LIVE_TEST"] == "1" else { throw XCTSkip("Opt-in installed Lean service test") }
        let project = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let workspace = project.appendingPathComponent("tmp/native-lean-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: workspace, withIntermediateDirectories: true)
        let server = LeanServerController()
        try server.configure(workspace: workspace.path, resources: project.appendingPathComponent("Resources"))
        server.start()
        for _ in 0..<200 where !server.running && server.isActive { try await Task.sleep(for: .milliseconds(50)) }
        XCTAssertTrue(server.running, server.status)
        XCTAssertTrue(server.localURL.hasPrefix("http://127.0.0.1:"))
        XCTAssertNotNil(server.connectionPath)
        XCTAssertTrue(try server.remoteConnectionFiles().isEmpty)
        let fake = workspace.appendingPathComponent("ngrok-fixture")
        let args = workspace.appendingPathComponent("ngrok-arguments.json")
        let script = "#!/usr/bin/python3\nimport json,sys,signal\nfrom pathlib import Path\nPath(" + String(reflecting: args.path) + ").write_text(json.dumps(sys.argv[1:]))\nprint(json.dumps({'url':'https://lean-fixture.invalid'}),flush=True)\nsignal.pause()\n"
        try script.write(to: fake, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: fake.path)
        server.ngrokPath = fake.path
        server.share()
        for _ in 0..<200 where server.publicURL.isEmpty { try await Task.sleep(for: .milliseconds(25)) }
        XCTAssertEqual(server.publicURL, "https://lean-fixture.invalid")
        let arguments = try JSONDecoder().decode([String].self, from: Data(contentsOf: args))
        XCTAssertTrue(arguments.contains("--inspect=false"))
        XCTAssertTrue(arguments.contains(server.localURL))
        XCTAssertEqual(Set(try server.remoteConnectionFiles().keys), ["connection.json", "server-token"])
        server.stopSharing()
        for _ in 0..<200 where server.sharing { try await Task.sleep(for: .milliseconds(25)) }
        XCTAssertTrue(server.running, "Closing the public tunnel must keep localhost running")
        XCTAssertTrue(try server.remoteConnectionFiles().isEmpty)
        server.stop()
        for _ in 0..<200 where server.isActive { try await Task.sleep(for: .milliseconds(50)) }
        XCTAssertFalse(server.isActive, "Server must drain before application exit")
    }
}
