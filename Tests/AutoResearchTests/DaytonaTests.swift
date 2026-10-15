import XCTest
import SwiftUI
import AppKit
import ResearchCore
@testable import AutoResearch

final class DaytonaTests: XCTestCase {
    var root: URL { URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func testKeychainRoundTripStoresKeyOutsideConfiguration() throws {
        let account = "test-" + UUID().uuidString
        defer { DaytonaKeychain.remove(account) }
        try DaytonaKeychain.save("fixture-only",account:account)
        XCTAssertEqual(try DaytonaKeychain.read(account),"fixture-only")
        try DaytonaKeychain.save("updated-fixture",account:account)
        XCTAssertEqual(try DaytonaKeychain.read(account),"updated-fixture")
        DaytonaKeychain.remove(account);XCTAssertThrowsError(try DaytonaKeychain.read(account))
    }
    @MainActor func testDaytonaSettingsPanelRendersWithoutExposingKey() async throws {
        _ = NSApplication.shared
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:root.appendingPathComponent("Resources/Defaults.json")))
        settings.workspaceRoot = try Workspace.directory(root.path,"tmp/daytona-ui-" + UUID().uuidString).path
        defer { try? FileManager.default.removeItem(atPath:settings.workspaceRoot) }
        let store = AppStore(resourceDirectory:root.appendingPathComponent("Resources"),defaults:settings)
        XCTAssertFalse(store.daytona.enabled);XCTAssertNil(store.daytonaClient)
        let data = try JSONEncoder().encode(store.daytona)
        XCTAssertFalse(String(decoding:data,as:UTF8.self).contains("apiKey"))
        let host = NSHostingView(rootView:ConnectionsView(store:store,initialTab:"Daytona").environment(\.colorScheme,.light).background(Color.white));host.frame = NSRect(x:0,y:0,width:710,height:610)
        let window = NSWindow(contentRect:host.frame,styleMask:[.borderless],backing:.buffered,defer:false);window.isReleasedWhenClosed = false;window.contentView = host;host.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(250));host.displayIfNeeded()
        let image = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in:host.bounds));host.cacheDisplay(in:host.bounds,to:image)
        let png = try XCTUnwrap(image.representation(using:.png,properties:[:]))
        let output = try Workspace.directory(root.path,"evidence/release-1.4.0");try png.write(to:output.appendingPathComponent("daytona-settings.png"))
    }
    @MainActor func testLiveCloudDeviceLoginStreamsCodeAndCancels() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["RESEARCH_LIVE_DAYTONA_LOGIN_TEST"] == "1",let keyFile = env["RESEARCH_DAYTONA_KEY_FILE"] else { throw XCTSkip("Cloud device-login initiation is opt-in.") }
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:root.appendingPathComponent("Resources/Defaults.json")))
        let evidence = try Workspace.directory(root.path,"evidence/live-daytona-login-" + UUID().uuidString)
        settings.workspaceRoot = try Workspace.directory(evidence.path,"runtime").path
        let store = AppStore(resourceDirectory:root.appendingPathComponent("Resources"),defaults:settings)
        let client = try DaytonaClient(apiKey:String(contentsOfFile:keyFile,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines),installID:store.daytona.installID,apiURL:URL(string:"https://app.daytona.io/api")!)
        store.daytonaClient = client
        store.connectDaytona(signIn:true)
        let deadline = Date().addingTimeInterval(120)
        while !store.daytonaLoginOutput.contains("auth.openai.com/codex/device") {
            if let error = store.error { throw ResearchError(error) }
            guard Date() < deadline else { store.cancelCloudLogin();throw ResearchError("Device login did not expose its official URL.") }
            try await Task.sleep(for:.milliseconds(250))
        }
        XCTAssertNotNil(store.daytonaLoginOutput.range(of:"[A-Z0-9]{4}-[A-Z0-9]{5}",options:.regularExpression))
        store.cancelCloudLogin()
        while store.daytonaBusy { try await Task.sleep(for:.milliseconds(250)) }
        XCTAssertEqual(store.sessions.count,0,"Sign-in must not launch research")
        try Workspace.write(["official_device_url_streamed":true,"code_streamed":true,"cancelled_without_research":true,"host_auth_read":false],to:evidence.appendingPathComponent("login-proof.json"))
        if let id = store.daytona.connectionSandboxID { try await client.stop(id,archive:false) }
    }
    @MainActor func testLiveDaytonaTwoWorkersAndFreshCheckpointSuccessors() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["RESEARCH_LIVE_DAYTONA_TEST"] == "1",let keyFile = env["RESEARCH_DAYTONA_KEY_FILE"],let authBridge = env["RESEARCH_DAYTONA_TEST_AUTH_BRIDGE"] else { throw XCTSkip("Actual Daytona and cloud model transport is opt-in.") }
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:root.appendingPathComponent("Resources/Defaults.json")))
        let evidence = try Workspace.directory(root.path,"evidence/live-daytona-release-" + UUID().uuidString)
        settings.workspaceRoot = try Workspace.directory(evidence.path,"runtime").path
        settings.codexModel = env["RESEARCH_DAYTONA_TEST_MODEL"] ?? "";settings.codexSandbox = "workspace-write"
        let store = AppStore(resourceDirectory:root.appendingPathComponent("Resources"),defaults:settings)
        store.daytona.enabled = true;store.daytona.pollSeconds = 1
        let client = try DaytonaClient(apiKey:String(contentsOfFile:keyFile,encoding:.utf8).trimmingCharacters(in:.whitespacesAndNewlines),installID:store.daytona.installID,apiURL:URL(string:"https://app.daytona.io/api")!)
        store.daytonaClient = client
        let connection = try await store.createCloudBox(client,role:"connection",group:nil)
        store.daytona.connectionSandboxID = connection;try store.saveDaytona()
        try await store.setupCloudBox(connection,client:client)
        // The opt-in bridge runs entirely inside the already authorized cloud
        // controller. It returns only a receipt; no credential bytes reach Mac.
        let bridge = try await ProcessRunner().run(Command("/usr/bin/python3",[authBridge,connection]),directory:evidence,environment:try store.environment())
        guard bridge.code == 0 else { throw ResearchError("Cloud-only test authentication bridge failed; original account files were not read.") }
        let source = try Workspace.directory(evidence.path,"source")
        try "Transport test repository".write(to:source.appendingPathComponent("README.md"),atomically:true,encoding:.utf8)
        for args in [["init","-q"],["add","."],["-c","user.name=Test","-c","user.email=test@localhost","-c","commit.gpgsign=false","-c","core.hooksPath=/dev/null","commit","-qm","fixture"]] {
            let r = try await ProcessRunner().run(Command("/usr/bin/git",args),directory:source);XCTAssertEqual(r.code,0)
        }
        let project = ResearchProject(repository:"",name:"Daytona transport verification",localPath:source.path,isCustom:true)
        store.projects = [project];store.select(project);store.provider = .codex
        store.setSharedConfiguration(.init(enabled:true,agentCount:2,intervalMinutes:10),for:project.id)
        let request = """
        This is a small checkpoint transport verification, not a research experiment. If a shared checkpoint bundle is supplied later in this very prompt, do not use tools: return all CLOUD_TRANSPORT_ markers present in its checkpoint contents and RESEARCH_STATUS: finished. Otherwise use one shell command to write RESEARCH_CHECKPOINT.md atomically in /workspace/checkout, containing a new CLOUD_TRANSPORT_ marker built from an OS-generated UUID. Use Python uuid.uuid4() for the marker: this restricted Linux runner does not expose /proc/sys/kernel/random/uuid, so reading that proc path previously failed before a checkpoint could be written. Write only the marker bytes, without an appended newline or literal backslash escapes. Use your own assigned checkout, which is writable; do not probe the parent volume root. Return the exact marker and RESEARCH_STATUS: finished. Do not install anything, modify other files, launch agents, use web/MCP, or read any other state.
        """
        store.startCloudResearch(project,request:request)
        func wait(_ condition: () -> Bool) async throws {
            let deadline = Date().addingTimeInterval(300)
            while !condition() {
                if let error = store.error { throw ResearchError(error) }
                guard Date() < deadline else { throw ResearchError("Live cloud verification deadline expired; inspect retained owned-worker receipts.") }
                try await Task.sleep(for:.milliseconds(250))
            }
        }
        do {
            try await wait { store.sessions.count == 2 && store.sessions.allSatisfy{$0.state == "Completed"} }
            let initial = store.sessions
            var markers: [String] = []
            for session in initial {
                let snap = try await client.call(session.cloud!.sandboxID,action:"checkpoint")
                guard let encoded = snap["checkpoint"] as? String,let bytes = Data(base64Encoded:encoded) else { throw ResearchError("Live model did not write its checkpoint.") }
                markers.append(String(decoding:bytes,as:UTF8.self).trimmingCharacters(in:.whitespacesAndNewlines))
            }
            store.synchronizeSharedResearch(project.id)
            try await wait { store.sessions.count == 4 && store.sessions.suffix(2).allSatisfy{$0.state == "Completed"} }
            XCTAssertEqual(Set(store.sessions.compactMap(\.providerSessionID)).count,4)
            for successor in store.sessions.suffix(2) {
                let answer = successor.messages.filter{$0.role == "assistant"}.map(\.text).joined(separator:"\n")
                for marker in markers { XCTAssertTrue(answer.contains(marker),answer) }
                XCTAssertFalse(successor.messages.contains{$0.role == "tool" && !$0.text.hasPrefix("CLI diagnostics")},"Successors must use injected context without rereading files")
            }
            store.stopSharedResearch(project.id);await store.sharedCycles[project.id]?.waitUntilStopped()
            try await Task.sleep(for:.seconds(1))
            try await wait { store.cloudCleanup.isEmpty }
            for s in initial { let box = try await client.inspect(s.cloud!.sandboxID);XCTAssertEqual(box.state,"archived") }
            try Workspace.write(store.sessions,to:evidence.appendingPathComponent("sessions.json"))
            try Workspace.write(["fresh_provider_sessions":4,"initial_workers":2,"fresh_successors":2,"synthesis_calls":0,"verified_archived_workers":2],to:evidence.appendingPathComponent("transport-proof.json"))
            print("Actual Daytona app transport verified: " + evidence.path)
        } catch {
            store.stopSharedResearch(project.id);await store.sharedCycles[project.id]?.waitUntilStopped()
            for box in store.cloudOwnedIDs { try? await client.stop(box,archive:false) }
            throw error
        }
        // Exercise real Linux cancellation without another model invocation.
        try await client.copyCloudAuthentication(from:connection,to:connection)
        let fixture = """
        #!/usr/local/bin/python3
        import subprocess,time,json,uuid
        subprocess.Popen(['sleep','60'])
        print(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}),flush=True)
        while True:time.sleep(.1)
        """
        try await client.upload(connection,files:["/opt/autoresearch/bin/codex":Data(fixture.utf8)])
        _ = try await client.execute(connection,command:"sudo -n chmod 755 /opt/autoresearch/bin/codex")
        let backend = DaytonaRunner(client:client,cloud:CloudSession(sandboxID:connection,stateID:"cancellation-fixture"),configuration:store.daytona)
        let task = Task { try await backend.run(Command("codex",[]),onLine:{ _,_ in }) }
        try await Task.sleep(for:.seconds(2));backend.stop()
        let cancelled = try await task.value;XCTAssertNotEqual(cancelled.code,0)
        try Workspace.write(["owned_remote_cli_drained":true,"model_invoked":false],to:evidence.appendingPathComponent("linux-cancellation-proof.json"))
        try await store.recoverCloudBox(connection,client:client)
    }
}
