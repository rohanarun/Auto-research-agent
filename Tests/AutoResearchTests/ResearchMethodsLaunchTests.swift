import XCTest
import ResearchCore
@testable import AutoResearch

final class ResearchMethodsLaunchTests: XCTestCase {
    @MainActor func testKeepResearchingReturnsCompletedBatchToCoordinator() async throws {
        let fixture = IndependentParallelTests()
        let (store, project, directory) = try await fixture.fixture(count: 2)
        defer { try? FileManager.default.removeItem(at: directory) }
        let source = URL(fileURLWithPath: try XCTUnwrap(project.localPath))
        try "0".write(to: source.appendingPathComponent("value"), atomically: true, encoding: .utf8)
        let cli = directory.appendingPathComponent("method-cli")
        try #"""
        #!/usr/bin/python3
        import sys,json,re,uuid
        from pathlib import Path
        if 'mcp' in sys.argv:print('[]');sys.exit(0)
        prompt=sys.stdin.read()
        print(json.dumps({'type':'error','message':'Reconnecting after temporary high demand'}))
        print(json.dumps({'type':'thread.started','thread_id':str(uuid.uuid4())}))
        match=re.search(r'Read (.+)/ResearchMethodsPolicy.md',prompt)
        if match:
         count=Path('coordinator-count');n=int(count.read_text())+1 if count.exists() else 1;count.write_text(str(n))
         if n==1:
          control=Path(match.group(1));ev=control/'evaluator.py'
          ev.write_text("import sys,json\nfrom pathlib import Path\np=Path(sys.argv[-1]);d=p/'.method-evaluation';d.mkdir(exist_ok=True);(d/'receipt').write_text('verified')\nprint(json.dumps({'valid':True,'score':0,'metrics':{},'evidence':['.method-evaluation/receipt']}))\n")
          (control/'plan.json').write_text(json.dumps(dict(objective='fixture',worker_prompt='Increment value and return',evaluator=str(ev),evaluation_seconds=1,direction='maximize')))
          answer='RESEARCH_METHOD_PLAN: ready'
         else:answer='Batch evidence reviewed; fixture has reached its test boundary.\nRESEARCH_STATUS: needs_input'
        else:
         p=Path('value');p.write_text(str(int(p.read_text())+1));answer='Research checkpoint saved'
        print(json.dumps({'type':'item.completed','item':{'id':'answer','type':'agent_message','text':answer}}))
        print(json.dumps({'type':'turn.completed'}))
        """#.write(to: cli, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: cli.path)
        store.settings.codexPath = cli.path
        var options = store.sharedConfiguration(project.id)
        var methods = try store.methodConfiguration(project.id)
        methods.forkAndFlush = true; methods.rsiTop = false
        methods.branches = 2; methods.rounds = 1
        methods.explorationMinutes = 0.05; methods.exploitationMinutes = 0.05
        options.methods = methods; store.setSharedConfiguration(options, for: project.id)
        store.startSharedResearch(project, request: "Run the bounded fixture with methods")
        let id = try XCTUnwrap(store.sessions.first?.id)
        store.keepResearching.insert(id)
        try await fixture.waitUntil {
            (try? String(contentsOf: source.appendingPathComponent("coordinator-count"), encoding: .utf8)) == "2"
                && store.sessions.first?.state == "Needs attention"
        }
        XCTAssertTrue(store.sessions[0].messages.contains { $0.text.contains("completed this batch") })
        XCTAssertFalse(store.sessions[0].messages.contains { $0.text.contains("has not prepared") }, "A coordinator input/completion decision is not a missing-plan failure")
        XCTAssertEqual(try String(contentsOf: source.appendingPathComponent("value"), encoding: .utf8), "0")
        XCTAssertEqual(store.sessions.count, 1)
        XCTAssertFalse(store.hasActiveRuns)
    }
    @MainActor func testCheckboxRoutesToOneCoordinatorAndInstallsTools() async throws {
        let fixture = IndependentParallelTests()
        let (store, project, _) = try await fixture.fixture(count: 5)
        var options = store.sharedConfiguration(project.id)
        var methods = try store.methodConfiguration(project.id)
        methods.rsiTop = true; methods.forkAndFlush = true; options.methods = methods
        store.setSharedConfiguration(options, for: project.id)
        store.play(project)
        try await fixture.waitUntil { store.sessions.count == 1 && fixture.started(store, count: 1) }
        let session = try XCTUnwrap(store.sessions.first)
        XCTAssertNil(session.sharedAgent, "Method runner owns branching; do not multiply two parallel schedulers")
        let checkout = URL(fileURLWithPath: try XCTUnwrap(session.workingDirectory))
        let input = try String(contentsOf: checkout.appendingPathComponent("fixture-input.txt"), encoding: .utf8)
        XCTAssertTrue(input.contains("Optional research methods are enabled"))
        let control = checkout.appendingPathComponent(".auto-research-methods/control/\(session.id.uuidString)")
        let installed = try XCTUnwrap(FileManager.default.contentsOfDirectory(at: control, includingPropertiesForKeys: nil).first)
        let config = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(contentsOf: installed.appendingPathComponent("config.json"))) as? [String: Any])
        let selected = try XCTUnwrap(config["options"] as? [String: Any])
        XCTAssertEqual(selected["rsiTop"] as? Bool, true)
        XCTAssertEqual(selected["forkAndFlush"] as? Bool, true)
        XCTAssertTrue(FileManager.default.fileExists(atPath: installed.appendingPathComponent("ResearchMethods.py").path))
        try fixture.release(session)
        try await fixture.waitUntil { store.sessions.first?.state == "Needs attention" }
        XCTAssertTrue(store.sessions[0].messages.contains { $0.text.contains("has not prepared") })
    }

    @MainActor func testLiveModelPlansAndAppExecutesBranches() async throws {
        let env = ProcessInfo.processInfo.environment
        guard env["RESEARCH_METHODS_LIVE"] == "1", let settingsPath = env["RESEARCH_METHODS_SETTINGS"] else { throw XCTSkip("Live optional-method canary is opt-in") }
        let fixture = IndependentParallelTests()
        let (store, project, directory) = try await fixture.fixture(count: 2)
        let selected = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: URL(fileURLWithPath: settingsPath)))
        store.settings.codexPath = selected.codexPath; store.settings.codexModel = selected.codexModel
        store.settings.codexSandbox = selected.codexSandbox
        let source = URL(fileURLWithPath: try XCTUnwrap(project.localPath))
        try "0".write(to: source.appendingPathComponent("value.txt"), atomically: true, encoding: .utf8)
        var options = store.sharedConfiguration(project.id)
        var methods = try store.methodConfiguration(project.id)
        methods.rsiTop = true; methods.forkAndFlush = true; methods.branches = 2
        methods.rounds = 1; methods.explorationMinutes = 2; methods.exploitationMinutes = 2
        options.methods = methods; store.setSharedConfiguration(options, for: project.id)
        store.startSharedResearch(project, request: """
        This is a bounded integration canary, not mathematical research. The complete task is to increment the integer in value.txt once per worker turn, at most to 2. Use the enabled research methods. Prepare a self-contained evaluator reading only value.txt, valid iff integer 0...2, score equal to that integer, metrics counter and remaining (=2-counter), and write its evidence only to .method-evaluation. Evaluation timeout 10 seconds; maximize. Pick a 60-second stall window, meaningful gain zero, one metric per view, two portfolio directions. Give workers instructions to read the integer, increment it exactly once if below two, verify it, and return immediately. Do not change original value.txt in the coordinator. Do not run web/MCP or other experiments. Read the installed policy, save plan.json in the specified directory, and return its ready marker so the app can execute. All files stay in this SSD workspace. Do not launch the runner inside a tool.
        """)
        let deadline = Date().addingTimeInterval(600)
        while Date() < deadline {
            if let session = store.sessions.first, ["Completed", "Needs attention", "Stopped"].contains(session.state) { break }
            try await Task.sleep(for: .milliseconds(250))
        }
        if store.hasActiveRuns { store.shutdown() }
        let session = try XCTUnwrap(store.sessions.first)
        try Workspace.write(session, to: directory.appendingPathComponent("LIVE-METHOD-SESSION.json"))
        XCTAssertEqual(session.state, "Completed", session.messages.suffix(3).map(\.text).joined(separator: "\n"))
        XCTAssertTrue(session.messages.contains { $0.text.contains("Research methods completed this batch. Evaluator score: 2") })
        XCTAssertEqual(try String(contentsOf: source.appendingPathComponent("value.txt"), encoding: .utf8), "0")
        print("LIVE_METHOD_EVIDENCE: \(directory.path)")
    }
}
