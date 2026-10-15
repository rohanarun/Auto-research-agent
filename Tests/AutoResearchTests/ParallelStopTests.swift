import XCTest
@testable import ResearchCore
@testable import AutoResearch

private final class StopObservedBackend: ProcessBackend, @unchecked Sendable {
    private let lock = NSLock()
    private var stopped = false
    var isStopped: Bool { lock.lock();defer { lock.unlock() };return stopped }
    func stop() { lock.lock();stopped = true;lock.unlock() }
    func sendInput(_ text: String) {}
    func run(_ command: Command, onLine: @escaping (String,Bool) -> Void) async throws -> ProcessResult {
        while !isStopped { try await Task.sleep(for:.milliseconds(10)) }
        return ProcessResult(code:-15,stdout:"",stderr:"")
    }
}

final class ParallelStopTests: XCTestCase {
    @MainActor func testStopSignalsIndependentCloudRunnersBeforeAwaitingTheirTasks() async throws {
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let directory = try Workspace.directory(root.path,"tmp/parallel-stop-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:directory) }
        var settings = try JSONDecoder().decode(Settings.self,from:Data(contentsOf:root.appendingPathComponent("Resources/Defaults.json")))
        settings.workspaceRoot = directory.path
        let store = AppStore(resourceDirectory:root.appendingPathComponent("Resources"),defaults:settings)
        let project = ResearchProject(repository:"",name:"Fixture",localPath:directory.path,isCustom:true)
        let group = UUID(), ids = [UUID(),UUID()]
        let backends = ids.map { _ in StopObservedBackend() }
        let entered = expectation(description:"Independent cycle awaiting workers")
        let drained = expectation(description:"All owned tasks drained")
        var tasks: [Task<Void,Never>] = []
        for (index,id) in ids.enumerated() {
            var session = ResearchSession(projectID:project.id,title:"Fixture",provider:.codex)
            session.id = id
            session.cloud = CloudSession(sandboxID:"fixture-\(index)",stateID:id.uuidString)
            session.sharedAgent = SharedAgent(groupID:group,index:index+1,generation:1,previousSessionID:nil,contextFile:nil,sharesContext:false,agentCount:2)
            store.sessions.append(session)
            let runner = ProcessRunner(backend:backends[index]);store.runners[id] = runner
            let task = Task { @MainActor in
                _ = try? await runner.run(Command("fixture",[]),directory:directory)
                store.runners.removeValue(forKey:id)
            }
            store.workerTasks[id] = task;tasks.append(task)
        }
        let workerTasks = tasks
        let cycle = SharedContextCycle(options:.init(enabled:false,agentCount:2,intervalMinutes:0,parallelEnabled:true),hooks:.init(
            prepare:{},launch:{ _,_,_ in ids },drain:{ _ in
                for task in workerTasks { await task.value };drained.fulfill()
            },capture:{ _,_,_ in XCTFail("Independent stop must not capture context");return directory },
            merge:{ $0 },retire:{ _ in XCTFail("Stop must not launch replacements") },cancelOperation:{},
            update:{ phase,generation,current,message in
                store.sharedRuns[project.id] = SharedRunStatus(groupID:group,sharesContext:false,generation:generation,phase:phase,sessionIDs:current,message:message)
            },waitForCompletion:{ _ in
                entered.fulfill()
                for task in workerTasks { await task.value }
            }))
        store.sharedCycles[project.id] = cycle;cycle.start()
        await fulfillment(of:[entered],timeout:5)
        store.stopSharedResearch(project.id)
        // This must be synchronous: the cancelled cycle is still awaiting its
        // unstructured workers, exactly as in the production cloud path.
        XCTAssertTrue(backends.allSatisfy(\.isStopped))
        await fulfillment(of:[drained],timeout:5)
        await cycle.waitUntilStopped()
        XCTAssertFalse(cycle.active);XCTAssertTrue(store.runners.isEmpty)
        XCTAssertEqual(store.sharedRuns[project.id]?.phase,.stopped)
        XCTAssertEqual(store.sessions.count,2)
    }
}
