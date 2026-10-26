import CryptoKit
import XCTest
import ResearchCore
@testable import AutoResearch

/// Explicit, opt-in native-app canary. The request supplies the human's check
/// budget and constraints. This test never calls the judge or provisions compute
/// itself; it exercises the same composer/CLI path as an app user.
/// Enable with KERNEL_STARTER_LIVE=1 and supply SETTINGS, WORKSPACE, BOARD,
/// REQUEST and DEADLINE_SECONDS using the KERNEL_STARTER_ environment prefix.
/// PROVIDER is optional; model selection comes unchanged from the settings file.
final class LiveKernelStarterTests: XCTestCase {
    private struct Receipt: Codable {
        var runID: String
        var board: String
        var provider: String
        var model: String
        var executable: String
        var runtime: String
        var evidence: String
        var requestSHA256: String
        var guideSHA256: String
        var deadlineSeconds: Double
        var startedAt: Date
        var finishedAt: Date?
        var sessionID: UUID?
        var providerSessionID: String?
        var workingDirectory: String?
        var state: String
        var timedOut: Bool
        // A completed CLI turn is not proof of an official check's outcome.
        var officialResultVerified = false
    }

    private var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var resources: URL { root.appendingPathComponent("Resources") }
    private func sha256(_ data: Data) -> String { SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined() }

    @MainActor func testLiveBuiltinComposerWithConfiguredCLI() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard environment["KERNEL_STARTER_LIVE"] == "1" else {
            throw XCTSkip("Live kernel starter canary requires explicit opt-in and a user-approved check budget.")
        }
        func required(_ key: String) throws -> String {
            guard let value = environment[key], !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw ResearchError("Missing live canary environment setting: \(key)")
            }
            return value
        }
        // Validate every supplied file/root before creating runtime artifacts.
        // Existing app settings, histories and checkouts are never modified.
        let workspace = try Workspace.validate(required("KERNEL_STARTER_WORKSPACE"))
        let settingsFile = try Workspace.validate(required("KERNEL_STARTER_SETTINGS"))
        let requestFile = try Workspace.validate(required("KERNEL_STARTER_REQUEST"))
        let board = try required("KERNEL_STARTER_BOARD")
        let durationText = try required("KERNEL_STARTER_DEADLINE_SECONDS")
        guard let duration = Double(durationText), duration.isFinite, duration > 0 else {
            throw ResearchError("KERNEL_STARTER_DEADLINE_SECONDS must be a positive finite duration.")
        }
        let providerName = environment["KERNEL_STARTER_PROVIDER"] ?? Provider.codex.rawValue
        guard let provider = Provider(rawValue: providerName) else {
            throw ResearchError("KERNEL_STARTER_PROVIDER must name a supported configured provider.")
        }
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self, from: Data(contentsOf: settingsFile))
        guard FileManager.default.isExecutableFile(atPath: settings.executable(provider)) else {
            throw ResearchError("The selected provider CLI is unavailable at its configured path.")
        }
        let requestData = try Data(contentsOf: requestFile)
        guard let request = String(data: requestData, encoding: .utf8),
              !request.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw ResearchError("The live canary request must contain UTF-8 research instructions and the approved check budget.")
        }
        let definitions = try BuiltinProjects.load(resources: resources)
        guard let definition = definitions.first(where: { $0.id == board }) else {
            throw ResearchError("KERNEL_STARTER_BOARD does not identify a bundled starter.")
        }
        let guide = try definition.instructions(resources: resources)
        let runID = board + "-" + UUID().uuidString.lowercased()
        let runtime = try Workspace.directory(workspace.path, runID)
        let evidence = try Workspace.directory(root.path, "evidence/live-kernel-starters/" + runID)
        settings.workspaceRoot = runtime.path
        let model: String
        switch provider {
        case .codex: model = settings.codexModel
        case .claude: model = settings.claudeModel
        case .opencode: model = settings.opencodeModel
        }
        var receipt = Receipt(runID: runID, board: board, provider: provider.rawValue,
            model: model, executable: settings.executable(provider), runtime: runtime.path,
            evidence: evidence.path, requestSHA256: sha256(requestData),
            guideSHA256: sha256(Data(guide.utf8)), deadlineSeconds: duration,
            startedAt: Date(), state: "Prepared", timedOut: false)
        let receiptFile = evidence.appendingPathComponent("app-canary-receipt.json")
        try Workspace.write(receipt, to: receiptFile)
        // Print paths only, never settings, request text, authentication output
        // or model/tool messages. Full app evidence remains in its SSD workspace.
        print("LIVE_KERNEL_STARTER_EVIDENCE: \(evidence.path)")
        print("LIVE_KERNEL_STARTER_RUNTIME: \(runtime.path)")
        let store = AppStore(resourceDirectory: resources, defaults: settings)
        defer {
            store.shutdown()
            // Preserve partial launches as well as completed runs. There is no
            // cleanup of source, transcripts or official-job evidence.
            if receipt.finishedAt == nil {
                receipt.finishedAt = Date()
                if let partial = store.sessions.first {
                    receipt.sessionID = partial.id
                    receipt.providerSessionID = partial.providerSessionID
                    receipt.workingDirectory = partial.workingDirectory
                    receipt.state = partial.state
                    try? Workspace.write(partial, to: evidence.appendingPathComponent("session.json"))
                } else { receipt.state = "Launch incomplete; no app session" }
                try? Workspace.write(receipt, to: receiptFile)
            }
        }
        store.daytona.enabled = false
        store.provider = provider
        let project = try XCTUnwrap(store.projects.first { $0.builtinID == board })
        store.select(project)
        store.composer = request
        store.send()
        let session = try XCTUnwrap(store.sessions.first { $0.projectID == project.id })
        let task = try XCTUnwrap(store.workerTasks[session.id])
        receipt.sessionID = session.id
        receipt.state = "Running"
        try Workspace.write(receipt, to: receiptFile)
        let deadline = Date().addingTimeInterval(duration)
        while store.runners[session.id] != nil, Date() < deadline {
            try await Task.sleep(for: .milliseconds(250))
        }
        if store.runners[session.id] != nil {
            receipt.timedOut = true
            receipt.state = "Deadline reached; official job state unverified"
            try Workspace.write(receipt, to: receiptFile)
            // Stop only this app-owned local CLI. Any already submitted official
            // check must be reconciled by job ID; it is never retried here.
            store.stop(session.id)
        }
        await task.value
        let terminal = try XCTUnwrap(store.sessions.first { $0.id == session.id })
        receipt.finishedAt = Date()
        receipt.providerSessionID = terminal.providerSessionID
        receipt.workingDirectory = terminal.workingDirectory
        receipt.state = terminal.state
        try Workspace.write(terminal, to: evidence.appendingPathComponent("session.json"))
        try Workspace.write(receipt, to: receiptFile)
        XCTAssertTrue(store.persist(), "The native app could not persist the canary history.")
        XCTAssertEqual(store.sessions.count, 1, "The canary must have one local coordinator.")
        XCTAssertNil(terminal.cloud)
        XCTAssertNil(terminal.sharedAgent)
        XCTAssertTrue(store.sharedRuns.isEmpty)
        XCTAssertFalse(store.hasActiveRuns)
        XCTAssertFalse(receipt.timedOut, "Live starter deadline reached; preserve and inspect the SSD evidence before any retry.")
        XCTAssertEqual(terminal.state, "Completed", "The native agent turn did not complete. Inspect the SSD evidence; no official result is inferred.")
        XCTAssertNotNil(terminal.providerSessionID, "No real provider session was observed.")
        XCTAssertTrue(terminal.messages.contains { $0.role == "user" && $0.text.contains(request.trimmingCharacters(in: .whitespacesAndNewlines)) && $0.text.contains(guide) },
            "The native launch did not preserve both the supplied request and bundled guide.")
        XCTAssertEqual(terminal.workingDirectory, runtime.appendingPathComponent(project.managedWorkspaceComponent).path)
    }
}
