import Foundation
import XCTest
import ResearchCore
@testable import AutoResearch

final class BuiltinSetupTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    @MainActor func testAccountRefreshRunsOnlyWhoamiAndReturnsUsername() async throws {
        let resources = root.appendingPathComponent("Resources")
        let directory = try Workspace.directory(root.path, "tmp/builtin-account-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,
            from: Data(contentsOf: resources.appendingPathComponent("Defaults.example.json")))
        settings.workspaceRoot = try Workspace.directory(directory.path, "runtime").path
        settings.codexPath = directory.appendingPathComponent("no-agent-should-run").path
        let okc = directory.appendingPathComponent("okc")
        try "#!/bin/sh\n[ \"$#\" -eq 1 ] && [ \"$1\" = whoami ] || exit 92\nprintf '%s\\n' fixture-user\n".write(to: okc, atomically: true, encoding: .utf8)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: okc.path)
        let store = AppStore(resourceDirectory: resources, defaults: settings)
        defer { store.shutdown() }
        let account = try await store.refreshOKCAccount()
        XCTAssertEqual(account, "fixture-user")
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertTrue(store.builtinDrafts.isEmpty)
    }

    @MainActor func testPreparingLoginAndScoredPromptDoesNotLaunchOrSpendQuota() throws {
        let resources = root.appendingPathComponent("Resources")
        let directory = try Workspace.directory(root.path, "tmp/builtin-setup-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: directory) }
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,
            from: Data(contentsOf: resources.appendingPathComponent("Defaults.example.json")))
        settings.workspaceRoot = try Workspace.directory(directory.path, "runtime").path
        settings.codexPath = "/no-coding-agent-should-run"
        let store = AppStore(resourceDirectory: resources, defaults: settings)
        defer { store.shutdown() }
        let project = try XCTUnwrap(store.builtinProjects.first)
        let input = try KernelResearchSetup(sourcePath: "kernel/submission.py", checkBudget: "1",
            allowScoredSubmissions: true, scoreBudget: "1", researchPrompt: "Check and score if qualified.",
            scoreCriterion: "My first clean baseline, only if no scored entry exists.", expectedAccount: "fixture-user")
        let prompt = try project.setupPrompt(input, resources: resources)
        XCTAssertTrue(prompt.contains("okc submit <projectID> <file> --json"))
        let first = try store.prepareOKCLogin(project.project)
        let second = try store.prepareOKCLogin(project.project)
        let choice = try store.prepareOKCLogin(project.project, chooseAccount: true)
        XCTAssertTrue(try String(contentsOf: choice, encoding: .utf8).contains("login --force"))
        XCTAssertNotEqual(first, second)
        XCTAssertEqual(try Workspace.validate(first.path), first)
        XCTAssertEqual(try String(contentsOf: first, encoding: .utf8), try String(contentsOf: second, encoding: .utf8))
        let permissions = try FileManager.default.attributesOfItem(atPath: first.path)[.posixPermissions] as? NSNumber
        XCTAssertEqual(permissions?.intValue, 0o700)
        XCTAssertTrue(store.sessions.isEmpty)
        XCTAssertTrue(store.workerTasks.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        XCTAssertFalse(store.hasActiveRuns)
        XCTAssertTrue(store.composer.isEmpty)
    }
}
