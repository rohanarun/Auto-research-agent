import XCTest
@testable import ResearchCore

final class BuiltinProjectsTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    private func fixture() throws -> URL {
        let directory = try Workspace.directory(root.path, "tmp/builtin-projects-" + UUID().uuidString)
        addTeardownBlock { try? FileManager.default.removeItem(at: directory) }
        let prompts = directory.appendingPathComponent("StarterPrompts")
        try FileManager.default.createDirectory(at: prompts, withIntermediateDirectories: true)
        try "Use your own official judge account.".write(to: prompts.appendingPathComponent("OfficialJudge.md"), atomically: true, encoding: .utf8)
        try "Study the public reference and retain failed experiments.".write(to: prompts.appendingPathComponent("example.md"), atomically: true, encoding: .utf8)
        try writeCatalog([definition()], in: directory)
        return directory
    }

    private func definition(id: String = "example", promptFile: String = "StarterPrompts/example.md", website: String = "https://example.org/problems/example") -> [String: Any] {
        ["id": id, "name": "Example kernel", "summary": "A public starter project", "category": "Kernel research",
         "website": website, "leaderboard": "https://example.org/leaderboard/example", "promptFile": promptFile,
         "suggestedPrompts": ["Read the task and explain its scoring rules."]]
    }

    private func writeCatalog(_ definitions: [[String: Any]], in directory: URL) throws {
        try JSONSerialization.data(withJSONObject: definitions).write(to: directory.appendingPathComponent("BuiltinProjects.json"))
    }

    func testBundledKernelProjectsHaveDistinctStableIdentitiesAndInstructions() throws {
        let resources = root.appendingPathComponent("Resources")
        let definitions = try BuiltinProjects.load(resources: resources)
        let expected = Set(["muse_bf16_proj", "muse_moe_w4a16", "muse_attention_decode", "muse_decode_layer", "muse_prefill_layer"])
        XCTAssertEqual(Set(definitions.map(\.id)), expected)
        XCTAssertEqual(Set(definitions.map { $0.project.id }).count, definitions.count)
        for definition in definitions {
            let project = definition.project
            XCTAssertEqual(project.id, "builtin:" + definition.id)
            XCTAssertEqual(project.builtinID, definition.id)
            XCTAssertEqual(project.managedWorkspaceComponent, "projects/builtins/" + definition.id)
            XCTAssertTrue(project.repository.isEmpty)
            XCTAssertNil(project.githubURL)
            XCTAssertFalse(try definition.instructions(resources: resources).isEmpty)
            XCTAssertFalse(definition.suggestedPrompts.isEmpty)
        }
    }

    func testLegacyProjectDecodesAndBuiltinRoundTrips() throws {
        let legacy = #"{"id":"owner/repo","name":"Legacy","repository":"owner/repo","summary":"","category":"Research","isCustom":false}"#
        let project = try JSONDecoder().decode(ResearchProject.self, from: Data(legacy.utf8))
        XCTAssertNil(project.builtinID)
        XCTAssertEqual(project.id, "owner/repo")
        XCTAssertEqual(project.managedWorkspaceComponent, "projects/owner/repo")
        let resources = try fixture()
        let builtin = try XCTUnwrap(BuiltinProjects.load(resources: resources).first).project
        let restored = try JSONDecoder().decode(ResearchProject.self, from: JSONEncoder().encode(builtin))
        XCTAssertEqual(restored, builtin)
    }

    func testMergeAddsBuiltinsToExistingInstallAndPreservesHistoryWithoutDuplicates() throws {
        let resources = try fixture()
        let builtins = try BuiltinProjects.load(resources: resources)
        let fetched = ResearchProject(repository: "owner/updated", name: "Fresh name")
        let stale = ResearchProject(repository: "owner/updated", name: "Old name")
        let historical = ResearchProject(repository: "owner/removed")
        let custom = ResearchProject(repository: "owner/custom", isCustom: true)
        let unused = ResearchProject(repository: "owner/unused")
        let sessionIDs: Set<String> = [historical.id, builtins[0].project.id]
        let merged = BuiltinProjects.merge(fetched: [fetched, fetched], existing: [stale, historical, custom, unused, builtins[0].project], builtins: builtins, sessionProjectIDs: sessionIDs)
        XCTAssertEqual(merged.map(\.id), [builtins[0].project.id, fetched.id, historical.id, custom.id])
        XCTAssertEqual(merged.first(where: { $0.id == fetched.id })?.name, "Fresh name")
        let offline = BuiltinProjects.merge(fetched: merged, existing: merged, builtins: builtins, sessionProjectIDs: sessionIDs)
        XCTAssertEqual(offline, merged)
    }

    func testPrepareSeparatesProjectsPreservesEditsAndRequiresNoRepository() throws {
        let resources = try fixture()
        try writeCatalog([definition(), definition(id: "another")], in: resources)
        let definitions = try BuiltinProjects.load(resources: resources)
        let workspace = try Workspace.directory(resources.path, "workspace")
        let first = try definitions[0].prepare(workspaceRoot: workspace.path, resources: resources)
        let second = try definitions[1].prepare(workspaceRoot: workspace.path, resources: resources)
        XCTAssertNotEqual(first, second)
        let guide = first.appendingPathComponent("auto-research.md")
        let initial = try String(contentsOf: guide, encoding: .utf8)
        XCTAssertTrue(initial.hasPrefix("Use your own official judge account."))
        XCTAssertTrue(initial.contains("Study the public reference"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: first.appendingPathComponent(".git").path))
        try "My edited research objective.".write(to: guide, atomically: true, encoding: .utf8)
        let receipt = first.appendingPathComponent("saved-result.json")
        try "Preserved receipt".write(to: receipt, atomically: true, encoding: .utf8)
        XCTAssertEqual(try definitions[0].prepare(workspaceRoot: workspace.path, resources: resources), first)
        XCTAssertEqual(try String(contentsOf: guide, encoding: .utf8), "My edited research objective.")
        XCTAssertEqual(try String(contentsOf: receipt, encoding: .utf8), "Preserved receipt")
    }

    func testCatalogRejectsInvalidIDsLinksDuplicatesAndEmptyPrompts() throws {
        let resources = try fixture()
        for invalid in [definition(id: "../escape"), definition(id: "example\n"), definition(website: "http://example.org"), definition(website: "https://user:secret@example.org")] {
            try writeCatalog([invalid], in: resources)
            XCTAssertThrowsError(try BuiltinProjects.load(resources: resources))
        }
        try writeCatalog([definition(), definition()], in: resources)
        XCTAssertThrowsError(try BuiltinProjects.load(resources: resources))
        var missingSuggestions = definition()
        missingSuggestions["suggestedPrompts"] = [" \n"]
        try writeCatalog([missingSuggestions], in: resources)
        XCTAssertThrowsError(try BuiltinProjects.load(resources: resources))
        try writeCatalog([definition()], in: resources)
        try " \n".write(to: resources.appendingPathComponent("StarterPrompts/example.md"), atomically: true, encoding: .utf8)
        XCTAssertThrowsError(try BuiltinProjects.load(resources: resources))
    }

    func testResourcesRejectTraversalMissingFileAndSymlinkEscapeBeforeCreatingWorkspace() throws {
        let resources = try fixture()
        let definition = try XCTUnwrap(BuiltinProjects.load(resources: resources).first)
        for path in ["../example.md", "/etc/hosts", "StarterPrompts/../example.md", "StarterPrompts/missing.md"] {
            try writeCatalog([self.definition(promptFile: path)], in: resources)
            XCTAssertThrowsError(try BuiltinProjects.load(resources: resources))
        }
        let prompt = resources.appendingPathComponent("StarterPrompts/example.md")
        try FileManager.default.removeItem(at: prompt)
        try FileManager.default.createSymbolicLink(at: prompt, withDestinationURL: URL(fileURLWithPath: "/etc/hosts"))
        try writeCatalog([self.definition()], in: resources)
        XCTAssertThrowsError(try BuiltinProjects.load(resources: resources))
        let workspace = resources.appendingPathComponent("not-created")
        XCTAssertThrowsError(try definition.prepare(workspaceRoot: workspace.path, resources: resources))
        XCTAssertFalse(FileManager.default.fileExists(atPath: workspace.path))
    }

    func testPrepareRejectsWorkspaceAndInstructionSymlinks() throws {
        let resources = try fixture()
        let definition = try XCTUnwrap(BuiltinProjects.load(resources: resources).first)
        let workspace = try Workspace.directory(resources.path, "workspace")
        let untouched = try Workspace.directory(resources.path, "untouched")
        let projects = workspace.appendingPathComponent("projects")
        try FileManager.default.createSymbolicLink(at: projects, withDestinationURL: untouched)
        XCTAssertThrowsError(try definition.prepare(workspaceRoot: workspace.path, resources: resources))
        XCTAssertTrue(try FileManager.default.contentsOfDirectory(atPath: untouched.path).isEmpty)
        try FileManager.default.removeItem(at: projects)
        let directory = try definition.prepare(workspaceRoot: workspace.path, resources: resources)
        let guide = directory.appendingPathComponent("auto-research.md")
        try FileManager.default.removeItem(at: guide)
        let target = untouched.appendingPathComponent("guide.md")
        try "Do not change".write(to: target, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: guide, withDestinationURL: target)
        XCTAssertThrowsError(try definition.prepare(workspaceRoot: workspace.path, resources: resources))
        XCTAssertEqual(try String(contentsOf: target, encoding: .utf8), "Do not change")
    }
}
