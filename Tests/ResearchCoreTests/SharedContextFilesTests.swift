import XCTest
@testable import ResearchCore

final class SharedContextFilesTests: XCTestCase {
    func testCloudPathMappingDoesNotRewriteCheckpointContents() throws {
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let folder = try Workspace.directory(root.path,"tmp/cloud-literal-" + UUID().uuidString + "/round-1")
        defer { try? FileManager.default.removeItem(at:folder.deletingLastPathComponent()) }
        let file = folder.appendingPathComponent("shared-context.md")
        let body = "Checkpoint includes a literal native reference: " + folder.deletingLastPathComponent().path + "/round-0/finding.md\nUnicode evidence: αβ\n"
        try body.write(to:file,atomically:true,encoding:.utf8)
        let injection = try SharedContextFiles.cloudInjection(file:file,remoteRoot:"/workspace/checkout/SHARED_CONTEXT")
        XCTAssertTrue(injection.hasSuffix(body));XCTAssertTrue(injection.contains("identical saved files"))
        XCTAssertFalse(injection.hasSuffix(body.replacingOccurrences(of:folder.deletingLastPathComponent().path,with:"/workspace/checkout/SHARED_CONTEXT")))
    }
    var root: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func testCompactCheckpointsAreSnapshottedAndIncludedDirectly() throws {
        let directory = try Workspace.directory(root.path, "tmp/checkpoint-context-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let round = try Workspace.directory(directory.path, "round-1")
        var agents: [SharedContextManifest.Agent] = []
        for number in 1...8 {
            let checkout = try Workspace.directory(directory.path, "agent-\(number)")
            let live = checkout.appendingPathComponent("RESEARCH_CHECKPOINT.md")
            let claim = "Agent \(number): verified improvement; next check pending; evidence in result.json."
            try claim.write(to: live, atomically: true, encoding: .utf8)
            let checkpoint = try XCTUnwrap(SharedContextFiles.snapshotCheckpoint(checkout: checkout, round: round, agent: number))
            try "Later work must not alter a saved round.".write(to: live, atomically: true, encoding: .utf8)
            XCTAssertEqual(try String(contentsOfFile: checkpoint, encoding: .utf8), claim)
            let transcript = round.appendingPathComponent("agent-\(number).json")
            try Workspace.write(["detail": "Transcript must remain out of prompt"], to: transcript)
            agents.append(.init(number: number, sessionID: UUID(), transcript: transcript.path, checkout: checkout.path, checkpoint: checkpoint))
        }
        let manifest = round.appendingPathComponent("manifest.json")
        try Workspace.write(SharedContextManifest(generation: 1, project: "fixture", previousContext: nil, agents: agents), to: manifest)
        let start = Date()
        let output = try SharedContextFiles.publish(manifest: manifest, policy: "Resume from checkpoints.")
        let text = try String(contentsOf: output, encoding: .utf8)
        for number in 1...8 { XCTAssertTrue(text.contains("Agent \(number): verified improvement")) }
        XCTAssertFalse(text.contains("Later work must not alter"))
        XCTAssertFalse(text.contains("Transcript must remain out of prompt"))
        print("Published eight checkpoints directly in \(Date().timeIntervalSince(start)) seconds.")
        let checkout = try Workspace.directory(directory.path, "missing")
        XCTAssertNil(try SharedContextFiles.snapshotCheckpoint(checkout: checkout, round: round, agent: 9))
        try FileManager.default.createSymbolicLink(at: checkout.appendingPathComponent("RESEARCH_CHECKPOINT.md"), withDestinationURL: output)
        XCTAssertThrowsError(try SharedContextFiles.snapshotCheckpoint(checkout: checkout, round: round, agent: 9))
    }

    func testIndexKeepsEverySourceWithoutEmbeddingTranscriptsOrPriorHistory() throws {
        let directory = try Workspace.directory(root.path, "tmp/context-index-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let previous = directory.appendingPathComponent("previous.md")
        try String(repeating: "old context\n", count: 10_000).write(to: previous, atomically: true, encoding: .utf8)
        var agents: [SharedContextManifest.Agent] = []
        for number in 1...4 {
            let transcript = directory.appendingPathComponent("agent-\(number).json")
            try Workspace.write(["message": String(repeating: "large transcript", count: 100_000)], to: transcript)
            let notes = directory.appendingPathComponent("notes-\(number).md")
            if number != 4 { try "Evidence marker \(number)".write(to: notes, atomically: true, encoding: .utf8) }
            agents.append(.init(number: number, sessionID: UUID(), transcript: transcript.path, checkout: directory.path, notes: number == 4 ? nil : notes.path))
        }
        let manifest = directory.appendingPathComponent("manifest.json")
        try Workspace.write(SharedContextManifest(generation: 2, project: "fixture", previousContext: previous.path, agents: agents), to: manifest)
        let started = Date()
        let context = try SharedContextFiles.publish(manifest: manifest, policy: "Use saved notes; verify relevant evidence.")
        let text = try String(contentsOf: context, encoding: .utf8)
        for agent in agents {
            XCTAssertTrue(text.contains(agent.transcript))
            if let notes = agent.notes { XCTAssertTrue(text.contains(notes)) }
        }
        XCTAssertTrue(text.contains(previous.path))
        XCTAssertTrue(text.contains("No notes saved"))
        XCTAssertFalse(text.contains("large transcript"))
        XCTAssertFalse(text.contains("old context"))
        XCTAssertLessThan(text.utf8.count, 10_000)
        let attributes = try FileManager.default.attributesOfItem(atPath: context.path)
        XCTAssertEqual((attributes[.posixPermissions] as? NSNumber)?.intValue, 0o600)
        print("Published four-agent context index in \(Date().timeIntervalSince(started)) seconds; \(text.utf8.count) bytes.")
        try FileManager.default.removeItem(atPath: agents[2].transcript)
        XCTAssertThrowsError(try SharedContextFiles.publish(manifest: manifest, policy: "retry"))
        XCTAssertEqual(try String(contentsOf: context, encoding: .utf8), text, "Failed publication must preserve the existing index")
    }

    func testDescriptiveNamesArePreservedAndSnapshotsStayIndependent() throws {
        let directory = try Workspace.directory(root.path, "tmp/topic-files-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let checkout = try Workspace.directory(directory.path, "checkout")
        let topics = try Workspace.directory(checkout.path, "RESEARCH_CONTEXT")
        let round = try Workspace.directory(directory.path, "round-1")
        let name = "negative-result__h23-frame-retiming__no-rank-mass-improvement-under-fixed-role-budget__exact-replay-passed.md"
        let note = topics.appendingPathComponent(name)
        try "Verified evidence, preserved before replacement.".write(to: note, atomically: true, encoding: .utf8)
        let first = try SharedContextFiles.snapshotDocuments(checkout: checkout, round: round, agent: 1)
        let second = try SharedContextFiles.snapshotDocuments(checkout: checkout, round: round, agent: 2)
        XCTAssertEqual(first.count, 1)
        XCTAssertEqual(URL(fileURLWithPath: first[0]).lastPathComponent, name)
        XCTAssertNotEqual(first, second)
        try "Later edits must not change saved evidence.".write(to: note, atomically: true, encoding: .utf8)
        XCTAssertEqual(try String(contentsOfFile: first[0], encoding: .utf8), "Verified evidence, preserved before replacement.")
        let transcript = round.appendingPathComponent("agent-1.json")
        try Workspace.write(["message": "details"], to: transcript)
        let manifest = round.appendingPathComponent("manifest.json")
        try Workspace.write(SharedContextManifest(generation: 1, project: "fixture", previousContext: nil, agents: [.init(number: 1, sessionID: UUID(), transcript: transcript.path, checkout: checkout.path, contextFiles: first)]), to: manifest)
        let index = try SharedContextFiles.publish(manifest: manifest, policy: "Select relevant topic filenames.")
        let content = try String(contentsOf: index, encoding: .utf8)
        XCTAssertTrue(content.contains(name))
        XCTAssertFalse(content.contains("Verified evidence"))
        let outside = directory.appendingPathComponent("outside.md")
        try "Outside checkout.".write(to: outside, atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(at: topics.appendingPathComponent("external.md"), withDestinationURL: outside)
        XCTAssertThrowsError(try SharedContextFiles.snapshotDocuments(checkout: checkout, round: round, agent: 3))
    }

    func testEmptySnapshotCannotPublishAndOldManifestWithoutNotesStillLoads() throws {
        let directory = try Workspace.directory(root.path, "tmp/context-index-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: directory) }
        let manifest = directory.appendingPathComponent("manifest.json")
        try Workspace.write(SharedContextManifest(generation: 1, project: "fixture", previousContext: nil, agents: []), to: manifest)
        XCTAssertThrowsError(try SharedContextFiles.publish(manifest: manifest, policy: "read"))
        let raw = """
        {"generation":1,"project":"fixture","agents":[{"number":1,"sessionID":"\(UUID().uuidString)","transcript":"saved.json","checkout":"checkout"}]}
        """
        let legacy = try JSONDecoder().decode(SharedContextManifest.self, from: Data(raw.utf8))
        XCTAssertNil(legacy.agents[0].notes)
        XCTAssertNil(legacy.agents[0].checkpoint)
    }
}
