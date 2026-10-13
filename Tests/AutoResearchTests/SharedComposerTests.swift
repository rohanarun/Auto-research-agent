import AppKit
import SwiftUI
import XCTest
import ResearchCore
@testable import AutoResearch

final class SharedComposerTests: XCTestCase {
    @MainActor func setupGroup() throws -> (AppStore, URL, UUID) {
        let (store, directory, _) = try ChatClosingTests().fixture()
        let group = UUID()
        for index in 1...2 {
            var session = ResearchSession(projectID: "test/research", title: "Agent \(index)", provider: .codex)
            session.sharedAgent = SharedAgent(groupID: group, index: index, generation: 1, previousSessionID: nil, contextFile: nil)
            session.workingDirectory = try Workspace.directory(directory.path, "agent-\(index)").path
            session.messages = [ChatMessage(role: "assistant", text: String(repeating: "Long research output.\n", count: 50))]
            store.sessions.append(session)
        }
        store.selectSharedGroup(group, projectID: "test/research")
        return (store, directory, group)
    }
    @MainActor func testSharedChatContainsVisibleEditableComposerWhileAgentsRun() async throws {
        _ = NSApplication.shared
        let (store, directory, _) = try setupGroup()
        defer { try? FileManager.default.removeItem(at: directory) }
        store.runners[store.sessions[0].id] = ProcessRunner()
        let host = NSHostingView(rootView: ContentView(store: store))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 980, height: 680), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        func find(_ view: NSView) -> ChatTextView? {
            if let editor = view as? ChatTextView { return editor }
            return view.subviews.compactMap(find).first
        }
        for size in [NSSize(width: 980, height: 680), NSSize(width: 1280, height: 860)] {
            window.setContentSize(size); host.layoutSubtreeIfNeeded()
            let editor = try XCTUnwrap(find(host))
            let scroll = try XCTUnwrap(editor.enclosingScrollView)
            let frame = scroll.convert(scroll.bounds, to: host)
            XCTAssertGreaterThan(frame.width, 100); XCTAssertGreaterThan(frame.height, 40)
            XCTAssertTrue(host.bounds.contains(frame), "Composer is outside the window: \(frame)")
            XCTAssertTrue(editor.isEditable)
        }
        let editor = try XCTUnwrap(find(host))
        editor.insertText("A draft for the group", replacementRange: NSRange(location: NSNotFound, length: 0))
        editor.keyDown(with: ChatInputTests().key())
        XCTAssertEqual(store.composer, "A draft for the group")
        XCTAssertNil(store.error); XCTAssertEqual(store.runners.count, 1)
        store.runners.removeAll()
        try await Task.sleep(for: .milliseconds(350))
    }
    @MainActor func testGroupFollowupTargetsSelectedAgentWithoutStartingNewGroup() async throws {
        let (store, directory, group) = try setupGroup()
        defer { try? FileManager.default.removeItem(at: directory) }
        let recipient = store.sessions[1].id
        store.composer = "Review the latest result"
        store.settings.codexPath = "/usr/bin/false" // Never start a real provider in this test.
        store.send(to: UUID())
        XCTAssertEqual(store.composer, "Review the latest result"); XCTAssertTrue(store.runners.isEmpty)
        store.error = nil
        store.runners[store.sessions[0].id] = ProcessRunner()
        store.send(to: recipient)
        XCTAssertEqual(store.composer, "Review the latest result"); XCTAssertNil(store.runners[recipient])
        store.runners.removeAll(); store.error = nil
        store.send(to: recipient)
        XCTAssertNotNil(store.runners[recipient]); XCTAssertEqual(store.runners.count, 1)
        XCTAssertEqual(store.sessions.count, 2); XCTAssertEqual(store.selectedSharedGroupID, group)
        XCTAssertNil(store.selectedSessionID); XCTAssertEqual(store.composer, "")
        if let task = store.workerTasks[recipient] { await task.value }
        try await Task.sleep(for: .milliseconds(350))
    }
}
