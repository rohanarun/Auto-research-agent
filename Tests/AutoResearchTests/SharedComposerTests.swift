import AppKit
import SwiftUI
import XCTest
import ResearchCore
@testable import AutoResearch

final class SharedComposerTests: XCTestCase {
    @MainActor func setupGroup() throws -> (AppStore, URL, UUID) {
        let (store, directory, _) = try ChatClosingTests().fixture()
        // UI tests must never invoke an installed account or real provider.
        store.settings.codexPath = "/usr/bin/false"
        store.settings.claudePath = "/usr/bin/false"
        store.settings.opencodePath = "/usr/bin/false"
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
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
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
        XCTAssertEqual(store.composer, "")
        XCTAssertEqual(store.responseNavigation.followups.map(\.text), ["A draft for the group"])
        XCTAssertEqual(store.responseNavigation.followups.first?.delivery, .pending)
        XCTAssertEqual(store.responseNavigation.followups.first?.agentIndex, 1)
        XCTAssertNil(store.error); XCTAssertEqual(store.runners.count, 1)
        store.runners.removeAll()
        try await Task.sleep(for: .milliseconds(350))
    }
    @MainActor func testGroupFollowupTargetsSelectedAgentWithoutStartingNewGroup() async throws {
        let (store, directory, group) = try setupGroup()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        let recipient = store.sessions[1].id
        store.composer = "Review the latest result"
        store.settings.codexPath = "/usr/bin/false" // Never start a real provider in this test.
        store.send(to: UUID())
        XCTAssertEqual(store.composer, "Review the latest result"); XCTAssertTrue(store.runners.isEmpty)
        store.error = nil
        store.runners[store.sessions[0].id] = ProcessRunner()
        store.send(to: recipient)
        XCTAssertNotNil(store.runners[recipient]); XCTAssertEqual(store.runners.count, 2)
        XCTAssertEqual(store.responseNavigation.followups.map(\.text), ["Review the latest result"])
        XCTAssertEqual(store.responseNavigation.followups.first?.agentIndex, 2)
        XCTAssertEqual(store.sessions.count, 2); XCTAssertEqual(store.selectedSharedGroupID, group)
        XCTAssertNil(store.selectedSessionID); XCTAssertEqual(store.composer, "")
        if let task = store.workerTasks[recipient] { await task.value }
        store.runners.removeAll()
        try await Task.sleep(for: .milliseconds(350))
    }
    @MainActor func testReplyNavigationPreservesDraftsFocusesComposerAndRendersHistory() async throws {
        _ = NSApplication.shared
        let (store, directory, _) = try setupGroup()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        store.auth = [.codex: "Fixture account"]
        for index in store.sessions.indices {
            store.sessions[index].state = "Completed"
            store.sessions[index].messages[0].text = "Agent \(index + 1) completed reply\n\n" + String(repeating: "Retained result with supporting evidence.\n", count: 50)
            _ = store.responseNavigation.receive(FleetReply(session: store.sessions[index], message: store.sessions[index].messages[0]))
        }
        let first = store.responseEntries[0], second = store.responseEntries[1]
        let host = NSHostingView(rootView: ContentView(store: store))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 980, height: 680), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host
        func editor(in view: NSView) -> ChatTextView? {
            if let value = view as? ChatTextView { return value }
            return view.subviews.compactMap { editor(in: $0) }.first
        }
        store.selectResponse(first.id)
        try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
        let input = try XCTUnwrap(editor(in: host))
        XCTAssertTrue(window.firstResponder === input)
        input.insertText("Draft for agent one", replacementRange: NSRange(location: NSNotFound, length: 0))
        store.selectResponse(second.id)
        try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
        XCTAssertEqual(input.string, "")
        XCTAssertTrue(window.firstResponder === input)
        input.insertText("Draft for agent two", replacementRange: NSRange(location: NSNotFound, length: 0))
        store.previousResponse()
        try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
        XCTAssertEqual(input.string, "Draft for agent one")
        XCTAssertEqual(store.responseScrollTarget, first.id)
        store.nextResponse()
        try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
        XCTAssertEqual(input.string, "Draft for agent two")
        XCTAssertEqual(store.responseScrollTarget, second.id)
        XCTAssertEqual(store.responsePosition, 1)
        XCTAssertTrue(store.workerTasks.isEmpty)
        XCTAssertTrue(store.runners.isEmpty)
        let root = ChatClosingTests().root
        let output = try Workspace.directory(root.path, "evidence/fleet-reply-navigation")
        for size in [NSSize(width: 980, height: 680), NSSize(width: 1280, height: 860)] {
            window.setContentSize(size)
            host.layoutSubtreeIfNeeded()
            store.selectResponse(second.id)
            try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
            let scroll = try XCTUnwrap(input.enclosingScrollView)
            XCTAssertTrue(host.bounds.contains(scroll.convert(scroll.bounds, to: host)))
            let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
                .write(to: output.appendingPathComponent("reply-\(Int(size.width)).png"))
        }
        store.selectSession(store.sessions[1])
        store.selectResponse(second.id)
        try await Task.sleep(for: .milliseconds(150)); host.layoutSubtreeIfNeeded()
        let individualInput = try XCTUnwrap(editor(in: host))
        XCTAssertEqual(individualInput.string, "Draft for agent two")
        XCTAssertTrue(window.firstResponder === individualInput)
        XCTAssertEqual(store.selectedSessionID, second.sessionID)
        let bitmap = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in: host.bounds))
        host.cacheDisplay(in: host.bounds, to: bitmap)
        try XCTUnwrap(bitmap.representation(using: .png, properties: [:]))
            .write(to: output.appendingPathComponent("reply-individual.png"))
    }

}
