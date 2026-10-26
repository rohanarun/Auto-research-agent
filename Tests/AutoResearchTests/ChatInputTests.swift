import AppKit
import XCTest
import SwiftUI
@testable import AutoResearch

final class ChatInputTests: XCTestCase {
    @MainActor func testHostedEditorBindsPastedMultilineTextAndSendsOnReturn() throws {
        _ = NSApplication.shared
        var draft = ""; var sent: [String] = []
        let host = NSHostingView(rootView: ChatInput(text: Binding(get: { draft }, set: { draft = $0 }), isEditable: true, onSend: { sent.append(draft); draft = "" }))
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 94), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = host; host.layoutSubtreeIfNeeded()
        func find(_ view: NSView) -> ChatTextView? {
            if let editor = view as? ChatTextView { return editor }
            return view.subviews.compactMap(find).first
        }
        let editor = try XCTUnwrap(find(host))
        XCTAssertGreaterThan(editor.frame.width, 0)
        editor.insertText("First line\nSecond line", replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertEqual(draft, "First line\nSecond line"); XCTAssertTrue(sent.isEmpty)
        editor.keyDown(with: key())
        XCTAssertEqual(sent, ["First line\nSecond line"]); XCTAssertEqual(draft, "")
    }
    @MainActor func testReplyFocusRequestIsAppliedOnceAndPreservesDraft() throws {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 160), styleMask: [.borderless], backing: .buffered, defer: false)
        let container = NSView(frame: window.contentView!.bounds)
        let editor = ChatTextView(frame: NSRect(x: 0, y: 0, width: 500, height: 80))
        let other = NSTextView(frame: NSRect(x: 0, y: 90, width: 500, height: 50))
        editor.isEditable = true
        editor.string = "Preserved agent-specific draft"
        container.addSubview(editor); container.addSubview(other); window.contentView = container
        window.makeFirstResponder(other)
        editor.focusForReply(nil)
        XCTAssertTrue(window.firstResponder === other)
        let first = UUID()
        editor.focusForReply(first)
        XCTAssertTrue(window.firstResponder === editor)
        XCTAssertEqual(editor.string, "Preserved agent-specific draft")
        XCTAssertEqual(editor.selectedRange().location, (editor.string as NSString).length)
        window.makeFirstResponder(other)
        editor.focusForReply(first)
        XCTAssertTrue(window.firstResponder === other, "Ordinary rerenders must not steal focus back.")
        editor.focusForReply(UUID())
        XCTAssertTrue(window.firstResponder === editor)
        XCTAssertEqual(editor.string, "Preserved agent-specific draft")
    }

    @MainActor func testReplyFocusWaitsForWindowAndDoesNotInterruptMarkedText() {
        _ = NSApplication.shared
        let editor = ChatTextView(frame: NSRect(x: 0, y: 0, width: 400, height: 80))
        editor.isEditable = true
        editor.focusForReply(UUID())
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 400, height: 80), styleMask: [.borderless], backing: .buffered, defer: false)
        window.contentView = editor
        XCTAssertTrue(window.firstResponder === editor)
        editor.setMarkedText("漢", selectedRange: NSRange(location: 1, length: 0), replacementRange: NSRange(location: NSNotFound, length: 0))
        editor.focusForReply(UUID())
        XCTAssertTrue(editor.hasMarkedText())
        XCTAssertEqual(editor.string, "漢")
        editor.unmarkText()
        XCTAssertFalse(editor.hasMarkedText())
        XCTAssertEqual(editor.string, "漢")
    }

    @MainActor func testLeavingReplyContextCancelsDeferredFocus() {
        _ = NSApplication.shared
        let window = NSWindow(contentRect: NSRect(x: 0, y: 0, width: 600, height: 160), styleMask: [.borderless], backing: .buffered, defer: false)
        let container = NSView(frame: window.contentView!.bounds)
        let editor = ChatTextView(frame: NSRect(x: 0, y: 0, width: 500, height: 80))
        let other = NSTextView(frame: NSRect(x: 0, y: 90, width: 500, height: 50))
        editor.isEditable = true
        container.addSubview(editor); container.addSubview(other); window.contentView = container
        editor.setMarkedText("漢", selectedRange: NSRange(location: 1, length: 0), replacementRange: NSRange(location: NSNotFound, length: 0))
        editor.focusForReply(UUID())
        editor.focusForReply(nil)
        window.makeFirstResponder(other)
        editor.unmarkText()
        XCTAssertTrue(window.firstResponder === other)
        XCTAssertEqual(editor.string, "漢")
    }

    @MainActor func key(_ code: UInt16 = 36, modifiers: NSEvent.ModifierFlags = []) -> NSEvent {
        NSEvent.keyEvent(with: .keyDown, location: .zero, modifierFlags: modifiers, timestamp: 0, windowNumber: 0, context: nil, characters: "\r", charactersIgnoringModifiers: "\r", isARepeat: false, keyCode: code)!
    }
    @MainActor func testReturnAndKeypadEnterSendWithoutInsertingNewline() {
        _ = NSApplication.shared
        let view = ChatTextView(); view.string = "A research question"; view.isEditable = true
        var sent = 0; view.onSend = { sent += 1 }
        view.keyDown(with: key()); view.keyDown(with: key(76)); view.keyDown(with: key(modifiers: .command))
        XCTAssertEqual(sent, 3); XCTAssertEqual(view.string, "A research question")
    }
    @MainActor func testShiftReturnInsertsNewlineWithoutSending() {
        _ = NSApplication.shared
        let view = ChatTextView(); view.isEditable = true; view.string = "Line one"; view.setSelectedRange(NSRange(location: 8, length: 0))
        var sent = 0; view.onSend = { sent += 1 }
        view.keyDown(with: key(modifiers: .shift))
        XCTAssertEqual(sent, 0); XCTAssertEqual(view.string, "Line one\n")
    }
    @MainActor func testReturnDoesNotSendWhileComposingAnIMECharacterOrDisabled() {
        _ = NSApplication.shared
        let view = ChatTextView(); view.isEditable = true
        var sent = 0; view.onSend = { sent += 1 }
        view.setMarkedText("漢", selectedRange: NSRange(location: 1, length: 0), replacementRange: NSRange(location: NSNotFound, length: 0))
        XCTAssertTrue(view.hasMarkedText()); view.keyDown(with: key()); XCTAssertEqual(sent, 0)
        view.unmarkText(); view.isEditable = false; view.keyDown(with: key()); XCTAssertEqual(sent, 0)
    }
}
