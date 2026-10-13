import AppKit
import SwiftUI

/// Handle Return before NSTextView inserts a paragraph. Marked text belongs to
/// the input method, so committing a composed character must never send a chat.
final class ChatTextView: NSTextView {
    var onSend: () -> Void = {}
    override func keyDown(with event: NSEvent) {
        let modifiers = event.modifierFlags.intersection(.deviceIndependentFlagsMask)
        if isEditable, !hasMarkedText(), [36, 76].contains(event.keyCode),
           modifiers.intersection([.shift, .option, .control]).isEmpty {
            onSend()
            return
        }
        super.keyDown(with: event)
    }
}

struct ChatInput: NSViewRepresentable {
    @Binding var text: String
    var isEditable: Bool
    var onSend: () -> Void

    func makeCoordinator() -> Coordinator { Coordinator(self) }
    func makeNSView(context: Context) -> NSScrollView {
        let scroll = NSScrollView()
        scroll.drawsBackground = false; scroll.hasVerticalScroller = true
        let view = ChatTextView()
        view.isRichText = false; view.drawsBackground = false
        view.font = .systemFont(ofSize: 14); view.textColor = .labelColor
        view.textContainerInset = NSSize(width: 5, height: 8)
        view.isHorizontallyResizable = false; view.isVerticallyResizable = true
        view.autoresizingMask = [.width]
        view.textContainer?.widthTracksTextView = true
        view.textContainer?.containerSize = NSSize(width: 0, height: CGFloat.greatestFiniteMagnitude)
        view.isAutomaticQuoteSubstitutionEnabled = false
        view.isAutomaticDashSubstitutionEnabled = false
        view.setAccessibilityLabel("Message the research agent")
        view.delegate = context.coordinator
        scroll.documentView = view
        return scroll
    }
    func updateNSView(_ scroll: NSScrollView, context: Context) {
        context.coordinator.parent = self
        guard let view = scroll.documentView as? ChatTextView else { return }
        view.onSend = onSend; view.isEditable = isEditable
        if view.string != text {
            let selection = view.selectedRange()
            view.string = text
            view.setSelectedRange(NSRange(location: min(selection.location, (text as NSString).length), length: 0))
        }
    }
    final class Coordinator: NSObject, NSTextViewDelegate {
        var parent: ChatInput
        init(_ parent: ChatInput) { self.parent = parent }
        func textDidChange(_ notification: Notification) {
            guard let view = notification.object as? NSTextView else { return }
            parent.text = view.string
        }
    }
}
