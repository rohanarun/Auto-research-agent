import SwiftUI
import AppKit
import ResearchCore

final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var store: AppStore?
    func applicationShouldTerminate(_ sender: NSApplication) -> NSApplication.TerminateReply {
        if store?.hasActiveRuns == true {
            let alert = NSAlert(); alert.messageText = "Stop research and quit?"
            alert.informativeText = "Owned local and cloud runs will be stopped. Your chats and project files will be preserved."
            alert.addButton(withTitle: "Stop and quit"); alert.addButton(withTitle: "Keep researching")
            if alert.runModal() != .alertFirstButtonReturn { return .terminateCancel }
        }
        store?.shutdown()
        guard store?.hasActiveRuns == true else { return .terminateNow }
        Task { @MainActor in
            while store?.hasActiveRuns == true { try? await Task.sleep(for: .milliseconds(100)) }
            store?.persist()
            sender.reply(toApplicationShouldTerminate: true)
        }
        return .terminateLater
    }
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
}

@main struct AutoResearchApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var delegate
    @StateObject private var store = AppStore()
    var body: some Scene {
        WindowGroup("Auto Research", id: "main") {
            ContentView(store: store)
                .frame(minWidth: 980, minHeight: 680)
                .preferredColorScheme(.light)
                .onAppear { delegate.store = store; NSApp.setActivationPolicy(.regular); NSApp.activate(ignoringOtherApps: true) }
                .task { await store.start() }
        }
        .defaultSize(width: 1280, height: 860)
        .windowStyle(.hiddenTitleBar)
        .commands {
            CommandGroup(replacing: .newItem) {
                Button("New chat") { store.newChat() }.keyboardShortcut("n")
                Button("Add project") { store.showAddProject = true }.keyboardShortcut("o")
            }
            CommandGroup(replacing: .appSettings) {
                Button("Connections & settings…") { store.showSettings = true }.keyboardShortcut(",")
            }
        }
    }
}
