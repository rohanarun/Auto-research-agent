import XCTest
import ResearchCore
@testable import AutoResearch

final class ComposerConnectionTests: XCTestCase {
    @MainActor func testSavedDirectChatDoesNotAdvertiseNewGatewaySelection() throws {
        let (store, directory, _) = try ChatClosingTests().fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        var session = ResearchSession(projectID: "test/research", title: "Existing direct chat", provider: .codex)
        session.providerSessionID = "fixture-session"
        store.sessions = [session]; store.selectSession(session)
        store.auth[.codex] = "Fixture connected"
        store.gateway = OpenRouterConfiguration(enabled: true, codexModel: "fixture/new-gateway")
        let label = ContentView(store: store).composerConnectionLabel
        XCTAssertTrue(label.contains("Direct CLI"))
        XCTAssertTrue(label.contains("Fixture connected"))
        XCTAssertFalse(label.contains("OpenRouter"))
        XCTAssertFalse(label.contains("fixture/new-gateway"))
    }

    @MainActor func testSavedGatewayChatShowsItsModelAndKeyPresenceOnly() throws {
        let (store, directory, _) = try ChatClosingTests().fixture()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        var session = ResearchSession(projectID: "test/research", title: "Gateway chat", provider: .claude)
        session.connectionRoute = "openrouter:claude"
        session.connectionModel = "fixture/saved-model"
        store.sessions = [session]; store.selectSession(session)
        store.gateway = OpenRouterConfiguration(enabled: false, claudeModel: "fixture/unrelated-model")
        store.gatewayKeyConfigured = true
        var label = ContentView(store: store).composerConnectionLabel
        XCTAssertTrue(label.contains("OpenRouter · fixture/saved-model · Key configured"))
        XCTAssertFalse(label.contains("fixture/unrelated-model"))
        XCTAssertFalse(label.contains("Direct CLI"))
        store.gatewayKeyConfigured = false
        label = ContentView(store: store).composerConnectionLabel
        XCTAssertTrue(label.contains("Key missing"))
        store.sessions[0].connectionModel = nil
        XCTAssertTrue(ContentView(store: store).composerConnectionLabel.contains("Model not recorded"))
    }

    @MainActor func testNewChatUsesConfiguredGatewayAndRecipientUsesItsSavedConnection() throws {
        let (store, directory, group) = try SharedComposerTests().setupGroup()
        defer { store.shutdown(); try? FileManager.default.removeItem(at: directory) }
        store.gateway = OpenRouterConfiguration(enabled: true, codexModel: "fixture/new-model")
        store.gatewayKeyConfigured = false
        store.newChat(); store.provider = .codex
        XCTAssertTrue(ContentView(store: store).composerConnectionLabel.contains("OpenRouter · fixture/new-model · Key missing"))
        store.sessions[1].connectionRoute = "openrouter:codex"
        store.sessions[1].connectionModel = "fixture/agent-two"
        store.selectSharedGroup(group, projectID: "test/research")
        store.selectResponseRecipient(store.sessions[1].id)
        XCTAssertTrue(ContentView(store: store).composerConnectionLabel.contains("OpenRouter · fixture/agent-two"))
        store.selectResponseRecipient(store.sessions[0].id)
        XCTAssertTrue(ContentView(store: store).composerConnectionLabel.contains("Direct CLI"))
    }
}
