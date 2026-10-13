import XCTest
@testable import ResearchCore

final class ResearchDashboardTests: XCTestCase {
    private func fixture() throws -> ResearchDashboard {
        let history = #"""
        {"as_of":"2026-10-13T17:00:00Z","refresh_after_seconds":5,"source_error":"One source is delayed",
         "scopes":{"alpha":{"id":"alpha","name":"Alpha","repo":"Example/Alpha"},"other":{"id":"other","name":"Other","repo":"Example/Other"}},
         "groups":{
          "alpha:0":{"id":"alpha:0","track":"alpha","name":"First metric","direction":"down",
            "reconstructed":{"value":28,"timeline":[{"from":10,"to":20,"value":189,"sources":["https://github.com/Example/Alpha/pull/2"],"leaders":["author"]},{"from":20,"to":30,"value":28,"sources":["https://github.com/Example/Alpha/pull/3"],"leaders":["author"]}]},
            "observed":{"value":null,"timeline":[]}},
          "alpha:1":{"id":"alpha:1","track":"alpha","name":"Different metric","direction":"up","reconstructed":{"value":0,"timeline":[{"from":10,"to":20,"value":null,"sources":[],"leaders":[]},{"from":30,"to":20,"value":5,"sources":[],"leaders":[]},{"from":20,"to":30,"value":0,"sources":[],"leaders":[]}]},"observed":{"value":null,"timeline":[]}},
          "other:0":{"id":"other:0","track":"other","name":"Unrelated metric","direction":"up","reconstructed":{"value":999,"timeline":[]},"observed":{"value":999,"timeline":[]}}
         }}
        """#
        let research = #"""
        {"tracks":{"alpha":{"checked_at":"2026-10-13T17:00:00Z","pulls":[
         {"number":2,"title":"Merged result","html_url":"https://github.com/Example/Alpha/pull/2","state":"closed","merged_at":"2026-10-12T00:00:00Z","created_at":"2026-10-11T00:00:00Z","updated_at":"2026-10-12T00:00:00Z","author":"author"},
         {"number":3,"title":"Open claim","html_url":"https://github.com/Example/Alpha/pull/3","state":"open","merged_at":null,"created_at":"2026-10-12T00:00:00Z","updated_at":"2026-10-13T00:00:00Z","author":"author"}]}}}
        """#
        return try ResearchDashboard.decode(history: Data(history.utf8), research: Data(research.utf8))
    }
    func testRepositoryScopingKeepsDifferentMetricsSeparate() throws {
        let data = try fixture()
        XCTAssertEqual(data.groups(for: "example/ALPHA").map(\.id), ["alpha:0", "alpha:1"])
        XCTAssertTrue(data.groups(for: "example/missing").isEmpty)
        XCTAssertNil(data.track(for: "example/missing"))
        XCTAssertEqual(data.history.source_error, "One source is delayed")
    }
    func testHistoryDoesNotInventValuesOrTreatOpenClaimsAsMerged() throws {
        let data = try fixture(); let groups = data.groups(for: "Example/Alpha")
        let track = try XCTUnwrap(data.track(for: "Example/Alpha"))
        XCTAssertEqual(track.status(for: groups[0].reconstructed.validIntervals.first), "Merged PR")
        XCTAssertEqual(track.status(for: groups[0].reconstructed.validIntervals.last), "Open proposal")
        XCTAssertTrue(groups[0].observed.validIntervals.isEmpty)
        XCTAssertNil(groups[0].observed.value)
        XCTAssertEqual(groups[1].reconstructed.validIntervals.count, 1)
        XCTAssertEqual(groups[1].reconstructed.validIntervals.first?.value, 0)
        XCTAssertEqual(track.recentPulls.first?.number, 3)
    }
    func testLiveBackendChartsWhenExplicitlyEnabled() async throws {
        guard ProcessInfo.processInfo.environment["RESEARCH_LIVE_DASHBOARD_TEST"] == "1" else { throw XCTSkip("Live backend check is opt-in; no CLI calls are made.") }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let settings = try JSONDecoder().decode(Settings.self, from: Data(contentsOf: root.appendingPathComponent("Resources/Defaults.json")))
        let data = try await ResearchDashboard.fetch(catalogURL: settings.catalogURL)
        XCTAssertFalse(data.history.groups.isEmpty)
        for scope in data.history.scopes.values where scope.repo != nil {
            let groups = data.groups(for: scope.repo!)
            XCTAssertFalse(groups.isEmpty, scope.id)
            XCTAssertTrue(groups.allSatisfy { $0.track == scope.id })
            XCTAssertNotNil(data.track(for: scope.repo!), scope.id)
            print("Backend verified:", scope.repo!, groups.count, "metrics;", groups.reduce(0) { $0 + $1.reconstructed.validIntervals.count }, "intervals")
        }
    }
}
