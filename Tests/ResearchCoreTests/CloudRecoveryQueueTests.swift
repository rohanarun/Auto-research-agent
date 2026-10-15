import XCTest
@testable import ResearchCore

final class CloudRecoveryQueueTests: XCTestCase {
    @MainActor func testAdmissionPrecedesRequestCreationAndCoalescesDuplicateWorker() async throws {
        let queue = CloudRecoveryQueue()
        var release: CheckedContinuation<Void,Never>?
        var requests: [String] = []
        let entered = expectation(description:"First recovery admitted")
        let first = Task {
            try await queue.run(id:"first") {
                requests.append("first")
                await withCheckedContinuation { continuation in release = continuation;entered.fulfill() }
            }
        }
        await fulfillment(of:[entered],timeout:5)
        let duplicate = Task { try await queue.run(id:"first") { requests.append("duplicate") } }
        let second = Task { try await queue.run(id:"second") { requests.append("second") } }
        for _ in 0..<100 where queue.pendingCount < 2 { await Task.yield() }
        XCTAssertEqual(queue.pendingCount,2)
        XCTAssertEqual(requests,["first"],"Waiting recoveries must not create HTTP requests")
        release?.resume()
        try await first.value;try await duplicate.value;try await second.value
        XCTAssertEqual(requests,["first","second"])
        XCTAssertEqual(queue.pendingCount,0)
    }

    @MainActor func testFailureReleasesAdmissionForNextWorker() async throws {
        let queue = CloudRecoveryQueue();var recovered = false
        do { try await queue.run(id:"failed") { throw ResearchError("Export failed; originals retained") };XCTFail("Expected failure") }
        catch { XCTAssertTrue(error.localizedDescription.contains("originals retained")) }
        try await queue.run(id:"next") { recovered = true }
        XCTAssertTrue(recovered);XCTAssertEqual(queue.pendingCount,0)
    }

    @MainActor func testCancelledResearchCallerStillCompletesRecovery() async throws {
        let queue = CloudRecoveryQueue();var finished = false
        let task = Task { try await queue.run(id:"stopped-model") { finished = true } }
        task.cancel()
        try await task.value
        XCTAssertTrue(finished);XCTAssertEqual(queue.pendingCount,0)
    }
}
