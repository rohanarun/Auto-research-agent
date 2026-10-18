import XCTest
@testable import ResearchCore
final class MethodTranscriptTests: XCTestCase {
    func line(_ event: [String: Any]) throws -> String {
        String(decoding: try JSONSerialization.data(withJSONObject: ["type":"research_method_event", "branch":"round-1-explore-epoch-2-branch-3", "provider":"codex", "key":"offset-42", "event":event]), as: UTF8.self)
    }
    func testCompletionAndFailureAreMessagesNotCoordinatorLifecycle() throws {
        XCTAssertEqual(MethodTranscript.decode(try line(["type":"turn.completed"]))?.first?.role, "status")
        XCTAssertEqual(MethodTranscript.decode(try line(["type":"turn.failed","error":["message":"failed branch"]]))?.first?.role,"error")
    }
    func testRepeatedReplayHasStableKeysAndWorkerLabel() throws {
        let input = try line(["type":"item.completed","item":["type":"agent_message","text":"evidence"]])
        let first = try XCTUnwrap(MethodTranscript.decode(input)?.first)
        XCTAssertEqual(first.key,MethodTranscript.decode(input)?.first?.key)
        XCTAssertTrue(first.speaker.contains("Worker 3")); XCTAssertTrue(first.speaker.contains("Checkpoint 2"))
    }
    func testOldSavedMessageRemainsReadable() throws {
        var old = try JSONSerialization.jsonObject(with:JSONEncoder().encode(ChatMessage(role:"assistant",text:"retained"))) as! [String:Any]
        old.removeValue(forKey:"sourceKey");old.removeValue(forKey:"speaker")
        let decoded = try JSONDecoder().decode(ChatMessage.self,from:JSONSerialization.data(withJSONObject:old))
        XCTAssertEqual(decoded.text,"retained");XCTAssertNil(decoded.sourceKey)
    }
}
