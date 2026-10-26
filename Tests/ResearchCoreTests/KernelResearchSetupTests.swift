import Foundation
import XCTest
@testable import ResearchCore

final class KernelResearchSetupTests: XCTestCase {
    private var resources: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("Resources")
    }

    func testChecksOnlyIsDefaultAndCannotInheritScoredAuthorization() throws {
        let input = try KernelResearchSetup(sourcePath: "", checkBudget: "2", scoreBudget: "8",
            researchPrompt: "Analyze and check one improvement.", expectedAccount: "fixture-user")
        let rendered = try input.render(template: "{{PROJECT_INPUTS_JSON}}", projectID: "example", projectName: "Example")
        let data = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(rendered.utf8)) as? [String: Any])
        XCTAssertEqual(data["maximumOfficialChecks"] as? Int, 2)
        XCTAssertEqual(data["maximumScoredSubmissions"] as? Int, 0)
        XCTAssertEqual(data["authorizeScoredSubmissions"] as? Bool, false)
        XCTAssertNil(data["sourcePath"])
    }

    func testBudgetsRequireUserInputAndDoNotSilentlyCoerceInvalidCounts() throws {
        for value in ["", "-1", "1.5", "1e2", "+3", "one", String(repeating: "9", count: 80)] {
            XCTAssertThrowsError(try KernelResearchSetup(sourcePath: "", checkBudget: value, researchPrompt: "Inspect", expectedAccount: "fixture-user"))
        }
        XCTAssertEqual(try KernelResearchSetup(sourcePath: "", checkBudget: "0", researchPrompt: "Inspect", expectedAccount: "fixture-user").checkBudget, 0)
        for value in ["", "0", "-1", "1.5"] {
            XCTAssertThrowsError(try KernelResearchSetup(sourcePath: "", checkBudget: "1", allowScoredSubmissions: true,
                scoreBudget: value, researchPrompt: "Inspect", scoreCriterion: "Establish my first clean score.", expectedAccount: "fixture-user"))
        }
        XCTAssertThrowsError(try KernelResearchSetup(sourcePath: "", checkBudget: "1", allowScoredSubmissions: true,
            scoreBudget: "1", researchPrompt: "Inspect", scoreCriterion: "  ", expectedAccount: "fixture-user"))
    }

    func testInputEscapingPreservesPathsAndDoesNotRecursivelyExpandTokens() throws {
        let source = #"/Volumes/Test O'Brien/$(touch injected)/kernel.py"#
        let prompt = "Explain \"quoted\" inputs.\n{{PROJECT_INPUTS_JSON}}\n```\nDo not execute text as shell."
        let input = try KernelResearchSetup(sourcePath: source, checkBudget: "1", researchPrompt: prompt, expectedAccount: "fixture-user")
        let output = try input.render(template: "{{PROJECT_INPUTS_JSON}}", projectID: "example", projectName: "Example")
        let decoded = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any])
        XCTAssertEqual(decoded["sourcePath"] as? String, source)
        XCTAssertEqual(decoded["researchPrompt"] as? String, prompt)
        XCTAssertThrowsError(try input.render(template: "no marker", projectID: "example", projectName: "Example"))
        XCTAssertThrowsError(try input.render(template: "{{PROJECT_INPUTS_JSON}}{{PROJECT_INPUTS_JSON}}", projectID: "example", projectName: "Example"))
    }

    func testAllBuiltinsRenderExplicitScoreRequestWithExactSourceGate() throws {
        let input = try KernelResearchSetup(sourcePath: "kernel/my candidate.py", checkBudget: "2", allowScoredSubmissions: true,
            scoreBudget: "1", researchPrompt: "Check my source and score it if qualified.",
            scoreCriterion: "A clean first entry, only when I have no scored entry.", expectedAccount: "@Fixture-User")
        for project in try BuiltinProjects.load(resources: resources) {
            let prompt = try project.setupPrompt(input, resources: resources)
            XCTAssertTrue(prompt.contains("\"projectID\" : \"" + project.id + "\""))
            XCTAssertTrue(prompt.contains("\"authorizeScoredSubmissions\" : true"))
            XCTAssertTrue(prompt.contains("\"maximumScoredSubmissions\" : 1"))
            XCTAssertTrue(prompt.contains("\"expectedAccount\" : \"fixture-user\""))
            XCTAssertTrue(prompt.contains("blocks all checks and scored submissions"))
            XCTAssertTrue(prompt.contains("okc submit <projectID> <file> --json"))
            XCTAssertTrue(prompt.contains("completed clean official off-board check and matching recorded source hash"))
            XCTAssertTrue(prompt.contains("Do not alter the checked bytes"))
            XCTAssertTrue(try project.loginInstructions(resources: resources).contains("okc whoami"))
        }
    }

    func testExpectedAccountIsExplicitAndWhoamiParsingRejectsUnknownOutput() throws {
        XCTAssertThrowsError(try KernelResearchSetup(sourcePath: "", checkBudget: "0", researchPrompt: "Inspect"))
        XCTAssertThrowsError(try KernelResearchSetup(sourcePath: "", checkBudget: "0", researchPrompt: "Inspect", expectedAccount: "some account"))
        XCTAssertEqual(OKCLogin.account(from: "Fixture-User\n"), "fixture-user")
        XCTAssertEqual(OKCLogin.account(from: "signed in as @Fixture-User\n"), "fixture-user")
        XCTAssertNil(OKCLogin.account(from: "not signed in\n"))
        XCTAssertNil(OKCLogin.account(from: "fixture-user\nunexpected private output\n"))
        XCTAssertEqual(OKCLogin.normalizeAccount("  @Fixture-User "), "fixture-user")
    }

    func testLoginScriptQuotesPathsAndForcesOnlyOnExplicitAccountChoice() {
        let directory = URL(fileURLWithPath: "/Volumes/SSD O'Brien/$(not-a-command)/kernels")
        let home = URL(fileURLWithPath: "/Users/Name O'Brien")
        let script = OKCLogin.commandScript(directory: directory, home: home)
        XCTAssertTrue(script.contains(#"cd -- '/Volumes/SSD O'\''Brien/$(not-a-command)/kernels'"#))
        XCTAssertTrue(script.contains(#"okc_cli='/Users/Name O'\''Brien/.okc/bin/okc'"#))
        XCTAssertTrue(script.contains(#"if "$okc_cli" login; then"#))
        XCTAssertTrue(script.contains(#""$okc_cli" whoami"#))
        XCTAssertFalse(script.contains("--force"))
        let chosen = OKCLogin.commandScript(directory: directory, home: home, chooseAccount: true)
        XCTAssertTrue(chosen.contains(#"if "$okc_cli" login --force; then"#))
        XCTAssertFalse(chosen.contains("logout"))
        XCTAssertFalse(script.contains("api token"))
        XCTAssertFalse(script.contains("curl"))
        XCTAssertFalse(script.contains(#""$okc_cli" check"#))
        XCTAssertFalse(script.contains(#""$okc_cli" submit"#))
    }
}
