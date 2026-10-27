import Foundation

/// User-supplied inputs only. The editable resource template defines research
/// behavior; this type validates budgets and safely serializes the request.
public struct KernelResearchSetup: Sendable {
    public let sourcePath: String
    public let checkBudget: Int
    public let scoreBudget: Int
    public let researchPrompt: String
    public let scoreCriterion: String
    public let expectedAccount: String
    public let useSuperExperiments: Bool

    public init(sourcePath: String, checkBudget: String, allowScoredSubmissions: Bool = false,
                scoreBudget: String = "", researchPrompt: String, scoreCriterion: String = "", expectedAccount: String = "",
                useSuperExperiments: Bool = true) throws {
        self.useSuperExperiments = useSuperExperiments
        self.expectedAccount = OKCLogin.normalizeAccount(expectedAccount)
        guard OKCLogin.isAccountName(self.expectedAccount) else {
            throw ResearchError("Enter the GitHub/OKC username you intend to use, then verify that account before pressing Play.")
        }
        self.sourcePath = sourcePath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !self.sourcePath.contains("\0") else { throw ResearchError("The kernel path contains an invalid character.") }
        self.checkBudget = try Self.budget(checkBudget, name: "check", positive: false)
        self.scoreBudget = allowScoredSubmissions ? try Self.budget(scoreBudget, name: "scored submission", positive: true) : 0
        self.researchPrompt = researchPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        self.scoreCriterion = scoreCriterion.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !self.researchPrompt.isEmpty else { throw ResearchError("Enter a research prompt before preparing the request.") }
        guard self.scoreBudget == 0 || !self.scoreCriterion.isEmpty else {
            throw ResearchError("Describe when a scored submission is allowed, including your first-baseline or improvement criterion.")
        }
    }

    private static func budget(_ text: String, name: String, positive: Bool) throws -> Int {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, value.utf8.allSatisfy({ $0 >= 48 && $0 <= 57 }),
              let count = Int(value), count >= (positive ? 1 : 0) else {
            throw ResearchError("Enter a \(positive ? "positive" : "nonnegative") whole-number \(name) budget.")
        }
        return count
    }

    public func render(template: String, projectID: String, projectName: String) throws -> String {
        let marker = "{{PROJECT_INPUTS_JSON}}"
        let sections = template.components(separatedBy: marker)
        guard sections.count == 2 else { throw ResearchError("The kernel setup template must contain its input placeholder exactly once.") }
        struct Inputs: Encodable {
            let projectID: String
            let projectName: String
            let expectedAccount: String
            let sourcePath: String?
            let maximumOfficialChecks: Int
            let maximumScoredSubmissions: Int
            let authorizeScoredSubmissions: Bool
            let researchPrompt: String
            let scoreCriterion: String?
            let useSuperExperiments: Bool
            let superExperimentsURL: String
        }
        let input = Inputs(projectID: projectID, projectName: projectName, expectedAccount: expectedAccount,
            sourcePath: sourcePath.isEmpty ? nil : sourcePath,
            maximumOfficialChecks: checkBudget, maximumScoredSubmissions: scoreBudget,
            authorizeScoredSubmissions: scoreBudget > 0, researchPrompt: researchPrompt,
            scoreCriterion: scoreCriterion.isEmpty ? nil : scoreCriterion,
            useSuperExperiments: useSuperExperiments,
            superExperimentsURL: "https://app.getsupers.com/experiments")
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
        let json = String(decoding: try encoder.encode(input), as: UTF8.self)
        // Insert once; user text resembling placeholders stays literal data.
        return sections[0] + json + sections[1]
    }
}

public enum OKCLogin {
    public static func commandScript(directory: URL, home: URL, chooseAccount: Bool = false) -> String {
        let fallback = home.appendingPathComponent(".okc/bin/okc").path
        let loginArguments = chooseAccount ? "login --force" : "login"
        return """
        #!/bin/zsh
        cd -- \(quote(directory.path)) || exit 1
        okc_cli=$(command -v okc 2>/dev/null)
        if [[ -z "$okc_cli" || ! -x "$okc_cli" ]]; then
          okc_cli=\(quote(fallback))
        fi
        if [[ ! -x "$okc_cli" ]]; then
          printf '%s\\n' 'OKC CLI is not installed. Follow the official instructions at https://openkernelchallenge.dev/ and then return to Choose account.'
          exit 1
        fi
        printf '%s\\n' 'Sign in with your own account and personally review any challenge rules. No checks or scored submissions run here.'
        if "$okc_cli" \(loginArguments); then
          "$okc_cli" whoami
        else
          exit $?
        fi
        """ + "\n"
    }

    public static func normalizeAccount(_ text: String) -> String {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return (value.hasPrefix("@") ? String(value.dropFirst()) : value).lowercased()
    }

    public static func isAccountName(_ text: String) -> Bool {
        text.range(of: "\\A[a-zA-Z0-9][a-zA-Z0-9_-]*\\z", options: .regularExpression) != nil
    }

    /// Interpret the official whoami identity line only. Unknown output must
    /// not be displayed verbatim or mistaken for a verified account.
    public static func account(from output: String) -> String? {
        let clean = output.replacingOccurrences(of: "\\u001B\\[[0-?]*[ -/]*[@-~]", with: "", options: .regularExpression)
        let bare = clean.trimmingCharacters(in: .whitespacesAndNewlines)
        if isAccountName(bare) { return normalizeAccount(bare) }
        let pattern = "(?im)^\\s*signed in as\\s+@?([a-zA-Z0-9][a-zA-Z0-9_-]*)\\s*$"
        guard let expression = try? NSRegularExpression(pattern: pattern),
              let match = expression.firstMatch(in: clean, range: NSRange(clean.startIndex..., in: clean)),
              let range = Range(match.range(at: 1), in: clean) else { return nil }
        return normalizeAccount(String(clean[range]))
    }

    private static func quote(_ text: String) -> String {
        "'" + text.replacingOccurrences(of: "'", with: "'\\''") + "'"
    }
}
