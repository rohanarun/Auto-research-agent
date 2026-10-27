import Foundation

/// Editable per-project form values. Saving a draft never starts research or
/// grants another run a fresh budget; only the user's Play action sends it.
public struct KernelResearchDraft: Codable, Equatable, Sendable {
    public var sourcePath: String
    public var checkBudget: String
    public var allowScoredSubmissions: Bool
    public var scoreBudget: String
    public var researchPrompt: String
    public var scoreCriterion: String
    public var expectedAccount: String
    public var useSuperExperiments: Bool

    public init(sourcePath: String = "", checkBudget: String = "", allowScoredSubmissions: Bool = false,
                scoreBudget: String = "", researchPrompt: String = "", scoreCriterion: String = "", expectedAccount: String = "",
                useSuperExperiments: Bool = true) {
        self.sourcePath = sourcePath
        self.checkBudget = checkBudget
        self.allowScoredSubmissions = allowScoredSubmissions
        self.scoreBudget = scoreBudget
        self.researchPrompt = researchPrompt
        self.scoreCriterion = scoreCriterion
        self.expectedAccount = expectedAccount
        self.useSuperExperiments = useSuperExperiments
    }

    public func validatedSetup() throws -> KernelResearchSetup {
        try KernelResearchSetup(sourcePath: sourcePath, checkBudget: checkBudget,
            allowScoredSubmissions: allowScoredSubmissions, scoreBudget: scoreBudget,
            researchPrompt: researchPrompt, scoreCriterion: scoreCriterion, expectedAccount: expectedAccount,
            useSuperExperiments: useSuperExperiments)
    }

    enum CodingKeys: String, CodingKey {
        case sourcePath, checkBudget, allowScoredSubmissions, scoreBudget, researchPrompt, scoreCriterion, expectedAccount, useSuperExperiments
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        sourcePath = try container.decodeIfPresent(String.self, forKey: .sourcePath) ?? ""
        checkBudget = try container.decodeIfPresent(String.self, forKey: .checkBudget) ?? ""
        allowScoredSubmissions = try container.decodeIfPresent(Bool.self, forKey: .allowScoredSubmissions) ?? false
        scoreBudget = try container.decodeIfPresent(String.self, forKey: .scoreBudget) ?? ""
        researchPrompt = try container.decodeIfPresent(String.self, forKey: .researchPrompt) ?? ""
        scoreCriterion = try container.decodeIfPresent(String.self, forKey: .scoreCriterion) ?? ""
        expectedAccount = try container.decodeIfPresent(String.self, forKey: .expectedAccount) ?? ""
        useSuperExperiments = try container.decodeIfPresent(Bool.self, forKey: .useSuperExperiments) ?? true
    }
}
