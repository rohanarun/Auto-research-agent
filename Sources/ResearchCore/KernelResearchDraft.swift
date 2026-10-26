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

    public init(sourcePath: String = "", checkBudget: String = "", allowScoredSubmissions: Bool = false,
                scoreBudget: String = "", researchPrompt: String = "", scoreCriterion: String = "", expectedAccount: String = "") {
        self.sourcePath = sourcePath
        self.checkBudget = checkBudget
        self.allowScoredSubmissions = allowScoredSubmissions
        self.scoreBudget = scoreBudget
        self.researchPrompt = researchPrompt
        self.scoreCriterion = scoreCriterion
        self.expectedAccount = expectedAccount
    }

    public func validatedSetup() throws -> KernelResearchSetup {
        try KernelResearchSetup(sourcePath: sourcePath, checkBudget: checkBudget,
            allowScoredSubmissions: allowScoredSubmissions, scoreBudget: scoreBudget,
            researchPrompt: researchPrompt, scoreCriterion: scoreCriterion, expectedAccount: expectedAccount)
    }
}
