import Foundation
import ResearchCore

extension AppStore {
    func builtinDraft(for project: ResearchProject) -> KernelResearchDraft {
        builtinDrafts[project.id] ?? KernelResearchDraft(researchPrompt: builtinProject(project)?.suggestedPrompts.first ?? "")
    }

    func loadBuiltinDrafts() {
        do {
            let root = try Workspace.validate(settings.workspaceRoot)
            let file = root.appendingPathComponent("kernel-setups.json")
            try rejectKernelSetupSymlink(file)
            guard FileManager.default.fileExists(atPath: file.path) else { return }
            _ = try Workspace.validate(file.path)
            guard file.resolvingSymlinksInPath().path.hasPrefix(root.path + "/") else {
                throw ResearchError("Kernel setup storage points outside the workspace.")
            }
            builtinDrafts = try JSONDecoder().decode([String: KernelResearchDraft].self, from: Data(contentsOf: file))
            builtinDraftsReadable = true
        } catch {
            builtinDraftsReadable = false
            self.error = "Saved kernel setups could not be read and will not be overwritten. \(error.localizedDescription)"
        }
    }

    @discardableResult
    func setBuiltinDraft(_ draft: KernelResearchDraft, for project: ResearchProject) -> Bool {
        do {
            guard builtinProject(project) != nil else { throw ResearchError("The selected built-in project's resources are unavailable.") }
            guard builtinDraftsReadable else { throw ResearchError("Restore the unreadable kernel-setups.json before saving new settings. Its original contents were preserved.") }
            let root = try Workspace.validate(settings.workspaceRoot)
            let file = root.appendingPathComponent("kernel-setups.json")
            try rejectKernelSetupSymlink(file)
            var next = builtinDrafts
            next[project.id] = draft
            try Workspace.write(next, to: file)
            builtinDrafts = next
            return true
        } catch { self.error = error.localizedDescription; return false }
    }

    /// All project Play controls arrive here. No work starts on app load or on
    /// form edits; the configured request becomes a normal visible app chat.
    func playBuiltin(_ project: ResearchProject) {
        if let existing = sessions.first(where: { $0.projectID == project.id && runners[$0.id] != nil }) {
            selectSession(existing)
            return
        }
        guard !builtinPlayPending.contains(project.id) else { return }
        let draft = builtinDraft(for: project)
        let prompt: String
        do {
            guard builtinDraftsReadable else { throw ResearchError("Saved kernel setup is unreadable. Restore it before starting research.") }
            guard let definition = builtinProject(project) else { throw ResearchError("This built-in project's resources are unavailable.") }
            prompt = try definition.setupPrompt(draft.validatedSetup(), resources: resourceDirectory)
        } catch {
            select(project)
            self.error = "Complete this project's setup before pressing Play. \(error.localizedDescription)"
            return
        }
        guard setBuiltinDraft(draft, for: project) else { select(project); return }
        builtinPlayPending.insert(project.id)
        error = nil
        builtinPlayTasks[project.id] = Task {
            defer {
                builtinPlayPending.remove(project.id)
                builtinPlayTasks.removeValue(forKey: project.id)
            }
            do {
                let account: String
                if let builtinAccountVerifier { account = try await builtinAccountVerifier() }
                else { account = try await refreshOKCAccount() }
                try Task.checkCancellation()
                guard OKCLogin.normalizeAccount(account) == OKCLogin.normalizeAccount(draft.expectedAccount) else {
                    throw ResearchError("The signed-in OKC account does not match the account in this project's setup. Verify the account or sign in before pressing Play again.")
                }
                guard builtinDraft(for: project) == draft else {
                    throw ResearchError("The project setup changed while verifying the account. Review it and press Play again.")
                }
                if let existing = sessions.first(where: { $0.projectID == project.id && runners[$0.id] != nil }) {
                    selectSession(existing)
                    return
                }
                select(project)
                newChat()
                composer = prompt
                send()
            } catch is CancellationError {
                // Closing the app cancels only the pending preparation; no run
                // or quota-consuming attempt has been launched.
            } catch {
                select(project)
                self.error = error.localizedDescription
            }
        }
    }

    private func rejectKernelSetupSymlink(_ file: URL) throws {
        if let attributes = try? FileManager.default.attributesOfItem(atPath: file.path),
           attributes[.type] as? FileAttributeType == .typeSymbolicLink {
            throw ResearchError("Kernel setup storage is a symbolic link. It was preserved; choose a regular workspace file before continuing.")
        }
    }
}
