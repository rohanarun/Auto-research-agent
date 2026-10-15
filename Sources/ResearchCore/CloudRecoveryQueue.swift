import Foundation

/// Recovery owns a full local archive until verification completes. Admit one
/// export at a time before creating network requests, so a queued export cannot
/// exhaust an HTTP timeout or compete with other full archive downloads.
@MainActor public final class CloudRecoveryQueue {
    private var pending: [String: Task<Void, Error>] = [:]
    private var tail: Task<Void, Never>?
    public var pendingCount: Int { pending.count }
    public init() {}

    public func run(id: String, operation: @escaping @MainActor () async throws -> Void) async throws {
        if let existing = pending[id] { try await existing.value;return }
        let predecessor = tail
        // An unstructured task deliberately survives cancellation of the model
        // turn that requested cleanup. No model operation is repeated here.
        let task = Task { @MainActor in
            await predecessor?.value
            try await operation()
        }
        pending[id] = task
        tail = Task { _ = try? await task.value }
        defer { pending.removeValue(forKey:id) }
        try await task.value
    }
}
