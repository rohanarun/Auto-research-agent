import Foundation

public struct SharedContextOptions: Codable, Equatable, Sendable {
    public var enabled: Bool
    public var agentCount: Int
    public var intervalMinutes: Double
    public init(enabled: Bool, agentCount: Int, intervalMinutes: Double) {
        self.enabled = enabled; self.agentCount = agentCount; self.intervalMinutes = intervalMinutes
    }
    public func validate() throws {
        guard agentCount >= 2 else { throw ResearchError("Shared context needs at least two agents.") }
        guard intervalMinutes.isFinite, intervalMinutes > 0, intervalMinutes * 60 < Double(Int64.max) / 1_000_000_000 else {
            throw ResearchError("Enter a positive synchronization interval in minutes.")
        }
    }
}

public struct SharedAgent: Codable, Sendable {
    public var groupID: UUID
    public var index: Int
    public var generation: Int
    public var previousSessionID: UUID?
    public var contextFile: String?
    public init(groupID: UUID, index: Int, generation: Int, previousSessionID: UUID?, contextFile: String?) {
        self.groupID = groupID; self.index = index; self.generation = generation
        self.previousSessionID = previousSessionID; self.contextFile = contextFile
    }
}

/// Owns the replacement boundary: drained workers -> durable snapshot -> successful
/// synthesis -> new sessions. Cancellation or a failed merge never starts replacements.
@MainActor public final class SharedContextCycle {
    public enum Phase: String { case preparing = "Preparing agents", researching = "Researching", capturing = "Saving contexts", merging = "Combining contexts", stopped = "Stopped", failed = "Needs attention" }
    public struct Hooks {
        public var prepare: () async throws -> Void
        public var launch: (Int, URL?, [UUID]) throws -> [UUID]
        public var drain: ([UUID]) async -> Void
        public var capture: ([UUID], Int, URL?) throws -> URL
        public var merge: (URL) async throws -> URL
        public var retire: ([UUID]) -> Void
        public var cancelOperation: () -> Void
        public var update: (Phase, Int, [UUID], String?) -> Void
        public init(prepare: @escaping () async throws -> Void, launch: @escaping (Int, URL?, [UUID]) throws -> [UUID], drain: @escaping ([UUID]) async -> Void, capture: @escaping ([UUID], Int, URL?) throws -> URL, merge: @escaping (URL) async throws -> URL, retire: @escaping ([UUID]) -> Void, cancelOperation: @escaping () -> Void, update: @escaping (Phase, Int, [UUID], String?) -> Void) {
            self.prepare = prepare; self.launch = launch; self.drain = drain; self.capture = capture; self.merge = merge
            self.retire = retire; self.cancelOperation = cancelOperation; self.update = update
        }
    }
    private let hooks: Hooks
    private let options: SharedContextOptions
    private var task: Task<Void, Never>?
    private var intervalTask: Task<Void, Error>?
    private var failure: String?
    public private(set) var active = false
    public init(options: SharedContextOptions, hooks: Hooks) { self.options = options; self.hooks = hooks }
    public func start() {
        guard !active else { return }; active = true
        task = Task { await execute() }
    }
    public func synchronize() { intervalTask?.cancel() }
    public func stop(failure: String? = nil) {
        self.failure = failure; task?.cancel(); intervalTask?.cancel(); hooks.cancelOperation()
    }
    public func waitUntilStopped() async { await task?.value }
    private func execute() async {
        var ids: [UUID] = []; var generation = 1; var context: URL?
        do {
            try options.validate()
            hooks.update(.preparing, generation, ids, nil)
            try await hooks.prepare(); try Task.checkCancellation()
            ids = try hooks.launch(generation, nil, [])
            while true {
                hooks.update(.researching, generation, ids, nil)
                let sleeper = Task<Void, Error> { try await Task.sleep(for: .seconds(options.intervalMinutes * 60)) }
                intervalTask = sleeper
                // Sync now cancels only this timer. Stop cancels the parent as well.
                _ = await sleeper.result; intervalTask = nil
                try Task.checkCancellation()
                hooks.update(.capturing, generation, ids, nil)
                await hooks.drain(ids); try Task.checkCancellation()
                let snapshot = try hooks.capture(ids, generation, context)
                hooks.update(.merging, generation, ids, nil)
                let combined = try await hooks.merge(snapshot)
                try Task.checkCancellation()
                hooks.retire(ids)
                generation += 1; context = combined
                ids = try hooks.launch(generation, context, ids)
            }
        } catch {
            hooks.cancelOperation()
            await hooks.drain(ids)
            active = false
            let cancelled = Task.isCancelled && failure == nil
            hooks.update(cancelled ? .stopped : .failed, generation, ids, cancelled ? nil : failure ?? error.localizedDescription)
        }
        active = false
    }
}
