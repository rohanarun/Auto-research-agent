import Foundation

public struct SharedContextOptions: Codable, Equatable, Sendable {
    public var parallelEnabled: Bool
    public var enabled: Bool
    public var agentCount: Int
    public var intervalMinutes: Double
    public var methods: ResearchMethodOptions?
    public var usesMethods: Bool { methods?.enabled == true }
    public init(enabled: Bool, agentCount: Int, intervalMinutes: Double, parallelEnabled: Bool? = nil) {
        self.parallelEnabled = parallelEnabled ?? enabled
        self.enabled = enabled; self.agentCount = agentCount; self.intervalMinutes = intervalMinutes
    }
    private enum CodingKeys: String, CodingKey { case parallelEnabled, enabled, agentCount, intervalMinutes, methods }
    public init(from decoder: Decoder) throws {
        let values = try decoder.container(keyedBy: CodingKeys.self)
        enabled = try values.decode(Bool.self, forKey: .enabled)
        parallelEnabled = try values.decodeIfPresent(Bool.self, forKey: .parallelEnabled) ?? enabled
        agentCount = try values.decode(Int.self, forKey: .agentCount)
        intervalMinutes = try values.decode(Double.self, forKey: .intervalMinutes)
        methods = try values.decodeIfPresent(ResearchMethodOptions.self, forKey: .methods)
    }
    public func validate() throws {
        try methods?.validate()
        guard agentCount >= 1 else { throw ResearchError("Research needs at least one agent.") }
        guard enabled else { return }
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
    public var sharesContext: Bool?
    public var agentCount: Int?
    public var isIndependent: Bool { sharesContext == false }
    public init(groupID: UUID, index: Int, generation: Int, previousSessionID: UUID?, contextFile: String?, sharesContext: Bool? = nil, agentCount: Int? = nil) {
        self.groupID = groupID; self.index = index; self.generation = generation
        self.previousSessionID = previousSessionID; self.contextFile = contextFile
        self.sharesContext = sharesContext; self.agentCount = agentCount
    }
}

/// Owns the replacement boundary: drained workers -> durable snapshot -> successful
/// file index -> new sessions. Cancellation or a failed publication never starts replacements.
@MainActor public final class SharedContextCycle {
    public enum Phase: String { case preparing = "Preparing agents", researching = "Researching", capturing = "Saving contexts", merging = "Sharing context files", completed = "Completed", stopped = "Stopped", failed = "Needs attention" }
    public struct Hooks {
        public var prepare: () async throws -> Void
        public var launch: (Int, URL?, [UUID]) throws -> [UUID]
        public var drain: ([UUID]) async -> Void
        public var capture: ([UUID], Int, URL?) async throws -> URL
        public var merge: (URL) async throws -> URL
        public var retire: ([UUID]) -> Void
        public var cancelOperation: () -> Void
        public var update: (Phase, Int, [UUID], String?) -> Void
        public var waitForCompletion: ([UUID]) async throws -> Void
        public init(prepare: @escaping () async throws -> Void, launch: @escaping (Int, URL?, [UUID]) throws -> [UUID], drain: @escaping ([UUID]) async -> Void, capture: @escaping ([UUID], Int, URL?) async throws -> URL, merge: @escaping (URL) async throws -> URL, retire: @escaping ([UUID]) -> Void, cancelOperation: @escaping () -> Void, update: @escaping (Phase, Int, [UUID], String?) -> Void, waitForCompletion: @escaping ([UUID]) async throws -> Void = { _ in }) {
            self.prepare = prepare; self.launch = launch; self.drain = drain; self.capture = capture; self.merge = merge
            self.retire = retire; self.cancelOperation = cancelOperation; self.update = update
            self.waitForCompletion = waitForCompletion
        }
    }
    public struct ResumeState {
        public let generation: Int
        public let sessionIDs: [UUID]
        public let context: URL?
        public init(generation: Int, sessionIDs: [UUID], context: URL?) {
            self.generation = generation; self.sessionIDs = sessionIDs; self.context = context
        }
    }
    private var resumeState: ResumeState?
    private let hooks: Hooks
    private let options: SharedContextOptions
    private var task: Task<Void, Never>?
    private var intervalTask: Task<Void, Error>?
    private var failure: String?
    public private(set) var active = false
    public init(options: SharedContextOptions, hooks: Hooks) { self.options = options; self.hooks = hooks }
    public func start(resuming: ResumeState? = nil) {
        guard !active else { return }; resumeState = resuming; active = true
        task = Task { await execute() }
    }
    public func synchronize() { intervalTask?.cancel() }
    public func stop(failure: String? = nil) {
        self.failure = failure; task?.cancel(); intervalTask?.cancel(); hooks.cancelOperation()
    }
    public func waitUntilStopped() async { await task?.value }
    private func execute() async {
        var ids: [UUID] = []; var generation = resumeState?.generation ?? 1; var context = resumeState?.context
        do {
            try options.validate()
            if let resumeState {
                guard resumeState.generation > 0, resumeState.sessionIDs.count == options.agentCount, Set(resumeState.sessionIDs).count == options.agentCount else { throw ResearchError("Invalid saved cycle membership; no workers resumed.") }
            }
            hooks.update(.preparing, generation, ids, nil)
            try await hooks.prepare(); try Task.checkCancellation()
            ids = try hooks.launch(generation, context, resumeState?.sessionIDs ?? [])
            if !options.enabled {
                hooks.update(.researching, generation, ids, nil)
                try await hooks.waitForCompletion(ids)
                try Task.checkCancellation()
                active = false
                hooks.update(.completed, generation, ids, nil)
                return
            }
            while true {
                hooks.update(.researching, generation, ids, nil)
                let sleeper = Task<Void, Error> { try await Task.sleep(for: .seconds(options.intervalMinutes * 60)) }
                intervalTask = sleeper
                // Sync now cancels only this timer. Stop cancels the parent as well.
                _ = await sleeper.result; intervalTask = nil
                try Task.checkCancellation()
                hooks.update(.capturing, generation, ids, nil)
                await hooks.drain(ids); try Task.checkCancellation()
                let snapshot = try await hooks.capture(ids, generation, context)
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
