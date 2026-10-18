import SwiftUI
import ResearchCore

/// A single toolbar row; detailed configuration stays in the settings popover.
struct SharedContextControls: View {
    @ObservedObject var store: AppStore
    let project: ResearchProject
    @State private var showingConfiguration = false
    private var options: SharedContextOptions { store.sharedConfiguration(project.id) }
    private var active: Bool { store.sharedActive(project.id) || store.running(project) }
    private var group: UUID? { store.relevantSharedGroup(project.id) }
    private var agents: [ResearchSession] {
        guard let group else { return [] }
        return SharedTranscript.latestAgents(in: store.sessions, projectID: project.id, groupID: group)
    }
    private var mode: String {
        if options.usesMethods {
            let names = [options.methods?.rsiTop == true ? "RSI-top" : nil,
                         options.methods?.forkAndFlush == true ? "Fork-and-flush" : nil].compactMap { $0 }.joined(separator: " + ")
            return "\(options.methods?.branches ?? 0) branches · " + names
        }
        return options.parallelEnabled ? "\(options.agentCount) agents · \(options.enabled ? "Sharing" : "Independent")" : "Single agent"
    }
    private func methodBinding<T>(_ key: WritableKeyPath<ResearchMethodOptions, T>, value: T) -> Binding<T> {
        Binding(get: { (try? store.methodConfiguration(project.id))?[keyPath: key] ?? value }, set: { newValue in
            do {
                var methods = try store.methodConfiguration(project.id); methods[keyPath: key] = newValue
                var updated = options; updated.methods = methods
                store.setSharedConfiguration(updated, for: project.id)
            } catch { store.error = error.localizedDescription }
        })
    }
    private func binding<T>(_ key: WritableKeyPath<SharedContextOptions, T>) -> Binding<T> {
        Binding(get: { options[keyPath: key] }, set: { value in
            var updated = options; updated[keyPath: key] = value
            store.setSharedConfiguration(updated, for: project.id)
        })
    }
    var body: some View {
        HStack(spacing: 10) {
            Button { showingConfiguration.toggle() } label: {
                HStack(spacing: 5) {
                    Image(systemName: options.parallelEnabled ? "person.2" : "person")
                    Text(mode)
                    Image(systemName: "chevron.down").font(.system(size: 8, weight: .semibold))
                }
            }
            .help("Research settings").accessibilityLabel("Research settings: \(mode)")
            .popover(isPresented: $showingConfiguration, arrowEdge: .bottom) { configuration }
            if let group {
                Menu {
                    Button("All agents") { store.selectSharedGroup(group, projectID: project.id) }
                    if agents.allSatisfy({ $0.cloud != nil }), !agents.isEmpty {
                        Button("Resume saved group (\(agents.count) agents)") {
                            store.resumeCloudResearch(project, groupID: group)
                        }
                        .disabled(active)
                        .help("Resume the same sandboxes, conversations and shared context; no new group is created.")
                        .accessibilityLabel("Resume saved group")
                    }
                    Divider()
                    ForEach(agents) { session in
                        Button("Agent \(session.sharedAgent!.index) · \(session.state)") { store.selectSession(session) }
                    }
                    if active && options.enabled {
                        Divider()
                        Button("Share now") { store.synchronizeSharedResearch(project.id) }
                            .disabled(store.sharedRuns[project.id]?.phase != .researching)
                    }
                } label: {
                    Text(store.session?.sharedAgent.map { "Agent \($0.index)" } ?? "All agents")
                }.menuStyle(.borderlessButton).fixedSize()
            }
            if let run = store.sharedRuns[project.id] {
                HStack(spacing: 5) {
                    if active { ProgressView().controlSize(.mini) }
                    Text(run.phase.rawValue).foregroundStyle(.secondary).lineLimit(1)
                }
                if active {
                    Button { store.stopSharedResearch(project.id) } label: { Image(systemName: "stop.fill") }
                        .help("Stop agents").accessibilityLabel("Stop agents")
                }
            }
        }.font(.system(size: 11)).buttonStyle(.plain).controlSize(.small)
    }
    private var configuration: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text("Research settings").font(.system(size: 13, weight: .semibold))
            Toggle("Parallel agents", isOn: binding(\.parallelEnabled)).toggleStyle(.switch)
                .disabled(options.usesMethods)
            if options.parallelEnabled && !options.usesMethods {
                HStack {
                    Text("Agents")
                    Spacer()
                    TextField("Count", value: binding(\.agentCount), format: .number)
                        .frame(width: 48).accessibilityLabel("Parallel agent count")
                    Stepper("Agent count", value: binding(\.agentCount), in: 2...Int.max).labelsHidden()
                }
                Toggle("Share context", isOn: binding(\.enabled)).toggleStyle(.switch)
                if options.enabled {
                    HStack {
                        Text("Share every")
                        Spacer()
                        TextField("Minutes", value: binding(\.intervalMinutes), format: .number)
                            .frame(width: 52).accessibilityLabel("Context sharing interval in minutes")
                        Text("min")
                    }
                }
            }
            Divider()
            Toggle("RSI-top exploration", isOn: methodBinding(\.rsiTop, value: false)).toggleStyle(.checkbox)
                .accessibilityLabel("RSI-top exploration")
                .help("Fixed task metrics and diverse experiments when verified progress stalls.")
            Toggle("Fork-and-flush", isOn: methodBinding(\.forkAndFlush, value: false)).toggleStyle(.checkbox)
                .accessibilityLabel("Fork-and-flush")
                .help("Copy the leader workspace into fresh-context branches, evaluate them, and continue the strongest valid result.")
            if options.usesMethods, let methods = try? store.methodConfiguration(project.id) {
                HStack {
                    Text("Branches")
                    Spacer()
                    TextField("Branches", value: methodBinding(\.branches, value: methods.branches), format: .number).frame(width: 52)
                }
                HStack {
                    Text("Explore / continue (min)")
                    Spacer()
                    TextField("Explore", value: methodBinding(\.explorationMinutes, value: methods.explorationMinutes), format: .number).frame(width: 44)
                    TextField("Continue", value: methodBinding(\.exploitationMinutes, value: methods.exploitationMinutes), format: .number).frame(width: 44)
                }
                HStack {
                    Text("Rounds per batch")
                    Spacer()
                    TextField("Rounds", value: methodBinding(\.rounds, value: methods.rounds), format: .number).frame(width: 52)
                }
                Toggle("Share branch evidence", isOn: binding(\.enabled)).toggleStyle(.switch)
                if options.enabled {
                    HStack {
                        Text("Share every")
                        Spacer()
                        TextField("Minutes", value: binding(\.intervalMinutes), format: .number).frame(width: 52)
                            .accessibilityLabel("Method context sharing interval in minutes")
                        Text("min")
                    }
                }
                Text("\(methods.branches) branches explore the same objective together, then one selected leader continues. Fresh contexts return at fork boundaries. Rounds limit each batch. Enable Keep researching to have the coordinator review results and plan a further batch. Stop ends the run.")
                    .font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
            Text(active ? "Stop the current run to change its settings." : options.usesMethods ? "Time-budgeted adaptations. The model defines the task evaluator and metrics before experiments."
                 : options.enabled && options.parallelEnabled
                 ? "Agents periodically share compact checkpoints in fresh sessions."
                 : "Each agent works independently on the same problem.")
                .font(.system(size: 11)).foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
        }
        .disabled(active).font(.system(size: 12)).controlSize(.small)
        .textFieldStyle(.roundedBorder).padding(20).frame(width: 340)
    }
}
