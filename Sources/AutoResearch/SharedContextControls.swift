import SwiftUI
import ResearchCore

struct SharedContextControls: View {
    @ObservedObject var store: AppStore
    let project: ResearchProject
    private var options: SharedContextOptions { store.sharedConfiguration(project.id) }
    private var active: Bool { store.sharedActive(project.id) }
    private func binding<T>(_ key: WritableKeyPath<SharedContextOptions, T>) -> Binding<T> {
        Binding(get: { options[keyPath: key] }, set: { value in
            var updated = options; updated[keyPath: key] = value
            store.setSharedConfiguration(updated, for: project.id)
        })
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 16) {
                Toggle("Share context", isOn: binding(\.enabled)).toggleStyle(.switch).controlSize(.small)
                    .help("Run parallel agents in independent checkouts and periodically combine their contexts into fresh sessions. Uses your selected CLI account.")
                if options.enabled {
                    HStack(spacing: 5) {
                        Text("Agents")
                        TextField("Count", value: binding(\.agentCount), format: .number).frame(width: 42).accessibilityLabel("Parallel agent count")
                        Stepper("Agent count", value: binding(\.agentCount), in: 2...Int.max).labelsHidden()
                    }.disabled(active)
                    HStack(spacing: 5) {
                        Text("Share every")
                        TextField("Minutes", value: binding(\.intervalMinutes), format: .number).frame(width: 52).accessibilityLabel("Context sharing interval in minutes")
                        Text("min")
                    }.disabled(active)
                }
                Spacer(minLength: 0)
            }
            if let run = store.sharedRuns[project.id] {
                HStack(spacing: 12) {
                    Text("Round \(run.generation) · \(run.phase.rawValue)").foregroundStyle(.secondary)
                    if !run.sessionIDs.isEmpty {
                        Menu("Agents") {
                            ForEach(run.sessionIDs, id: \.self) { id in
                                if let session = store.sessions.first(where: { $0.id == id }) {
                                    Button("\(session.title) · \(session.state)") { store.selectSession(session) }
                                }
                            }
                        }.fixedSize()
                    }
                    Spacer(minLength: 0)
                    if active {
                        Button("Share now") { store.synchronizeSharedResearch(project.id) }.disabled(run.phase != .researching)
                        Button("Stop agents") { store.stopSharedResearch(project.id) }
                    }
                }
            } else if options.enabled {
                Text("Play starts \(options.agentCount) agents. Each sharing cycle saves their contexts and replaces their sessions.")
                    .foregroundStyle(.secondary).fixedSize(horizontal: false, vertical: true)
            }
        }
        .font(.system(size: 11))
        .textFieldStyle(.roundedBorder)
        .padding(.horizontal, 28).padding(.bottom, 14)
        .animation(.easeOut(duration: 0.16), value: options.enabled)
    }
}
