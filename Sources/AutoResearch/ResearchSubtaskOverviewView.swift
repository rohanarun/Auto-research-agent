import SwiftUI
import ResearchCore

struct ResearchSubtaskOverviewView: View {
    let task: ResearchSubtask
    let session: ResearchSession
    var configuredSummary: String? = nil
    @State private var overview: ResearchSubtaskOverview?
    @State private var expanded = true

    var body: some View {
        let description = overview ?? task.overview
        VStack(alignment: .leading, spacing: 9) {
            Button {
                withAnimation(.easeInOut(duration: 0.18)) { expanded.toggle() }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: task.symbol)
                    Text(task.overviewTitle).fontWeight(.semibold)
                    Spacer()
                    Text(overview == nil ? "Workflow overview" : session.scripted == true ? "Script experiment" : "Agent's experiment").foregroundStyle(.secondary)
                    Image(systemName: expanded ? "chevron.up" : "chevron.down")
                }.font(.system(size: 11)).contentShape(Rectangle())
            }.buttonStyle(.plain).accessibilityLabel(task.overviewTitle)
                .accessibilityValue(expanded ? "Expanded" : "Collapsed")
            Text(overview?.objective ?? configuredSummary ?? description.objective).font(.system(size: 13, weight: .medium))
                .fixedSize(horizontal: false, vertical: true)
            if expanded {
                Text(description.method).font(.system(size: 12)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
                Text(description.meaning).font(.system(size: 11)).foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(16).frame(maxWidth: .infinity, alignment: .leading)
        .background(Color(red: 0.30, green: 0.34, blue: 0.78).opacity(0.045), in: RoundedRectangle(cornerRadius: 10))
        .padding(.horizontal, 26).padding(.top, 12).padding(.bottom, 4)
        .task(id: session.state + (session.workingDirectory ?? "")) {
            repeat {
                if let directory = session.workingDirectory {
                    overview = LeanSearch.readOverview(at: LeanSearch.overviewURL(directory: URL(fileURLWithPath: directory), sessionID: session.id))
                }
                guard ["Preparing", "Running", "Stopping"].contains(session.state) else { break }
                do { try await Task.sleep(for: .seconds(2)) } catch { break }
            } while !Task.isCancelled
        }
    }
}
