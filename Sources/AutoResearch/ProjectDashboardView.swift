import SwiftUI
import Charts
import ResearchCore

@MainActor private final class DashboardLoader: ObservableObject {
    @Published var data: ResearchDashboard?
    @Published var loading = false
    @Published var error: String?
    var address = ""
    func refresh(_ address: String) async -> Double? {
        guard !loading else { return nil }
        if self.address != address { data = nil; self.address = address }
        loading = true; defer { loading = false }
        do {
            let next = try await ResearchDashboard.fetch(catalogURL: address)
            try Task.checkCancellation()
            data = next; error = nil
            let delay = next.history.refresh_after_seconds
            return delay.isFinite && delay > 0 ? delay : nil
        } catch is CancellationError { return nil }
        catch {
            self.error = "Update unavailable"
            let delay = data?.history.refresh_after_seconds
            return delay.map { $0.isFinite && $0 > 0 } == true ? delay : nil
        }
    }
}

struct ProjectDashboardView: View {
    let project: ResearchProject
    let catalogURL: String
    @StateObject private var loader = DashboardLoader()
    @State private var groupID = ""
    @State private var refreshID = UUID()
    @State private var observed = false
    @State private var selectedDate: Date?
    @State private var showSourceStatus = false
    private let tint = Color(red: 0.30, green: 0.34, blue: 0.78)
    private var groups: [DashboardGroup] { loader.data?.groups(for: project.repository) ?? [] }
    private var group: DashboardGroup? { groups.first { $0.id == groupID } ?? groups.first }
    private var history: DashboardHistory? { observed ? group?.observed : group?.reconstructed }
    private var track: DashboardTrack? { loader.data?.track(for: project.repository) }
    private var sourceError: String? { loader.error ?? loader.data?.history.source_error ?? loader.data?.research.error }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 24) {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(loader.data?.scope(for: project.repository)?.name ?? project.name.replacingOccurrences(of: "-", with: " "))
                            .font(.system(size: 28, weight: .semibold)).lineLimit(2)
                        if !project.summary.isEmpty { Text(project.summary).font(.system(size: 13)).foregroundStyle(.secondary).lineLimit(2) }
                    }
                    Spacer(minLength: 16)
                    if let url = project.githubURL { Link(destination: url) { Image(systemName: "arrow.up.right").padding(9) }.help("Open repository") }
                    Button { refreshID = UUID() } label: {
                        Image(systemName: "arrow.clockwise").padding(9)
                    }.buttonStyle(.plain).disabled(loader.loading).help("Refresh chart data")
                }
                if let group, let history {
                    chartCard(group, history)
                    recentActivity
                } else if loader.loading && loader.data == nil {
                    ProgressView("Loading chart…").frame(maxWidth: .infinity, minHeight: 260)
                } else {
                    VStack(spacing: 10) {
                        Image(systemName: "chart.xyaxis.line").font(.system(size: 28)).foregroundStyle(.secondary)
                        Text(loader.error == nil ? "No chart published yet" : "Chart data is unavailable").font(.system(size: 17, weight: .medium))
                        Text(loader.error == nil ? "You can explore the repository before starting." : "Refresh to try again.").font(.system(size: 13)).foregroundStyle(.secondary)
                    }.frame(maxWidth: .infinity, minHeight: 250).background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 16))
                }
                HStack(spacing: 8) {
                    Circle().fill(sourceError == nil ? Color.green : .orange).frame(width: 5, height: 5)
                    Text(loader.error == nil ? "Backend data" : "Last loaded data").font(.system(size: 11))
                    if let timestamp = track?.checked_at ?? loader.data?.history.as_of, let date = parseDate(timestamp) {
                        Text(date, style: .time).font(.system(size: 11)).monospacedDigit()
                    }
                    if sourceError != nil {
                        Button { showSourceStatus.toggle() } label: { Image(systemName: "info.circle") }.buttonStyle(.plain)
                            .help("Some backend data may be delayed")
                            .popover(isPresented: $showSourceStatus) { Text(sourceError ?? "").font(.system(size: 12)).padding(16).frame(width: 330) }
                    }
                    Spacer()
                    if let url = URL(string: catalogURL), url.scheme == "https" { Link("View source", destination: url).font(.system(size: 11)) }
                }.foregroundStyle(.secondary)
            }.padding(28)
        }
        .task(id: catalogURL + refreshID.uuidString) {
            while !Task.isCancelled {
                guard let delay = await loader.refresh(catalogURL) else { break }
                do { try await Task.sleep(for: .seconds(delay)) } catch { break }
            }
        }
        .onChange(of: project.id) { _, _ in groupID = ""; selectedDate = nil }
    }

    private func chartCard(_ group: DashboardGroup, _ history: DashboardHistory) -> some View {
        let intervals = history.validIntervals
        let last = intervals.last
        let selected = selectedDate.flatMap { date in intervals.last { $0.from <= date.timeIntervalSince1970 } }
        let displayed = selected ?? last
        return VStack(alignment: .leading, spacing: 18) {
            HStack(alignment: .top) {
                VStack(alignment: .leading, spacing: 5) {
                    Text(number(selected?.value ?? history.value)).font(.system(size: 36, weight: .semibold)).monospacedDigit().contentTransition(.numericText())
                    HStack(spacing: 8) {
                        Text(track?.status(for: displayed) ?? "Source claim")
                        if let url = displayed?.sourceURL { Link("Source ↗", destination: url) }
                    }.font(.system(size: 11)).foregroundStyle(.secondary)
                }
                Spacer()
                Picker("History", selection: $observed) {
                    Text("Estimated").tag(false)
                    Text("Observed").tag(true)
                }.pickerStyle(.segmented).labelsHidden().frame(width: 180)
                    .help("Backend claim history with estimated dates, or changes observed by the backend")
            }
            if groups.count > 1 {
                Picker("Metric", selection: Binding(get: { group.id }, set: { groupID = $0; selectedDate = nil })) {
                    ForEach(groups) { Text($0.name).tag($0.id) }
                }.labelsHidden()
            } else { Text(group.name).font(.system(size: 12, weight: .medium)).foregroundStyle(.secondary) }
            if intervals.isEmpty {
                Text("No history available for this view.").foregroundStyle(.secondary).frame(maxWidth: .infinity, minHeight: 230)
            } else {
                Chart {
                    ForEach(Array(intervals.enumerated()), id: \.offset) { _, interval in
                        LineMark(x: .value("Date", Date(timeIntervalSince1970: interval.from)), y: .value("Claim", interval.value!))
                            .interpolationMethod(.stepEnd).foregroundStyle(tint).lineStyle(StrokeStyle(lineWidth: 2.5))
                    }
                    if let last {
                        LineMark(x: .value("Date", Date(timeIntervalSince1970: last.to)), y: .value("Claim", last.value!))
                            .interpolationMethod(.stepEnd).foregroundStyle(tint).lineStyle(StrokeStyle(lineWidth: 2.5))
                    }
                    if let selectedDate, let selected, let value = selected.value {
                        RuleMark(x: .value("Selected date", selectedDate)).foregroundStyle(.gray.opacity(0.3))
                        PointMark(x: .value("Date", selectedDate), y: .value("Claim", value)).foregroundStyle(tint)
                    }
                }
                .chartXSelection(value: $selectedDate)
                .chartYScale(domain: .automatic(includesZero: false))
                .chartYAxis { AxisMarks(position: .leading) { value in
                    AxisGridLine().foregroundStyle(Color.black.opacity(0.06))
                    AxisValueLabel { if let value = value.as(Double.self) { Text(number(value)).font(.system(size: 10)).monospacedDigit() } }
                } }
                .chartXAxis { AxisMarks(values: .automatic(desiredCount: 4)) }
                .frame(height: 245)
            }
            HStack {
                Text(group.direction == "up" ? "Higher is better" : group.direction == "down" ? "Lower is better" : "Backend metric")
                Spacer()
                if let selectedDate { Text(selectedDate, format: .dateTime.month(.abbreviated).day().hour().minute()) }
                else { Text(observed ? "Observed claims" : "Claims · estimated timing") }
            }.font(.system(size: 10)).foregroundStyle(.secondary)
        }.padding(22).background(Color.black.opacity(0.025), in: RoundedRectangle(cornerRadius: 16))
        .onChange(of: observed) { _, _ in selectedDate = nil }
    }
    @ViewBuilder private var recentActivity: some View {
        if let track, !track.pulls.isEmpty {
            VStack(alignment: .leading, spacing: 14) {
                HStack { Text("Recent activity").font(.system(size: 13, weight: .semibold)); Spacer(); Text("\(track.pulls.count) PRs").font(.system(size: 11)).foregroundStyle(.secondary) }
                ForEach(track.recentPulls.prefix(3)) { pull in
                    if let url = pull.url {
                        Link(destination: url) {
                            HStack(spacing: 12) {
                                Text("#" + String(pull.number)).font(.system(size: 11)).foregroundStyle(tint).frame(width: 48, alignment: .leading)
                                Text(pull.title).font(.system(size: 12)).foregroundStyle(.primary).lineLimit(1)
                                Spacer(minLength: 8)
                                Text(pull.status).font(.system(size: 10)).foregroundStyle(.secondary)
                                Image(systemName: "arrow.up.right").font(.system(size: 9)).foregroundStyle(.secondary)
                            }
                        }
                    }
                }
            }
        }
    }
    private func number(_ value: Double?) -> String {
        guard let value, value.isFinite else { return "—" }
        let formatter = NumberFormatter()
        formatter.numberStyle = value != 0 && (abs(value) < 0.001 || abs(value) >= 1_000_000) ? .scientific : .decimal
        formatter.usesSignificantDigits = true; formatter.maximumSignificantDigits = 7
        return formatter.string(from: NSNumber(value: value)) ?? String(value)
    }
    private func parseDate(_ text: String) -> Date? {
        let formatter = ISO8601DateFormatter(); formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter.date(from: text) ?? ISO8601DateFormatter().date(from: text)
    }
}
