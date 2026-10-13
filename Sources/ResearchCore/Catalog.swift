import Foundation

public enum Catalog {
    public static func matches(_ pattern: String, _ text: String) -> [[String]] {
        guard let regex = try? NSRegularExpression(pattern: pattern, options: [.dotMatchesLineSeparators]) else { return [] }
        let ns = text as NSString
        return regex.matches(in: text, range: NSRange(location: 0, length: ns.length)).map { match in
            (0..<match.numberOfRanges).map { match.range(at: $0).location == NSNotFound ? "" : ns.substring(with: match.range(at: $0)) }
        }
    }
    static func clean(_ s: String) -> String {
        s.replacingOccurrences(of: "<[^>]+>", with: "", options: .regularExpression)
            .replacingOccurrences(of: "&amp;", with: "&").replacingOccurrences(of: "&#39;", with: "'")
            .replacingOccurrences(of: "&quot;", with: "\"").trimmingCharacters(in: .whitespacesAndNewlines)
    }
    public static func parse(_ html: String) throws -> [ResearchProject] {
        var projects: [ResearchProject] = []
        var seen = Set<String>()
        func add(_ project: ResearchProject) {
            if seen.insert(project.id).inserted { projects.append(project) }
        }
        // Source project cards contain their own canonical repository link and display copy.
        for article in matches("<article\\b[^>]*class=[\"'][^\"']*project-entry[^\"']*[\"'][^>]*>(.*?)</article>", html) {
            let body = article[1]
            let roots = matches("href=[\"']https://github\\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?[\"']", body)
            guard let repo = roots.last?[1] else { continue }
            let title = matches("<h3[^>]*>(.*?)</h3>", body).first?[1]
            let summary = matches("class=[\"']project-description[\"'][^>]*>\\s*<p>(.*?)</p>", body).first?[1] ?? ""
            let category = matches("data-category=[\"']([^\"']+)", article[0]).first?[1] ?? "Research"
            add(ResearchProject(repository: repo, name: title.map(clean), summary: clean(summary), category: clean(category)))
        }
        // The chart tracks are declared as data in the page, including tracks without static cards.
        for match in matches("\\brepo[\"']?\\s*:\\s*[\"']([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)[\"']", html) {
            add(ResearchProject(repository: match[1], category: "Mathematics"))
        }
        // Include canonical repository anchors used by the research launch and source sections.
        for match in matches("href=[\"']https://github\\.com/([A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+)/?[\"']", html) {
            add(ResearchProject(repository: match[1]))
        }
        guard !projects.isEmpty else { throw ResearchError("The catalog did not contain repository links. Your saved projects have been preserved.") }
        return projects
    }
    public static func fetch(_ address: String) async throws -> [ResearchProject] {
        guard let url = URL(string: address), url.scheme == "https" else { throw ResearchError("Use an HTTPS catalog URL.") }
        let (data, response) = try await URLSession.shared.data(from: url)
        guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw ResearchError("The catalog server could not be reached. Showing saved projects.") }
        return try parse(String(decoding: data, as: UTF8.self))
    }
}
