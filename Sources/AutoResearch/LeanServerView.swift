import SwiftUI

struct LeanServerView: View {
    @ObservedObject var server: LeanServerController
    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Shared Lean verification").font(.headline)
            Text("Search on your agents; compile and verify on this Mac. The server retains its dependency cache, candidate sources, failed checks and receipts.").foregroundStyle(.secondary)
            HStack {
                Text(server.storagePath).font(.caption).lineLimit(2).textSelection(.enabled)
                Spacer()
                Button("Choose server folder") { server.chooseStorage() }.disabled(server.isActive)
            }
            TextField("Installed Lean executable", text: $server.leanPath).textFieldStyle(.roundedBorder).disabled(server.isActive)
            HStack {
                Button("Start server") { server.start() }.disabled(server.isActive)
                Button("Stop server") { server.stop() }.disabled(!server.isActive)
                Button("Open profiles and logs") { server.openFolder() }
            }
            Text(server.status).font(.callout).textSelection(.enabled)
            if !server.localURL.isEmpty { Text(server.localURL).font(.system(.body, design: .monospaced)).textSelection(.enabled) }
            Text("Local research runs receive the connection automatically. The initial kernel-check profile tests the service; full QRH verification needs its prepared proof project and pinned challenge in server-config.json. Stop the server before editing profiles.").font(.caption).foregroundStyle(.secondary)
            Divider()
            Text("Allow other nodes to connect").font(.headline)
            TextField("ngrok executable", text: $server.ngrokPath).textFieldStyle(.roundedBorder).disabled(server.sharing)
            HStack {
                Button("Share with ngrok") { server.share() }.disabled(!server.running || server.sharing)
                Button("Stop sharing") { server.stopSharing() }.disabled(!server.sharing)
                Button("Copy access token") { server.copyToken() }.disabled(!server.running)
            }
            if !server.publicURL.isEmpty { Text(server.publicURL).font(.system(.body, design: .monospaced)).textSelection(.enabled) }
            Text("Uses your existing ngrok login. Share the HTTPS address and access token only with intended nodes. Requests are authenticated; submitted source runs in a restricted macOS process. Stopping sharing closes public access while local verification stays available.").font(.caption).foregroundStyle(.secondary)
            Spacer()
        }
    }
}
