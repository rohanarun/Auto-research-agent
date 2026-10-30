import XCTest
import SwiftUI
import AppKit
import ResearchCore
@testable import AutoResearch

final class Agent37SettingsTests: XCTestCase {
    @MainActor func testProviderSwitchPersistsSeparateSettingsAndBlocksWrongChatResume() throws {
        let projectRoot = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let resources = projectRoot.appendingPathComponent("Resources")
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:resources.appendingPathComponent("Defaults.json")))
        let runtime = try Workspace.directory(projectRoot.path,"tmp/agent37-settings-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:runtime) }
        settings.workspaceRoot = runtime.path
        let store = AppStore(resourceDirectory:resources,defaults:settings)
        let daytonaInstall = store.daytona.installID
        store.daytona.connectionSandboxID = UUID().uuidString
        store.cloudOwnedIDs = [store.daytona.connectionSandboxID!]
        let daytonaConnection = store.daytona.connectionSandboxID
        store.selectSandboxProvider(.agent37)
        XCTAssertNil(store.error);XCTAssertEqual(store.daytona.sandboxProvider,.agent37)
        XCTAssertEqual(store.daytona.snapshot,"agent37-base");XCTAssertTrue(store.cloudOwnedIDs.isEmpty)
        XCTAssertNil(store.daytona.connectionSandboxID);XCTAssertNotEqual(store.daytona.installID,daytonaInstall)
        store.daytona.connectionSandboxID = "ab12cd34ef";store.cloudOwnedIDs = ["ab12cd34ef"]
        store.daytona.enabled = true;try store.saveDaytona()
        let restored = AppStore(resourceDirectory:resources,defaults:settings)
        XCTAssertEqual(restored.daytona.sandboxProvider,.agent37);XCTAssertEqual(restored.cloudOwnedIDs,["ab12cd34ef"])
        let project = ResearchProject(repository:"owner/repo",name:"Fixture")
        var chat = ResearchSession(projectID:project.id,title:"Daytona",provider:.codex)
        chat.cloud = CloudSession(sandboxID:daytonaConnection!,stateID:"existing")
        store.sessions = [chat]
        store.launchCloud(chat.id,project:project,request:"continue")
        XCTAssertTrue(store.error?.contains("Select Daytona") == true);XCTAssertTrue(store.runners.isEmpty)
        store.error = nil
        store.runners[chat.id] = ProcessRunner()
        store.selectSandboxProvider(.daytona)
        XCTAssertEqual(store.daytona.sandboxProvider,.agent37,"Provider switching must not race an active runner")
        store.runners.removeAll()
        store.selectSandboxProvider(.daytona)
        XCTAssertEqual(store.daytona.installID,daytonaInstall);XCTAssertEqual(store.daytona.connectionSandboxID,daytonaConnection)
        XCTAssertEqual(store.cloudOwnedIDs,[daytonaConnection!])
    }
    @MainActor func testAgent37SettingsPanelRenders() async throws {
        _ = NSApplication.shared
        let root = URL(fileURLWithPath:#filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let resources = root.appendingPathComponent("Resources")
        var settings = try JSONDecoder().decode(ResearchCore.Settings.self,from:Data(contentsOf:resources.appendingPathComponent("Defaults.json")))
        let runtime = try Workspace.directory(root.path,"tmp/agent37-ui-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at:runtime) }
        settings.workspaceRoot = runtime.path
        let store = AppStore(resourceDirectory:resources,defaults:settings)
        store.selectSandboxProvider(.agent37)
        let view = ConnectionsView(store:store,initialTab:"Daytona").environment(\.colorScheme,.light).background(Color.white)
        let host = NSHostingView(rootView:view);host.frame = NSRect(x:0,y:0,width:710,height:610)
        let window = NSWindow(contentRect:host.frame,styleMask:[.borderless],backing:.buffered,defer:false)
        window.isReleasedWhenClosed = false;window.contentView = host
        host.layoutSubtreeIfNeeded();try await Task.sleep(for:.milliseconds(250));host.displayIfNeeded()
        let image = try XCTUnwrap(host.bitmapImageRepForCachingDisplay(in:host.bounds));host.cacheDisplay(in:host.bounds,to:image)
        let png = try XCTUnwrap(image.representation(using:.png,properties:[:]))
        let output = try Workspace.directory(root.path,"evidence/agent37")
        try png.write(to:output.appendingPathComponent("settings.png"))
    }
    func testKeychainServicesAreIsolatedAcrossProviders() throws {
        let account = "test-provider-" + UUID().uuidString
        defer { for provider in SandboxProvider.allCases { DaytonaKeychain.remove(account,provider:provider) } }
        try DaytonaKeychain.save("daytona-fixture",account:account)
        try DaytonaKeychain.save("agent37-fixture",account:account,provider:.agent37)
        XCTAssertEqual(try DaytonaKeychain.read(account),"daytona-fixture")
        XCTAssertEqual(try DaytonaKeychain.read(account,provider:.agent37),"agent37-fixture")
        DaytonaKeychain.remove(account,provider:.agent37)
        XCTAssertEqual(try DaytonaKeychain.read(account),"daytona-fixture")
    }
}
