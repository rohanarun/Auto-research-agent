import CoreGraphics
import Foundation
import AppKit

let windows = CGWindowListCopyWindowInfo([.optionAll], kCGNullWindowID) as? [[String: Any]] ?? []
let appPID = NSRunningApplication.runningApplications(withBundleIdentifier: "com.getsupers.autoresearch").first?.processIdentifier
for window in windows where window[kCGWindowOwnerPID as String] as? Int32 == appPID && window[kCGWindowName as String] as? String == "Auto Research" && window[kCGWindowLayer as String] as? Int == 0 {
    if let number = window[kCGWindowNumber as String] as? Int { print(number); break }
}
