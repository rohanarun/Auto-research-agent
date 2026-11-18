import AppKit
import SwiftUI

/// App colors that follow the system appearance. Light values are the original palette.
enum Theme {
    static let ink = adaptive(light: (0.13, 0.14, 0.15), dark: (0.90, 0.91, 0.92))
    static let muted = adaptive(light: (0.46, 0.47, 0.48), dark: (0.60, 0.61, 0.63))
    static let sidebar = adaptive(light: (0.955, 0.954, 0.946), dark: (0.125, 0.128, 0.135))
    static let accent = adaptive(light: (0.30, 0.34, 0.78), dark: (0.55, 0.60, 0.98))
    /// Main content and composer background (white in light mode).
    static let surface = adaptive(light: (1, 1, 1), dark: (0.085, 0.088, 0.095))

    private static func adaptive(light: (CGFloat, CGFloat, CGFloat), dark: (CGFloat, CGFloat, CGFloat)) -> Color {
        Color(nsColor: NSColor(name: nil) { appearance in
            let c = appearance.bestMatch(from: [.darkAqua, .aqua]) == .darkAqua ? dark : light
            return NSColor(srgbRed: c.0, green: c.1, blue: c.2, alpha: 1)
        })
    }
}
