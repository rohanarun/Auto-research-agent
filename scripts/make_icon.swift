import AppKit

let destination = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(at: destination, withIntermediateDirectories: true)
for size in [16, 32, 64, 128, 256, 512, 1024] {
    let image = NSImage(size: NSSize(width: size, height: size))
    image.lockFocus()
    let context = NSGraphicsContext.current!.cgContext
    context.scaleBy(x: CGFloat(size) / 1024, y: CGFloat(size) / 1024)
    let background = NSBezierPath(roundedRect: NSRect(x: 40, y: 40, width: 944, height: 944), xRadius: 220, yRadius: 220)
    NSColor(calibratedRed: 0.13, green: 0.15, blue: 0.21, alpha: 1).setFill(); background.fill()
    let gradient = NSGradient(starting: NSColor(calibratedRed: 0.37, green: 0.40, blue: 0.75, alpha: 1), ending: NSColor(calibratedRed: 0.10, green: 0.12, blue: 0.19, alpha: 1))!
    gradient.draw(in: background, angle: -60)
    let points = [NSPoint(x: 512, y: 760), NSPoint(x: 285, y: 360), NSPoint(x: 745, y: 360)]
    NSColor.white.withAlphaComponent(0.82).setStroke()
    let triangle = NSBezierPath(); triangle.move(to: points[0]); triangle.line(to: points[1]); triangle.line(to: points[2]); triangle.close(); triangle.lineWidth = 29; triangle.lineJoinStyle = .round; triangle.stroke()
    for point in points {
        NSColor(calibratedRed: 0.84, green: 0.86, blue: 1, alpha: 1).setFill()
        NSBezierPath(ovalIn: NSRect(x: point.x - 58, y: point.y - 58, width: 116, height: 116)).fill()
    }
    NSColor.white.setFill(); NSBezierPath(ovalIn: NSRect(x: 477, y: 477, width: 70, height: 70)).fill()
    image.unlockFocus()
    let rep = NSBitmapImageRep(data: image.tiffRepresentation!)!
    let png = rep.representation(using: .png, properties: [:])!
    if size <= 512 { try png.write(to: destination.appendingPathComponent("icon_\(size)x\(size).png")) }
    if size >= 32 { try png.write(to: destination.appendingPathComponent("icon_\(size / 2)x\(size / 2)@2x.png")) }
}
