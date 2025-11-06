//
//  MainWindowController.swift
//  MLXR
//
//  Manages the main application window that hosts the WebView.
//

import Cocoa

class MainWindowController: NSWindowController {

    // MARK: - Properties

    private var webViewController: WebViewController?

    // MARK: - Initialization

    convenience init() {
        // Create window
        let window = NSWindow(
            contentRect: NSRect(x: 0, y: 0, width: 1200, height: 800),
            styleMask: [.titled, .closable, .miniaturizable, .resizable, .fullSizeContentView],
            backing: .buffered,
            defer: false
        )

        self.init(window: window)

        setupWindow()
        setupWebView()
    }

    // MARK: - Setup

    private func setupWindow() {
        guard let window = window else { return }

        window.title = "MLXR"
        window.titlebarAppearsTransparent = false
        window.titleVisibility = .visible
        window.center()
        window.setFrameAutosaveName("MLXRMainWindow")
        window.isReleasedWhenClosed = false

        // Set minimum size
        window.minSize = NSSize(width: 800, height: 600)

        // Modern macOS window appearance
        window.backgroundColor = NSColor.windowBackgroundColor

        // Allow tabbing
        window.tabbingMode = .preferred
    }

    private func setupWebView() {
        webViewController = WebViewController()

        if let webView = webViewController?.view {
            window?.contentView = webView
        }
    }

    // MARK: - Window Management

    override func showWindow(_ sender: Any?) {
        super.showWindow(sender)
        window?.makeKeyAndOrderFront(sender)
        NSApp.activate(ignoringOtherApps: true)
    }
}
