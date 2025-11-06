//
//  TrayController.swift
//  MLXR
//
//  Manages the system tray (menu bar) icon and menu.
//

import Cocoa
import SwiftUI

class TrayController: NSObject {

    // MARK: - Properties

    private weak var appDelegate: AppDelegate?
    private var statusItem: NSStatusItem?
    private var popover: NSPopover?
    private var menu: NSMenu?

    // MARK: - Initialization

    init(appDelegate: AppDelegate) {
        self.appDelegate = appDelegate
        super.init()
        setupStatusItem()
        setupMenu()
        setupPopover()
    }

    // MARK: - Setup

    private func setupStatusItem() {
        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.squareLength)

        if let button = statusItem?.button {
            // Use template image for tray icon (automatically adapts to dark/light mode)
            if let image = NSImage(named: "TrayIcon") {
                image.isTemplate = true
                button.image = image
            } else {
                // Fallback to text if image not found
                button.title = "MLX"
            }

            button.action = #selector(togglePopover)
            button.sendAction(on: [.leftMouseUp, .rightMouseUp])
        }
    }

    private func setupMenu() {
        menu = NSMenu()

        // Window
        menu?.addItem(NSMenuItem(title: "Show MLXR", action: #selector(showMainWindow), keyEquivalent: ""))

        menu?.addItem(NSMenuItem.separator())

        // Daemon Controls
        menu?.addItem(NSMenuItem(title: "Start Daemon", action: #selector(startDaemon), keyEquivalent: ""))
        menu?.addItem(NSMenuItem(title: "Stop Daemon", action: #selector(stopDaemon), keyEquivalent: ""))
        menu?.addItem(NSMenuItem(title: "Restart Daemon", action: #selector(restartDaemon), keyEquivalent: ""))

        menu?.addItem(NSMenuItem.separator())

        // Settings & About
        menu?.addItem(NSMenuItem(title: "Preferences...", action: #selector(showPreferences), keyEquivalent: ","))
        menu?.addItem(NSMenuItem(title: "About MLXR", action: #selector(showAbout), keyEquivalent: ""))

        menu?.addItem(NSMenuItem.separator())

        // Check for Updates
        menu?.addItem(NSMenuItem(title: "Check for Updates...", action: #selector(checkForUpdates), keyEquivalent: ""))

        menu?.addItem(NSMenuItem.separator())

        // Quit
        menu?.addItem(NSMenuItem(title: "Quit MLXR", action: #selector(quitApp), keyEquivalent: "q"))

        // Set targets
        for item in menu?.items ?? [] {
            item.target = self
        }
    }

    private func setupPopover() {
        popover = NSPopover()
        popover?.contentSize = NSSize(width: 300, height: 200)
        popover?.behavior = .transient
        popover?.contentViewController = NSHostingController(rootView: TrayPopoverView())
    }

    // MARK: - Actions

    @objc private func togglePopover() {
        guard let button = statusItem?.button else { return }

        let event = NSApp.currentEvent
        if event?.type == .rightMouseUp {
            // Right click - show menu
            showMenu()
        } else {
            // Left click - toggle popover
            if let popover = popover {
                if popover.isShown {
                    popover.performClose(nil)
                } else {
                    popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
                }
            }
        }
    }

    @objc private func showMenu() {
        guard let button = statusItem?.button, let menu = menu else { return }
        statusItem?.menu = menu
        button.performClick(nil)
        // Reset menu to nil so next click shows popover
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.1) {
            self.statusItem?.menu = nil
        }
    }

    @objc private func showMainWindow() {
        appDelegate?.showMainWindow()
    }

    @objc private func startDaemon() {
        appDelegate?.startDaemon()
    }

    @objc private func stopDaemon() {
        appDelegate?.stopDaemon()
    }

    @objc private func restartDaemon() {
        appDelegate?.restartDaemon()
    }

    @objc private func showPreferences() {
        appDelegate?.showMainWindow()
        // Navigate to settings tab via bridge
        // This will be handled by the WebView
    }

    @objc private func showAbout() {
        let alert = NSAlert()
        alert.messageText = "MLXR"
        alert.informativeText = "High-performance LLM inference engine for Apple Silicon\n\nVersion 1.0.0\nCopyright © 2025"
        alert.alertStyle = .informational
        alert.addButton(withTitle: "OK")
        alert.runModal()
    }

    @objc private func checkForUpdates() {
        appDelegate?.checkForUpdates()
    }

    @objc private func quitApp() {
        appDelegate?.quitApp()
    }
}
