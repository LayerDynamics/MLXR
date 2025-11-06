//
//  AppDelegate.swift
//  MLXR
//
//  Main application delegate for MLXR macOS app.
//  Manages app lifecycle, tray icon, and main window.
//

import Cocoa
import ServiceManagement

@main
class AppDelegate: NSObject, NSApplicationDelegate {

    // MARK: - Properties

    private var mainWindowController: MainWindowController?
    private var trayController: TrayController?
    private var daemonManager: DaemonManager?

    // MARK: - App Lifecycle

    func applicationDidFinishLaunching(_ notification: Notification) {
        print("MLXR: Application launched")

        // Initialize daemon manager
        daemonManager = DaemonManager.shared

        // Setup notifications
        setupNotifications()

        // Setup update manager
        setupUpdateManager()

        // Setup tray icon
        setupTray()

        // Check if this is first run
        if isFirstRun() {
            performFirstRunSetup()
        }

        // Start daemon if not running
        Task {
            await startDaemonIfNeeded()
        }

        // Show main window
        showMainWindow()
    }

    func applicationWillTerminate(_ notification: Notification) {
        print("MLXR: Application terminating")
        // Don't stop daemon - it should keep running
    }

    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool {
        // Keep app running in tray even when window is closed
        return false
    }

    func applicationSupportsSecureRestorableState(_ app: NSApplication) -> Bool {
        return true
    }

    // MARK: - Window Management

    func showMainWindow() {
        if mainWindowController == nil {
            mainWindowController = MainWindowController()
        }
        mainWindowController?.showWindow(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func hideMainWindow() {
        mainWindowController?.close()
    }

    // MARK: - Tray Management

    private func setupTray() {
        trayController = TrayController(appDelegate: self)
    }

    // MARK: - Notifications

    private func setupNotifications() {
        let notificationManager = NotificationManager.shared

        // Setup notification categories with actions
        notificationManager.setupNotificationCategories()

        // Request authorization
        Task {
            let authorized = await notificationManager.requestAuthorization()
            print("MLXR: Notification authorization: \(authorized)")
        }

        // Observe update check requests
        NotificationCenter.default.addObserver(
            self,
            selector: #selector(handleCheckForUpdates),
            name: .checkForUpdates,
            object: nil
        )
    }

    // MARK: - Updates

    private func setupUpdateManager() {
        let updateManager = UpdateManager.shared

        // Enable automatic update checks
        updateManager.setAutomaticUpdateChecksEnabled(true)
        updateManager.setAutomaticDownloadEnabled(false) // User confirmation required

        // Start periodic checks
        updateManager.startAutomaticUpdateChecks()

        print("MLXR: Update manager initialized - version \(updateManager.getCurrentVersion())")
    }

    @objc private func handleCheckForUpdates() {
        UpdateManager.shared.checkForUpdates()
    }

    @objc func checkForUpdates() {
        UpdateManager.shared.checkForUpdates()
    }

    // MARK: - First Run Setup

    private func isFirstRun() -> Bool {
        let key = "MLXRHasLaunchedBefore"
        let hasLaunched = UserDefaults.standard.bool(forKey: key)
        if !hasLaunched {
            UserDefaults.standard.set(true, forKey: key)
            return true
        }
        return false
    }

    private func performFirstRunSetup() {
        print("MLXR: Performing first run setup")

        // Create application support directories
        let fileManager = FileManager.default
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let mlxrDir = appSupport.appendingPathComponent("MLXRunner")

        let directories = [
            mlxrDir,
            mlxrDir.appendingPathComponent("models"),
            mlxrDir.appendingPathComponent("cache"),
            mlxrDir.appendingPathComponent("bin"),
            mlxrDir.appendingPathComponent("run"),
        ]

        for dir in directories {
            try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        }

        // Copy default config
        if let defaultConfig = Bundle.main.url(forResource: "server", withExtension: "yaml") {
            let destConfig = mlxrDir.appendingPathComponent("server.yaml")
            if !fileManager.fileExists(atPath: destConfig.path) {
                try? fileManager.copyItem(at: defaultConfig, to: destConfig)
            }
        }

        // Install daemon binary
        if let daemonSource = Bundle.main.url(forResource: "mlxrunnerd", withExtension: nil) {
            let daemonDest = mlxrDir.appendingPathComponent("bin/mlxrunnerd")
            if !fileManager.fileExists(atPath: daemonDest.path) {
                try? fileManager.copyItem(at: daemonSource, to: daemonDest)
                // Make executable
                try? fileManager.setAttributes([.posixPermissions: 0o755], ofItemAtPath: daemonDest.path)
            }
        }

        print("MLXR: First run setup complete")
    }

    // MARK: - Daemon Management

    private func startDaemonIfNeeded() async {
        guard let manager = daemonManager else { return }

        do {
            let isRunning = try await manager.isDaemonRunning()
            if !isRunning {
                print("MLXR: Starting daemon...")
                try await manager.startDaemon()
                NotificationManager.shared.notifyDaemonStarted()
            } else {
                print("MLXR: Daemon already running")
            }
        } catch {
            print("MLXR: Error starting daemon: \(error)")
            NotificationManager.shared.notifyError(
                title: "Failed to Start Daemon",
                message: error.localizedDescription,
                critical: true
            )
        }
    }

    // MARK: - Actions

    @objc func startDaemon() {
        Task {
            do {
                try await daemonManager?.startDaemon()
                NotificationManager.shared.notifyDaemonStarted()
            } catch {
                NotificationManager.shared.notifyError(
                    title: "Failed to Start Daemon",
                    message: error.localizedDescription,
                    critical: true
                )
            }
        }
    }

    @objc func stopDaemon() {
        Task {
            do {
                try await daemonManager?.stopDaemon()
                NotificationManager.shared.notifyDaemonStopped()
            } catch {
                NotificationManager.shared.notifyError(
                    title: "Failed to Stop Daemon",
                    message: error.localizedDescription,
                    critical: true
                )
            }
        }
    }

    @objc func restartDaemon() {
        Task {
            do {
                try await daemonManager?.restartDaemon()
                NotificationManager.shared.notifyDaemonStarted()
            } catch {
                NotificationManager.shared.notifyError(
                    title: "Failed to Restart Daemon",
                    message: error.localizedDescription,
                    critical: true
                )
            }
        }
    }

    @objc func quitApp() {
        NSApp.terminate(nil)
    }
}
