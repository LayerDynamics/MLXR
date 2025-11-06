//
//  UpdateManager.swift
//  MLXR
//
//  Manages application updates using Sparkle framework.
//

import Foundation

#if canImport(Sparkle)
import Sparkle
#endif

class UpdateManager: NSObject {

    // MARK: - Properties

    #if canImport(Sparkle)
    private var updaterController: SPUStandardUpdaterController?
    #endif

    private var updateCheckTimer: Timer?
    private let checkInterval: TimeInterval = 3600 // 1 hour

    // Update state
    private(set) var isUpdateAvailable = false
    private(set) var latestVersion: String?
    private(set) var releaseNotes: String?

    // MARK: - Singleton

    static let shared = UpdateManager()

    private override init() {
        super.init()
        setupUpdater()
    }

    // MARK: - Setup

    private func setupUpdater() {
        #if canImport(Sparkle)
        // Initialize Sparkle updater controller
        updaterController = SPUStandardUpdaterController(
            startingUpdater: true,
            updaterDelegate: self,
            userDriverDelegate: self
        )

        print("UpdateManager: Sparkle initialized")
        #else
        print("UpdateManager: Sparkle not available - auto-updates disabled")
        #endif
    }

    // MARK: - Manual Update Check

    /// Manually check for updates
    func checkForUpdates() {
        #if canImport(Sparkle)
        updaterController?.checkForUpdates(nil)
        #else
        print("UpdateManager: Cannot check for updates - Sparkle not integrated")
        NotificationManager.shared.notifyError(
            title: "Updates Unavailable",
            message: "Auto-update functionality is not available in this build."
        )
        #endif
    }

    /// Check for updates silently (no UI)
    func checkForUpdatesInBackground() {
        #if canImport(Sparkle)
        updaterController?.updater.checkForUpdatesInBackground()
        #else
        print("UpdateManager: Background update check skipped - Sparkle not integrated")
        #endif
    }

    // MARK: - Automatic Update Checks

    /// Start automatic update checks
    func startAutomaticUpdateChecks() {
        guard updateCheckTimer == nil else { return }

        // Check immediately
        checkForUpdatesInBackground()

        // Schedule periodic checks
        updateCheckTimer = Timer.scheduledTimer(
            withTimeInterval: checkInterval,
            repeats: true
        ) { [weak self] _ in
            self?.checkForUpdatesInBackground()
        }

        print("UpdateManager: Automatic update checks started (interval: \(checkInterval)s)")
    }

    /// Stop automatic update checks
    func stopAutomaticUpdateChecks() {
        updateCheckTimer?.invalidate()
        updateCheckTimer = nil
        print("UpdateManager: Automatic update checks stopped")
    }

    // MARK: - Update Control

    /// Install available update
    func installUpdate() {
        #if canImport(Sparkle)
        // This will trigger the update process
        updaterController?.checkForUpdates(nil)
        #else
        print("UpdateManager: Cannot install update - Sparkle not integrated")
        #endif
    }

    // MARK: - Configuration

    /// Enable/disable automatic update checks
    func setAutomaticUpdateChecksEnabled(_ enabled: Bool) {
        #if canImport(Sparkle)
        updaterController?.updater.automaticallyChecksForUpdates = enabled

        if enabled {
            startAutomaticUpdateChecks()
        } else {
            stopAutomaticUpdateChecks()
        }
        #endif
    }

    /// Enable/disable automatic download
    func setAutomaticDownloadEnabled(_ enabled: Bool) {
        #if canImport(Sparkle)
        updaterController?.updater.automaticallyDownloadsUpdates = enabled
        #endif
    }

    // MARK: - Version Info

    /// Get current app version
    func getCurrentVersion() -> String {
        let version = Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "1.0.0"
        let build = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "1"
        return "\(version) (\(build))"
    }

    /// Get appcast URL
    func getAppcastURL() -> String? {
        return Bundle.main.object(forInfoDictionaryKey: "SUFeedURL") as? String
    }
}

// MARK: - Sparkle Updater Delegate

#if canImport(Sparkle)
extension UpdateManager: SPUUpdaterDelegate {

    /// Called when update is found
    func updater(_ updater: SPUUpdater, didFindValidUpdate item: SUAppcastItem) {
        isUpdateAvailable = true
        latestVersion = item.displayVersionString
        releaseNotes = item.itemDescription

        print("UpdateManager: Update found - version \(item.displayVersionString)")

        // Notify user
        NotificationManager.shared.notifyUpdateAvailable(
            version: item.displayVersionString,
            releaseNotes: item.itemDescription
        )
    }

    /// Called when no update is found
    func updaterDidNotFindUpdate(_ updater: SPUUpdater) {
        isUpdateAvailable = false
        latestVersion = nil
        releaseNotes = nil

        print("UpdateManager: No update available")
    }

    /// Called when update check fails
    func updater(_ updater: SPUUpdater, didFailToDownloadUpdate item: SUAppcastItem, error: Error) {
        print("UpdateManager: Failed to download update - \(error.localizedDescription)")

        NotificationManager.shared.notifyError(
            title: "Update Failed",
            message: "Failed to download update: \(error.localizedDescription)"
        )
    }
}

// MARK: - Sparkle User Driver Delegate

extension UpdateManager: SPUStandardUserDriverDelegate {

    /// Customize update alert appearance
    func standardUserDriverWillShowUpdate(_ update: SUAppcastItem, reply: @escaping (SPUUpdateAlertChoice) -> Void) {
        // Default behavior - show standard update alert
        // Can be customized here if needed
        print("UpdateManager: Showing update alert for version \(update.displayVersionString)")
    }

    /// Called before installing update
    func standardUserDriverWillInstallUpdate() {
        print("UpdateManager: Installing update...")
    }

    /// Called after update is installed (before relaunch)
    func standardUserDriverDidFinishUpdateCycle() {
        print("UpdateManager: Update cycle finished")
    }
}
#endif

// MARK: - Update State

struct UpdateInfo {
    let version: String
    let releaseNotes: String?
    let releaseDate: Date?
    let downloadURL: URL?
}
