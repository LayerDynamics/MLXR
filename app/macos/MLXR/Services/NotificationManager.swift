//
//  NotificationManager.swift
//  MLXR
//
//  Manages user notifications for daemon status, errors, and updates.
//

import Foundation
import UserNotifications
import AppKit

class NotificationManager: NSObject {

    // MARK: - Properties

    private let notificationCenter = UNUserNotificationCenter.current()
    private var isAuthorized = false

    // Notification identifiers
    private enum NotificationID {
        static let daemonStarted = "com.mlxr.notification.daemon.started"
        static let daemonStopped = "com.mlxr.notification.daemon.stopped"
        static let daemonCrashed = "com.mlxr.notification.daemon.crashed"
        static let modelLoaded = "com.mlxr.notification.model.loaded"
        static let modelUnloaded = "com.mlxr.notification.model.unloaded"
        static let error = "com.mlxr.notification.error"
        static let updateAvailable = "com.mlxr.notification.update.available"
    }

    // MARK: - Singleton

    static let shared = NotificationManager()

    private override init() {
        super.init()
        notificationCenter.delegate = self
    }

    // MARK: - Authorization

    /// Request notification permissions
    func requestAuthorization() async -> Bool {
        do {
            isAuthorized = try await notificationCenter.requestAuthorization(options: [.alert, .sound, .badge])
            return isAuthorized
        } catch {
            print("Failed to request notification authorization: \(error)")
            return false
        }
    }

    /// Check current authorization status
    func checkAuthorizationStatus() async -> Bool {
        let settings = await notificationCenter.notificationSettings()
        isAuthorized = settings.authorizationStatus == .authorized
        return isAuthorized
    }

    // MARK: - Daemon Notifications

    /// Notify that daemon started successfully
    func notifyDaemonStarted() {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "MLXR Daemon Started"
        content.body = "The inference engine is now running and ready to serve requests."
        content.sound = .default

        sendNotification(identifier: NotificationID.daemonStarted, content: content)
    }

    /// Notify that daemon stopped
    func notifyDaemonStopped(reason: String? = nil) {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "MLXR Daemon Stopped"
        content.body = reason ?? "The inference engine has been stopped."
        content.sound = .default

        sendNotification(identifier: NotificationID.daemonStopped, content: content)
    }

    /// Notify that daemon crashed
    func notifyDaemonCrashed() {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "MLXR Daemon Crashed"
        content.body = "The inference engine encountered an error and stopped. Click to restart."
        content.sound = .defaultCritical
        content.categoryIdentifier = "DAEMON_CRASHED"

        sendNotification(identifier: NotificationID.daemonCrashed, content: content)
    }

    // MARK: - Model Notifications

    /// Notify that a model was loaded
    func notifyModelLoaded(modelName: String) {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "Model Loaded"
        content.body = "\(modelName) is now available for inference."
        content.sound = .default

        sendNotification(identifier: NotificationID.modelLoaded, content: content)
    }

    /// Notify that a model was unloaded
    func notifyModelUnloaded(modelName: String) {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "Model Unloaded"
        content.body = "\(modelName) has been unloaded from memory."

        sendNotification(identifier: NotificationID.modelUnloaded, content: content)
    }

    // MARK: - Error Notifications

    /// Notify about an error
    func notifyError(title: String, message: String, critical: Bool = false) {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = title
        content.body = message
        content.sound = critical ? .defaultCritical : .default

        sendNotification(identifier: NotificationID.error, content: content)
    }

    // MARK: - Update Notifications

    /// Notify that an update is available
    func notifyUpdateAvailable(version: String, releaseNotes: String? = nil) {
        guard isAuthorized else { return }

        let content = UNMutableNotificationContent()
        content.title = "Update Available"
        content.body = "MLXR \(version) is available. Click to update."
        content.sound = .default
        content.categoryIdentifier = "UPDATE_AVAILABLE"

        if let releaseNotes = releaseNotes {
            content.subtitle = releaseNotes
        }

        sendNotification(identifier: NotificationID.updateAvailable, content: content)
    }

    // MARK: - Helper Methods

    private func sendNotification(identifier: String, content: UNMutableNotificationContent) {
        let request = UNNotificationRequest(
            identifier: identifier,
            content: content,
            trigger: nil // Deliver immediately
        )

        notificationCenter.add(request) { error in
            if let error = error {
                print("Failed to send notification: \(error)")
            }
        }
    }

    /// Remove all pending notifications
    func removeAllNotifications() {
        notificationCenter.removeAllPendingNotificationRequests()
    }

    /// Remove specific notification by identifier
    func removeNotification(identifier: String) {
        notificationCenter.removePendingNotificationRequests(withIdentifiers: [identifier])
    }

    /// Setup notification categories with actions
    func setupNotificationCategories() {
        // Daemon crashed category with restart action
        let restartAction = UNNotificationAction(
            identifier: "RESTART_DAEMON",
            title: "Restart",
            options: [.foreground]
        )

        let daemonCrashedCategory = UNNotificationCategory(
            identifier: "DAEMON_CRASHED",
            actions: [restartAction],
            intentIdentifiers: [],
            options: []
        )

        // Update available category with update action
        let updateAction = UNNotificationAction(
            identifier: "INSTALL_UPDATE",
            title: "Install Update",
            options: [.foreground]
        )

        let updateAvailableCategory = UNNotificationCategory(
            identifier: "UPDATE_AVAILABLE",
            actions: [updateAction],
            intentIdentifiers: [],
            options: []
        )

        notificationCenter.setNotificationCategories([
            daemonCrashedCategory,
            updateAvailableCategory
        ])
    }
}

// MARK: - UNUserNotificationCenterDelegate

extension NotificationManager: UNUserNotificationCenterDelegate {

    /// Handle notification when app is in foreground
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        willPresent notification: UNNotification,
        withCompletionHandler completionHandler: @escaping (UNNotificationPresentationOptions) -> Void
    ) {
        // Show notification even when app is in foreground
        completionHandler([.banner, .sound, .badge])
    }

    /// Handle notification action
    func userNotificationCenter(
        _ center: UNUserNotificationCenter,
        didReceive response: UNNotificationResponse,
        withCompletionHandler completionHandler: @escaping () -> Void
    ) {
        let actionIdentifier = response.actionIdentifier

        switch actionIdentifier {
        case "RESTART_DAEMON":
            // Restart daemon
            Task {
                do {
                    try await DaemonManager.shared.startDaemon()
                } catch {
                    notifyError(title: "Failed to Restart", message: error.localizedDescription)
                }
            }

        case "INSTALL_UPDATE":
            // Open main window to install update
            DispatchQueue.main.async {
                NSApp.activate(ignoringOtherApps: true)
                // Trigger update check
                NotificationCenter.default.post(name: .checkForUpdates, object: nil)
            }

        case UNNotificationDefaultActionIdentifier:
            // User tapped notification - open main window
            DispatchQueue.main.async {
                NSApp.activate(ignoringOtherApps: true)
            }

        default:
            break
        }

        completionHandler()
    }
}

// MARK: - Notification Names

extension Notification.Name {
    static let checkForUpdates = Notification.Name("com.mlxr.checkForUpdates")
}
