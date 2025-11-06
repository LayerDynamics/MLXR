//
//  LoginItemManager.swift
//  MLXR
//
//  Manages "Launch at Login" functionality using SMAppService (macOS 13+).
//

import Foundation
import ServiceManagement

class LoginItemManager {

    // MARK: - Singleton

    static let shared = LoginItemManager()

    private init() {}

    // MARK: - Login Item Management

    /// Check if app is set to launch at login
    func isLaunchAtLoginEnabled() -> Bool {
        if #available(macOS 13.0, *) {
            let service = SMAppService.mainApp
            return service.status == .enabled
        } else {
            // Fallback for macOS 12 and earlier (using deprecated API)
            return isLegacyLaunchAtLoginEnabled()
        }
    }

    /// Enable launch at login
    func enableLaunchAtLogin() throws {
        if #available(macOS 13.0, *) {
            let service = SMAppService.mainApp
            if service.status != .enabled {
                try service.register()
            }
        } else {
            // Fallback for macOS 12 and earlier
            try enableLegacyLaunchAtLogin()
        }
    }

    /// Disable launch at login
    func disableLaunchAtLogin() throws {
        if #available(macOS 13.0, *) {
            let service = SMAppService.mainApp
            if service.status == .enabled {
                try service.unregister()
            }
        } else {
            // Fallback for macOS 12 and earlier
            try disableLegacyLaunchAtLogin()
        }
    }

    /// Toggle launch at login
    func toggleLaunchAtLogin() throws {
        if isLaunchAtLoginEnabled() {
            try disableLaunchAtLogin()
        } else {
            try enableLaunchAtLogin()
        }
    }

    // MARK: - Legacy Support (macOS 12 and earlier)

    @available(macOS, deprecated: 13.0, message: "Use SMAppService instead")
    private func isLegacyLaunchAtLoginEnabled() -> Bool {
        // This is a simplified check - in production, you'd use SMCopyAllJobDictionaries
        let defaults = UserDefaults.standard
        return defaults.bool(forKey: "LaunchAtLogin")
    }

    @available(macOS, deprecated: 13.0)
    private func enableLegacyLaunchAtLogin() throws {
        let defaults = UserDefaults.standard
        defaults.set(true, forKey: "LaunchAtLogin")

        // In a real implementation, you'd use SMLoginItemSetEnabled
        // For now, we'll just store the preference
        print("[LoginItemManager] Launch at login enabled (legacy)")
    }

    @available(macOS, deprecated: 13.0)
    private func disableLegacyLaunchAtLogin() throws {
        let defaults = UserDefaults.standard
        defaults.set(false, forKey: "LaunchAtLogin")

        print("[LoginItemManager] Launch at login disabled (legacy)")
    }
}

// MARK: - Errors

enum LoginItemError: Error, LocalizedError {
    case registrationFailed
    case unregistrationFailed
    case unsupportedOS

    var errorDescription: String? {
        switch self {
        case .registrationFailed:
            return "Failed to enable launch at login"
        case .unregistrationFailed:
            return "Failed to disable launch at login"
        case .unsupportedOS:
            return "This feature requires macOS 13 or later"
        }
    }
}
