//
//  KeychainManager.swift
//  MLXR
//
//  Manages secure storage of auth tokens in macOS Keychain.
//

import Foundation
import Security

class KeychainManager {

    // MARK: - Properties

    private let service = "com.mlxr.app"
    private let account = "daemon-auth-token"

    // MARK: - Singleton

    static let shared = KeychainManager()

    private init() {}

    // MARK: - Token Management

    /// Get auth token from keychain
    func getAuthToken() -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne
        ]

        var result: AnyObject?
        let status = SecItemCopyMatching(query as CFDictionary, &result)

        guard status == errSecSuccess,
              let data = result as? Data,
              let token = String(data: data, encoding: .utf8) else {
            return nil
        }

        return token
    }

    /// Save auth token to keychain
    func saveAuthToken(_ token: String) -> Bool {
        // Delete existing token first
        _ = deleteAuthToken()

        guard let data = token.data(using: .utf8) else {
            return false
        }

        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account,
            kSecValueData as String: data,
            kSecAttrAccessible as String: kSecAttrAccessibleAfterFirstUnlock
        ]

        let status = SecItemAdd(query as CFDictionary, nil)
        return status == errSecSuccess
    }

    /// Delete auth token from keychain
    func deleteAuthToken() -> Bool {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: account
        ]

        let status = SecItemDelete(query as CFDictionary)
        return status == errSecSuccess || status == errSecItemNotFound
    }

    /// Generate new auth token
    func generateAuthToken() -> String {
        let letters = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return String((0..<32).map { _ in letters.randomElement()! })
    }

    /// Rotate auth token (generate new and save)
    func rotateAuthToken() -> String? {
        let newToken = generateAuthToken()
        if saveAuthToken(newToken) {
            return newToken
        }
        return nil
    }
}
