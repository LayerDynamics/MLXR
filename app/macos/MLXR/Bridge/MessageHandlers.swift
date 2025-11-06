//
//  MessageHandlers.swift
//  MLXR
//
//  Handles messages from JavaScript bridge and dispatches to appropriate handlers.
//

import Foundation
import WebKit
import AppKit

class MessageHandlers: NSObject, WKScriptMessageHandler, HostBridge {

    // MARK: - Properties

    private let socketClient: UnixSocketClient
    private let fileManager = FileManager.default

    // MARK: - Initialization

    override init() {
        self.socketClient = UnixSocketClient()
        super.init()
    }

    // MARK: - WKScriptMessageHandler

    func userContentController(_ userContentController: WKUserContentController, didReceive message: WKScriptMessage) {
        guard let body = message.body as? [String: Any] else {
            print("[Bridge] Invalid message format")
            return
        }

        Task {
            await handleMessage(body, webView: message.webView)
        }
    }

    // MARK: - Message Handling

    private func handleMessage(_ body: [String: Any], webView: WKWebView?) async {
        guard let id = body["id"] as? Int,
              let method = body["method"] as? String else {
            print("[Bridge] Missing id or method in message")
            return
        }

        let params = body["params"] as? [String: Any]

        print("[Bridge] Received message: \(method) (id=\(id))")

        do {
            let result = try await dispatchMethod(method, params: params)
            await sendResponse(id: id, result: result, error: nil, to: webView)
        } catch {
            print("[Bridge] Error handling \(method): \(error)")
            await sendResponse(id: id, result: nil, error: error.localizedDescription, to: webView)
        }
    }

    private func dispatchMethod(_ method: String, params: [String: Any]?) async throws -> Any? {
        switch method {
        case "request":
            guard let path = params?["path"] as? String else {
                throw BridgeError.invalidParams
            }
            let init = params?["init"] as? [String: Any]
            let response = try await request(path: path, init: init)
            return try? JSONEncoder().encode(response)

        case "openPathDialog":
            guard let type = params?["type"] as? String else {
                throw BridgeError.invalidParams
            }
            return try await openPathDialog(type: type)

        case "readConfig":
            return try await readConfig()

        case "writeConfig":
            guard let yaml = params?["yaml"] as? String else {
                throw BridgeError.invalidParams
            }
            try await writeConfig(yaml: yaml)
            return nil

        case "startDaemon":
            try await startDaemon()
            return nil

        case "stopDaemon":
            try await stopDaemon()
            return nil

        case "getVersion":
            return try await getVersion()

        default:
            throw BridgeError.invalidMethod(method)
        }
    }

    private func sendResponse(id: Int, result: Any?, error: String?, to webView: WKWebView?) async {
        await MainActor.run {
            let resultJSON: String
            if let result = result {
                if let data = try? JSONSerialization.data(withJSONObject: result),
                   let string = String(data: data, encoding: .utf8) {
                    resultJSON = string
                } else if let string = result as? String {
                    resultJSON = "\"\(string.replacingOccurrences(of: "\"", with: "\\\""))\""
                } else {
                    resultJSON = "null"
                }
            } else {
                resultJSON = "null"
            }

            let errorJSON = error != nil ? "\"\(error!.replacingOccurrences(of: "\"", with: "\\\""))\"" : "null"

            let script = "window.handleBridgeResponse(\(id), \(errorJSON), \(resultJSON));"
            webView?.evaluateJavaScript(script) { _, error in
                if let error = error {
                    print("[Bridge] Error sending response: \(error)")
                }
            }
        }
    }

    // MARK: - HostBridge Implementation

    func request(path: String, init: [String: Any]?) async throws -> BridgeResponse {
        // Parse request options
        let method = init?["method"] as? String ?? "GET"
        let headers = init?["headers"] as? [String: String] ?? [:]
        let body = init?["body"] as? String

        // Make request to daemon via Unix Socket
        do {
            let response = try await socketClient.request(
                method: method,
                path: path,
                headers: headers,
                body: body
            )
            return response
        } catch {
            throw BridgeError.requestFailed(error.localizedDescription)
        }
    }

    func openPathDialog(type: String) async throws -> String? {
        return await MainActor.run {
            let panel = NSOpenPanel()
            panel.canChooseFiles = false
            panel.canChooseDirectories = true
            panel.allowsMultipleSelection = false
            panel.canCreateDirectories = true

            switch type {
            case "models":
                panel.title = "Select Models Directory"
                panel.message = "Choose where to store model files"
            case "cache":
                panel.title = "Select Cache Directory"
                panel.message = "Choose where to store cache files"
            default:
                panel.title = "Select Directory"
            }

            let response = panel.runModal()
            if response == .OK, let url = panel.url {
                return url.path
            }
            return nil
        }
    }

    func readConfig() async throws -> String {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let configPath = appSupport.appendingPathComponent("MLXRunner/server.yaml")

        guard fileManager.fileExists(atPath: configPath.path) else {
            // Return default config if file doesn't exist
            return """
            # MLXR Server Configuration
            max_batch_tokens: 2048
            target_latency_ms: 80
            enable_speculative: true
            draft_model: ""
            kv_persistence: true
            """
        }

        return try String(contentsOf: configPath, encoding: .utf8)
    }

    func writeConfig(yaml: String) async throws {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let configPath = appSupport.appendingPathComponent("MLXRunner/server.yaml")

        try yaml.write(to: configPath, atomically: true, encoding: .utf8)
    }

    func startDaemon() async throws {
        try await DaemonManager.shared.startDaemon()
    }

    func stopDaemon() async throws {
        try await DaemonManager.shared.stopDaemon()
    }

    func getVersion() async throws -> [String: String] {
        let appVersion = Bundle.main.infoDictionary?["CFBundleShortVersionString"] as? String ?? "1.0.0"

        // Try to get daemon version
        var daemonVersion = "Unknown"
        if let isRunning = try? await DaemonManager.shared.isDaemonRunning(), isRunning {
            // TODO: Fetch daemon version from /health or /version endpoint
            daemonVersion = "1.0.0"
        }

        return [
            "app": appVersion,
            "daemon": daemonVersion
        ]
    }
}
