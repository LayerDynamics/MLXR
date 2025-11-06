//
//  WebViewController.swift
//  MLXR
//
//  Hosts the WKWebView that displays the React UI.
//  Integrates the JavaScript bridge for frontend ↔ native communication.
//

import Cocoa
import WebKit

class WebViewController: NSViewController {

    // MARK: - Properties

    private var webView: WKWebView!
    private var messageHandler: MessageHandlers?
    private let isDevelopment = ProcessInfo.processInfo.environment["MLXR_DEV_MODE"] == "1"

    // MARK: - Lifecycle

    override func loadView() {
        setupWebView()
        view = webView
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        loadUI()
    }

    // MARK: - Setup

    private func setupWebView() {
        // Configure WKWebView
        let config = WKWebViewConfiguration()

        // Enable developer tools in development mode
        if isDevelopment {
            config.preferences.setValue(true, forKey: "developerExtrasEnabled")
        }

        // Setup user content controller for bridge
        let contentController = WKUserContentController()

        // Inject bridge script
        if let bridgeScript = loadBridgeScript() {
            let userScript = WKUserScript(
                source: bridgeScript,
                injectionTime: .atDocumentStart,
                forMainFrameOnly: true
            )
            contentController.addUserScript(userScript)
        }

        // Setup message handlers
        messageHandler = MessageHandlers()
        contentController.add(messageHandler!, name: "hostBridge")

        config.userContentController = contentController

        // Create WebView
        webView = WKWebView(frame: .zero, configuration: config)
        webView.navigationDelegate = self
        webView.uiDelegate = self

        // Allow back/forward gestures
        webView.allowsBackForwardNavigationGestures = false

        // Background color
        webView.setValue(false, forKey: "drawsBackground")
    }

    private func loadBridgeScript() -> String? {
        guard let scriptURL = Bundle.main.url(forResource: "BridgeInjector", withExtension: "js") else {
            print("WebViewController: BridgeInjector.js not found")
            return nil
        }
        return try? String(contentsOf: scriptURL, encoding: .utf8)
    }

    private func loadUI() {
        if isDevelopment {
            // Development mode - load from Vite dev server
            if let url = URL(string: "http://localhost:5173") {
                print("WebViewController: Loading UI from dev server: \(url)")
                let request = URLRequest(url: url)
                webView.load(request)
            }
        } else {
            // Production mode - load from bundle
            if let indexPath = Bundle.main.path(forResource: "index", ofType: "html", inDirectory: "Resources/ui") {
                let url = URL(fileURLWithPath: indexPath)
                let baseURL = url.deletingLastPathComponent()
                print("WebViewController: Loading UI from bundle: \(url)")
                webView.loadFileURL(url, allowingReadAccessTo: baseURL)
            } else {
                print("WebViewController: ERROR - UI bundle not found at Resources/ui/index.html")
                loadErrorPage()
            }
        }
    }

    private func loadErrorPage() {
        let html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>MLXR - UI Not Found</title>
            <style>
                body {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background: #1a1a1a;
                    color: #fff;
                }
                .error {
                    text-align: center;
                }
                h1 { color: #ff4444; }
            </style>
        </head>
        <body>
            <div class="error">
                <h1>⚠️ UI Not Found</h1>
                <p>The React UI could not be loaded.</p>
                <p>Please build the UI first:</p>
                <pre>cd app/ui && npm run build</pre>
            </div>
        </body>
        </html>
        """
        webView.loadHTMLString(html, baseURL: nil)
    }
}

// MARK: - WKNavigationDelegate

extension WebViewController: WKNavigationDelegate {

    func webView(_ webView: WKWebView, didFinish navigation: WKNavigation!) {
        print("WebViewController: Page loaded successfully")
    }

    func webView(_ webView: WKWebView, didFail navigation: WKNavigation!, withError error: Error) {
        print("WebViewController: Navigation failed: \(error.localizedDescription)")
    }

    func webView(_ webView: WKWebView, didFailProvisionalNavigation navigation: WKNavigation!, withError error: Error) {
        print("WebViewController: Provisional navigation failed: \(error.localizedDescription)")

        if isDevelopment {
            // Show helpful message if dev server isn't running
            let html = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>MLXR - Dev Server Not Running</title>
                <style>
                    body {
                        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        background: #1a1a1a;
                        color: #fff;
                    }
                    .error {
                        text-align: center;
                    }
                    h1 { color: #ff9900; }
                    pre {
                        background: #333;
                        padding: 1rem;
                        border-radius: 4px;
                        display: inline-block;
                    }
                </style>
            </head>
            <body>
                <div class="error">
                    <h1>🔧 Dev Server Not Running</h1>
                    <p>Please start the Vite dev server:</p>
                    <pre>cd app/ui && npm run dev</pre>
                    <p><small>Error: \(error.localizedDescription)</small></p>
                </div>
            </body>
            </html>
            """
            webView.loadHTMLString(html, baseURL: nil)
        }
    }
}

// MARK: - WKUIDelegate

extension WebViewController: WKUIDelegate {

    func webView(_ webView: WKWebView, runJavaScriptAlertPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping () -> Void) {
        let alert = NSAlert()
        alert.messageText = "MLXR"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.runModal()
        completionHandler()
    }

    func webView(_ webView: WKWebView, runJavaScriptConfirmPanelWithMessage message: String,
                 initiatedByFrame frame: WKFrameInfo, completionHandler: @escaping (Bool) -> Void) {
        let alert = NSAlert()
        alert.messageText = "MLXR"
        alert.informativeText = message
        alert.addButton(withTitle: "OK")
        alert.addButton(withTitle: "Cancel")
        let response = alert.runModal()
        completionHandler(response == .alertFirstButtonReturn)
    }
}
