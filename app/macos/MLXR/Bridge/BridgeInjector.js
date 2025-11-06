//
//  BridgeInjector.js
//  MLXR
//
//  JavaScript bridge injected into WebView to enable React ↔ Swift communication.
//  Creates window.__HOST__ interface matching app/ui/src/types/bridge.ts
//

(function() {
    'use strict';

    // Check if webkit message handlers are available
    if (!window.webkit || !window.webkit.messageHandlers || !window.webkit.messageHandlers.hostBridge) {
        console.warn('[Bridge] WebKit message handlers not available');
        return;
    }

    console.log('[Bridge] Initializing MLXR Host Bridge');

    // Message ID counter for tracking requests/responses
    let messageId = 0;
    const pendingMessages = new Map();

    // Send message to native side
    function sendMessage(method, params) {
        return new Promise((resolve, reject) => {
            const id = ++messageId;
            const message = { id, method, params };

            // Store promise handlers
            pendingMessages.set(id, { resolve, reject });

            // Send to native
            window.webkit.messageHandlers.hostBridge.postMessage(message);

            // Timeout after 30 seconds
            setTimeout(() => {
                if (pendingMessages.has(id)) {
                    pendingMessages.delete(id);
                    reject(new Error(`Bridge timeout: ${method}`));
                }
            }, 30000);
        });
    }

    // Handle response from native side
    window.handleBridgeResponse = function(id, error, result) {
        const pending = pendingMessages.get(id);
        if (!pending) {
            console.warn('[Bridge] Received response for unknown message ID:', id);
            return;
        }

        pendingMessages.delete(id);

        if (error) {
            pending.reject(new Error(error));
        } else {
            pending.resolve(result);
        }
    };

    // Create Host Bridge interface
    window.__HOST__ = {
        /**
         * Make an HTTP request to the daemon via Unix Domain Socket
         * @param {string} path - Request path (e.g., '/v1/models')
         * @param {RequestInit} init - Fetch options (method, headers, body)
         * @returns {Promise<Response>} - Fetch Response object
         */
        async request(path, init) {
            const response = await sendMessage('request', { path, init });

            // Convert response to a Response-like object
            return {
                ok: response.status >= 200 && response.status < 300,
                status: response.status,
                statusText: response.statusText || '',
                headers: new Headers(response.headers || {}),
                json: async () => response.body ? JSON.parse(response.body) : null,
                text: async () => response.body || '',
                blob: async () => new Blob([response.body]),
                arrayBuffer: async () => new TextEncoder().encode(response.body).buffer,
            };
        },

        /**
         * Open file/folder picker dialog
         * @param {string} type - 'models' or 'cache'
         * @returns {Promise<string|null>} - Selected path or null if cancelled
         */
        async openPathDialog(type) {
            return await sendMessage('openPathDialog', { type });
        },

        /**
         * Read server configuration
         * @returns {Promise<string>} - YAML config content
         */
        async readConfig() {
            return await sendMessage('readConfig', {});
        },

        /**
         * Write server configuration
         * @param {string} yaml - YAML config content
         * @returns {Promise<void>}
         */
        async writeConfig(yaml) {
            return await sendMessage('writeConfig', { yaml });
        },

        /**
         * Start the daemon
         * @returns {Promise<void>}
         */
        async startDaemon() {
            return await sendMessage('startDaemon', {});
        },

        /**
         * Stop the daemon
         * @returns {Promise<void>}
         */
        async stopDaemon() {
            return await sendMessage('stopDaemon', {});
        },

        /**
         * Get app and daemon versions
         * @returns {Promise<{app: string, daemon: string}>}
         */
        async getVersion() {
            return await sendMessage('getVersion', {});
        },
    };

    console.log('[Bridge] Host Bridge initialized successfully');

    // Dispatch custom event to notify app that bridge is ready
    window.dispatchEvent(new CustomEvent('hostBridgeReady'));
})();
