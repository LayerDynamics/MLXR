//
//  TrayPopoverView.swift
//  MLXR
//
//  SwiftUI view for the tray popover showing quick status.
//

import SwiftUI

struct TrayPopoverView: View {

    @StateObject private var statusMonitor = DaemonStatusMonitor()

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            // Header
            HStack {
                Image(nsImage: NSImage(named: "AppIcon") ?? NSImage())
                    .resizable()
                    .frame(width: 32, height: 32)

                VStack(alignment: .leading, spacing: 2) {
                    Text("MLXR")
                        .font(.headline)
                    Text(statusMonitor.isRunning ? "Running" : "Stopped")
                        .font(.caption)
                        .foregroundColor(statusMonitor.isRunning ? .green : .red)
                }

                Spacer()

                Circle()
                    .fill(statusMonitor.isRunning ? Color.green : Color.red)
                    .frame(width: 8, height: 8)
            }

            Divider()

            // Metrics (if daemon is running)
            if statusMonitor.isRunning {
                VStack(alignment: .leading, spacing: 8) {
                    MetricRow(label: "Model", value: statusMonitor.currentModel ?? "None")
                    MetricRow(label: "Tokens/sec", value: String(format: "%.1f", statusMonitor.tokensPerSecond))
                    MetricRow(label: "Latency", value: String(format: "%.0f ms", statusMonitor.latency))
                    MetricRow(label: "KV Cache", value: String(format: "%.0f%%", statusMonitor.kvCacheUsage))
                }
            } else {
                Text("Daemon is not running")
                    .font(.caption)
                    .foregroundColor(.secondary)
            }

            Divider()

            // Actions
            HStack {
                if statusMonitor.isRunning {
                    Button("Stop") {
                        stopDaemon()
                    }
                    .buttonStyle(.bordered)
                } else {
                    Button("Start") {
                        startDaemon()
                    }
                    .buttonStyle(.borderedProminent)
                }

                Spacer()

                Button("Open") {
                    openMainWindow()
                }
                .buttonStyle(.bordered)
            }
        }
        .padding()
        .frame(width: 300)
        .onAppear {
            statusMonitor.startMonitoring()
        }
        .onDisappear {
            statusMonitor.stopMonitoring()
        }
    }

    // MARK: - Actions

    private func startDaemon() {
        Task {
            try? await DaemonManager.shared.startDaemon()
        }
    }

    private func stopDaemon() {
        Task {
            try? await DaemonManager.shared.stopDaemon()
        }
    }

    private func openMainWindow() {
        if let appDelegate = NSApp.delegate as? AppDelegate {
            appDelegate.showMainWindow()
        }
        // Close popover
        NSApp.deactivate()
    }
}

// MARK: - Metric Row

struct MetricRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack {
            Text(label)
                .font(.caption)
                .foregroundColor(.secondary)
            Spacer()
            Text(value)
                .font(.caption)
                .fontWeight(.medium)
        }
    }
}

// MARK: - Daemon Status Monitor

class DaemonStatusMonitor: ObservableObject {
    @Published var isRunning = false
    @Published var currentModel: String?
    @Published var tokensPerSecond: Double = 0
    @Published var latency: Double = 0
    @Published var kvCacheUsage: Double = 0

    private var timer: Timer?

    func startMonitoring() {
        updateStatus()
        timer = Timer.scheduledTimer(withTimeInterval: 1.0, repeats: true) { [weak self] _ in
            self?.updateStatus()
        }
    }

    func stopMonitoring() {
        timer?.invalidate()
        timer = nil
    }

    private func updateStatus() {
        Task { @MainActor in
            // Check if daemon is running
            if let isRunning = try? await DaemonManager.shared.isDaemonRunning() {
                self.isRunning = isRunning

                if isRunning {
                    // Fetch metrics from daemon
                    // TODO: Implement actual metrics fetching via UnixSocketClient
                    // For now, use placeholder values
                    self.currentModel = "TinyLlama-1.1B"
                    self.tokensPerSecond = 0.0
                    self.latency = 0.0
                    self.kvCacheUsage = 0.0
                }
            }
        }
    }
}

// MARK: - Preview

struct TrayPopoverView_Previews: PreviewProvider {
    static var previews: some View {
        TrayPopoverView()
    }
}
