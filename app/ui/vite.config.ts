import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import path from 'path'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@': path.resolve(__dirname, './src'),
    },
  },
  build: {
    target: 'esnext',
    minify: 'esbuild',
    cssMinify: true,
    rollupOptions: {
      output: {
        // Manual chunk splitting for optimal caching
        manualChunks: {
          // Core React libraries (rarely change)
          'vendor-react': ['react', 'react-dom', 'react-router-dom'],

          // UI components (change occasionally)
          'vendor-ui': [
            '@radix-ui/react-dialog',
            '@radix-ui/react-dropdown-menu',
            '@radix-ui/react-tooltip',
            '@radix-ui/react-select',
            '@radix-ui/react-switch',
            '@radix-ui/react-checkbox',
            '@radix-ui/react-slider',
            '@radix-ui/react-tabs',
            '@radix-ui/react-separator',
            '@radix-ui/react-slot',
            '@radix-ui/react-progress',
            '@radix-ui/react-label',
          ],

          // State management (change occasionally)
          'vendor-state': ['zustand', '@tanstack/react-query'],

          // Charts (large, rarely used immediately)
          'vendor-charts': ['recharts'],

          // Utilities (small, frequently used)
          'vendor-utils': [
            'clsx',
            'tailwind-merge',
            'date-fns',
            'class-variance-authority',
          ],

          // Markdown and syntax highlighting
          'vendor-content': [
            'react-markdown',
            'remark-gfm',
            'react-syntax-highlighter',
          ],

          // Internationalization
          'vendor-i18n': ['react-i18next', 'i18next'],

          // Animations
          'vendor-animation': ['framer-motion'],

          // Command palette
          'vendor-cmdk': ['cmdk'],

          // Virtualization
          'vendor-virtual': ['@tanstack/react-virtual'],
        },
      },
    },
    // Optimize chunk size for WebView
    chunkSizeWarningLimit: 500, // 500kb warning threshold
    sourcemap: false, // Disable source maps in production for smaller size
  },
  // Improve dev server performance
  server: {
    port: 5173,
    strictPort: true,
    fs: {
      strict: false,
    },
    // Proxy API requests to daemon during development
    proxy: {
      '/v1': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/api': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
      '/metrics': {
        target: 'http://localhost:8080',
        changeOrigin: true,
      },
    },
  },
  // Enable aggressive optimizations
  optimizeDeps: {
    include: [
      'react',
      'react-dom',
      'react-router-dom',
      'zustand',
      '@tanstack/react-query',
      '@tanstack/react-virtual',
    ],
  },
  // Preview server config
  preview: {
    port: 4173,
    strictPort: true,
  },
})
