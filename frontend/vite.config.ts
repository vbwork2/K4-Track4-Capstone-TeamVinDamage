import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

declare const process: { env: Record<string, string | undefined> };

// Only one route: / . No P-115 workspace and no authentication redirects.
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0', port: 5173,
    proxy: {
      '/warning-api': {
        target: process.env.WARNING_BRIDGE_URL || 'http://127.0.0.1:8765',
        changeOrigin: true,
        rewrite: path => path.replace(/^\/warning-api/, ''),
      },
    },
  },
  build: { outDir: 'dist', target: 'es2022' },
});
