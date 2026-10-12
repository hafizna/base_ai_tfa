import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  // Plotly's WebGL dependencies use Node's Buffer/global names in browsers.
  resolve: { alias: { buffer: 'buffer/' } },
  define: { global: 'globalThis' },
  server: {
    proxy: {
      '/api': process.env.VITE_API_TARGET || 'http://localhost:8000',
    },
  },
})
