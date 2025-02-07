import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';
import path from 'path';

export default defineConfig({
  plugins: [react()],
  resolve: {
    alias: {
      '@assets': path.resolve(__dirname, 'src/assets'),
      '@components': path.resolve(__dirname, 'src/components'),
    },
  },
  build: {
    outDir: 'DIST_DEPLOYMENT_FRONT_END',
    assetsDir: 'assets', // Ensures assets are properly copied
  },
});
