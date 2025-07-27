import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

// PostCSS with Tailwind CSS is automatically picked up by Vite
// through postcss.config.js, so no extra plugin is needed here.

export default defineConfig({
  plugins: [react()],
});
