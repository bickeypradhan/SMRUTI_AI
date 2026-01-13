import path from 'path';
import { defineConfig, loadEnv } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig(({ mode }) => {
  const env = loadEnv(mode, '.', '');

  return {
    server: {
      port: 3000,
      host: true,
      allowedHosts: ['.trycloudflare.com']
    },
    plugins: [react()],
    define: {
      'process.env.API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.GEMINI_API_KEY': JSON.stringify(env.GEMINI_API_KEY),
      'process.env.LLM_PROVIDER': JSON.stringify(env.LLM_PROVIDER || ''),
      'process.env.GROQ_API_KEY': JSON.stringify(env.GROQ_API_KEY || ''),
      'process.env.GROQ_MODEL': JSON.stringify(env.GROQ_MODEL || ''),
      'process.env.OPENROUTER_API_KEY': JSON.stringify(env.OPENROUTER_API_KEY || ''),
      'process.env.OPENROUTER_MODEL': JSON.stringify(env.OPENROUTER_MODEL || '')
    },
    resolve: {
      alias: {
        '@': path.resolve(__dirname, '.')
      }
    }
  };
});
