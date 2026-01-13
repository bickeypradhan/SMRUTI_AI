# SMRUTI — PowerPoint-Ready Outline (5 Slides)

## Slide 1 — SMRUTI: Project Overview
- Single-page React web app for capturing and maintaining “decision memory” over time
- Tracks the decision plus the reasoning (constraints, alternatives, confidence)
- AI-assisted drafting, clarifying, and updating of decision records
- Local-first: runs fully in the browser with offline-friendly persistence
- Optional voice mode: microphone input and audio pipeline support

## Slide 2 — Tech Stack (Frontend + Tooling)
- React 19 + ReactDOM 19 (SPA rendering with `createRoot`)
- TypeScript ~5.8 (ES2022 target, ESM)
- Vite 6 dev/build pipeline (`dev`, `build`, `preview`)
- Tailwind CSS (CDN) for UI styling + dark mode via `dark` class
- `lucide-react` for icons and `react-markdown` for Markdown rendering

## Slide 3 — Architecture & Data Model
- Client-only architecture: `index.html` loads `index.tsx` as the main entrypoint
- UI state driven by React hooks (`useState`, `useEffect`, `useRef`, `useMemo`)
- Core entities: Decision, ChatThread, Interaction, AIResponse
- Versioned decisions (v1, v2…) with lifecycle state (`active` / `archived`)
- Design goal: continuity across sessions via structured records

## Slide 4 — AI Integration & Configuration
- Uses Google Gemini SDK: `@google/genai` (text + Live API types)
- Environment-based configuration injected by Vite (`process.env.*`)
- Primary key: `GEMINI_API_KEY` (set in `.env.local` for local dev)
- Optional provider hooks present (Groq/OpenRouter env keys and model names)
- Renders model output as Markdown for readable, formatted responses

## Slide 5 — Storage, Audio, and Deployment Notes
- Persistence via IndexedDB (`SmrutiDB`) with stores: `decisions`, `chatThreads`
- Indexed queries for fast retrieval (e.g., by decision rootId/status)
- Voice/audio pipeline: Web Audio API + PCM16/base64 helpers + audio blobs
- Security note: client-injected API keys are exposed; use restricted keys/proxy in production
- Deployment: static build output from Vite; host on any static hosting/CDN
