# SMRUTI — Technical Project Slides (5)

---

## Slide 1 — What the Project Is

**SMRUTI (Decision Memory System)** is a single-page web app that captures, refines, and persists decisions (the “what” and the “why”) over time.

- UI: React-based SPA rendered into `#root`
- Core capability: decision + conversation capture, summarization, and continuity
- AI: integrates LLMs (Gemini primary) for assistance/clarification and structured updates
- Persistence: local-first storage in the browser (IndexedDB)
- Optional modality: microphone-driven audio capture/streaming for Live API workflows

---

## Slide 2 — Technologies Used (Frontend + Tooling)

**Build & Tooling**
- Node.js runtime for dev/build scripts
- Vite 6 (`vite`, `vite build`, `vite preview`) as dev server + bundler
- TypeScript ~5.8 with ES2022 target and bundler module resolution
- ES Modules (`"type": "module"`)

**Frontend Stack**
- React 19 + ReactDOM 19 (createRoot)
- React hooks (`useState`, `useEffect`, `useRef`, `useMemo`) for state + lifecycle
- Tailwind CSS (via CDN) for utility-first styling + dark mode (`dark` class)
- `lucide-react` for iconography
- `react-markdown` for rendering assistant output as Markdown
- Google Fonts (Inter)

---

## Slide 3 — App Architecture & Data Model

**Runtime Architecture**
- Single entrypoint: `index.html` loads `index.tsx` as a module
- Client-only (no backend in repo); runs fully in-browser
- State-driven UI: decisions, threads, interactions rendered from in-memory state

**Core Data Structures (TypeScript interfaces)**
- `Decision`: versioned decision record (title/category/intent/constraints/alternatives/finalDecision/confidence)
- `ChatThread` → `Interaction` → `AIResponse` for conversation history + AI output metadata
- Status lifecycle: decisions can be `active` or `archived`

**Local Persistence**
- IndexedDB database `SmrutiDB`
- Object stores:
  - `decisions` with indexes like `rootId` and `status`
  - `chatThreads` for conversation persistence

---

## Slide 4 — AI/LLM Integration & Environment Configuration

**Primary AI SDK**
- `@google/genai` (Gemini) used for LLM calls and Live API message types

**Provider Configuration (Vite `define`)**
- Injected at build-time into `process.env.*` for client usage:
  - `GEMINI_API_KEY` / `API_KEY`
  - optional: `LLM_PROVIDER`, `GROQ_API_KEY`, `GROQ_MODEL`, `OPENROUTER_API_KEY`, `OPENROUTER_MODEL`

**Prompting / Output Handling**
- AI responses are represented as structured objects (`AIResponse`) including intent + confidence
- Markdown formatting supported via `react-markdown`

**Local Dev Setup**
- Requires `.env.local` with `GEMINI_API_KEY` (per README)

---

## Slide 5 — Audio + Browser APIs, Security & Deployment Notes

**Audio / Live Modality (Browser APIs)**
- Microphone permission requested (`metadata.json`)
- Audio processing helpers:
  - Base64 encode/decode helpers for binary payloads
  - Web Audio API (`AudioContext`) for decoding PCM-like buffers
  - PCM16 conversion and packaging into `Blob` objects for live streaming workflows

**Security Considerations (Client-Side)**
- API keys are injected into the client bundle; treat them as exposed (use restricted keys and/or proxy in production)
- Local-first storage (IndexedDB) keeps user data in-browser, but is device-bound

**Deployment Model**
- Static site output via `vite build` (can be hosted on any static hosting/CDN)
- `vite preview` supports local production preview
