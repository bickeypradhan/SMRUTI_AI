
import React, { useState, useEffect, useRef, useMemo } from 'react';
import { createRoot } from 'react-dom/client';
import { GoogleGenAI, Type, Modality, LiveServerMessage, Blob } from "@google/genai";
import ReactMarkdown from 'react-markdown';
import { 
  Plus, 
  History, 
  BrainCircuit, 
  LayoutDashboard, 
  ChevronRight, 
  Save, 
  Mic, 
  MicOff, 
  Volume2, 
  VolumeX,
  ArrowLeft,
  Clock,
  CheckCircle2,
  AlertCircle,
  Archive,
  Trash2,
  X,
  Menu,
  Sun,
  Moon,
  Tag,
  BarChart3,
  RefreshCw,
  ExternalLink,
  Globe,
  SendHorizontal,
  Sparkles,
  Zap,
  PhoneOff,
  MessageSquare,
  PanelLeftClose,
  PanelLeftOpen
} from 'lucide-react';

// --- Core Utility Functions for Audio (Required for Live API) ---

function decode(base64: string) {
  const binaryString = atob(base64);
  const len = binaryString.length;
  const bytes = new Uint8Array(len);
  for (let i = 0; i < len; i++) {
    bytes[i] = binaryString.charCodeAt(i);
  }
  return bytes;
}

function encode(bytes: Uint8Array) {
  let binary = '';
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  return btoa(binary);
}

async function decodeAudioData(
  data: Uint8Array,
  ctx: AudioContext,
  sampleRate: number,
  numChannels: number,
): Promise<AudioBuffer> {
  const dataInt16 = new Int16Array(data.buffer);
  const frameCount = dataInt16.length / numChannels;
  const buffer = ctx.createBuffer(numChannels, frameCount, sampleRate);

  for (let channel = 0; channel < numChannels; channel++) {
    const channelData = buffer.getChannelData(channel);
    for (let i = 0; i < frameCount; i++) {
      channelData[i] = dataInt16[i * numChannels + channel] / 32768.0;
    }
  }
  return buffer;
}

function createAudioBlob(data: Float32Array): Blob {
  const l = data.length;
  const int16 = new Int16Array(l);
  for (let i = 0; i < l; i++) {
    int16[i] = data[i] * 32768;
  }
  return {
    data: encode(new Uint8Array(int16.buffer)),
    mimeType: 'audio/pcm;rate=16000',
  };
}

// --- Types & Interfaces ---

type DecisionStatus = 'active' | 'archived';

interface Decision {
  id: string;
  rootId: string;
  title: string;
  category: string;
  intent: string;
  constraints: string[];
  alternatives: string[];
  rejectedReasons: string[];
  finalDecision: string;
  confidence: string;
  reasoning: string;
  version: number;
  status: DecisionStatus;
  createdAt: number;
  updatedAt: number;
  updatedBy: 'user' | 'ai';
}

interface AIResponse {
  intent: 'assist' | 'update' | 'clarify' | 'general' | 'mixed';
  explanation: string;
  proposedChange?: Partial<Decision>;
  confidence: number;
  groundingChunks?: any[];
}

interface Interaction {
  id: string;
  query: string;
  response: AIResponse;
  timestamp: number;
}

interface ChatThread {
  id: string;
  title: string;
  interactions: Interaction[];
  updatedAt: number;
  createdAt: number;
}

const clampText = (value: string, maxLen: number) => {
  const v = (value || '').trim();
  if (v.length <= maxLen) return v;
  return v.slice(0, maxLen - 1).trimEnd() + '…';
};

const extractFirstBoldMarkdown = (markdown: string) => {
  const m = (markdown || '').match(/\*\*(.+?)\*\*/);
  return (m?.[1] || '').trim();
};

const extractFirstSentence = (text: string) => {
  const cleaned = (text || '')
    .replace(/\r\n/g, '\n')
    .replace(/```[\s\S]*?```/g, '')
    .replace(/\*\*(.+?)\*\*/g, '$1')
    .replace(/`([^`]+)`/g, '$1')
    .trim();

  if (!cleaned) return '';
  const parts = cleaned.split(/(?<=[.!?])\s+/g);
  return (parts[0] || cleaned).trim();
};

const isImportantOrDecisionRelated = (text: string) => {
  const t = (text || '').toLowerCase();
  if (!t.trim()) return false;

  // Strong signals
  if (/(\bbest suggestion\b|\bnext steps\b|\baction items\b|\bdecision\b|\bfinal decision\b|\bwe should\b|\brecommend\b|\bi suggest\b|\bgo with\b|\bchoose\b|\bcommit\b)/i.test(text)) {
    return true;
  }

  // Heuristic: multiple highlights indicates summarised/key points.
  const boldCount = (text.match(/\*\*/g) || []).length;
  if (boldCount >= 4) return true;

  // Medium signals
  if (/(\bpros and cons\b|\btrade-?offs\b|\balternative\b|\bconstraints\b|\brisk\b|\bplan\b|\broadmap\b)/i.test(text)) {
    return true;
  }

  return false;
};

const confidenceToLabel = (confidence: number) => {
  if (confidence >= 0.8) return 'High';
  if (confidence >= 0.6) return 'Medium';
  return 'Low';
};

const createDecisionFromInteraction = (interaction: Interaction): Decision => {
  const rootId = Math.random().toString(36).substring(2, 11);
  const titleFromBold = extractFirstBoldMarkdown(interaction.response.explanation);
  const title = clampText(titleFromBold || interaction.query || 'Saved from Inquiry', 64) || 'Saved from Inquiry';
  const summaryFromBold = titleFromBold;
  const summaryFromSentence = extractFirstSentence(interaction.response.explanation);
  const finalDecision = clampText(summaryFromBold || summaryFromSentence || 'Saved insight', 140) || 'Saved insight';

  const reasoning = [
    'Saved from conversation.',
    '',
    'User:',
    interaction.query || '',
    '',
    'Assistant:',
    interaction.response.explanation || '',
  ].join('\n').trim();

  const now = Date.now();
  return {
    id: `${rootId}_v1`,
    rootId,
    title,
    category: 'Inquiry',
    intent: interaction.response.intent || 'general',
    constraints: [],
    alternatives: [],
    rejectedReasons: [],
    finalDecision,
    confidence: confidenceToLabel(interaction.response.confidence ?? 0.6),
    reasoning,
    version: 1,
    status: 'active',
    createdAt: now,
    updatedAt: now,
    updatedBy: 'user',
  };
};

// --- Database Service (IndexedDB) ---

const DB_NAME = 'SmrutiDB';
const DB_VERSION = 2; // Incremented for chatThreads
const DECISION_STORE = 'decisions';
const THREAD_STORE = 'chatThreads';

const initDB = (): Promise<IDBDatabase> => {
  return new Promise((resolve, reject) => {
    const request = indexedDB.open(DB_NAME, DB_VERSION);
    request.onupgradeneeded = (event) => {
      const db = (event.target as IDBOpenDBRequest).result;
      if (!db.objectStoreNames.contains(DECISION_STORE)) {
        const store = db.createObjectStore(DECISION_STORE, { keyPath: 'id' });
        store.createIndex('rootId', 'rootId', { unique: false });
        store.createIndex('status', 'status', { unique: false });
      }
      if (!db.objectStoreNames.contains(THREAD_STORE)) {
        db.createObjectStore(THREAD_STORE, { keyPath: 'id' });
      }
    };
    request.onsuccess = () => resolve(request.result);
    request.onerror = () => reject(request.error);
  });
};

const storage = {
  async saveDecision(decision: Decision): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DECISION_STORE, 'readwrite');
      transaction.objectStore(DECISION_STORE).put(decision);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
  async getAllDecisions(): Promise<Decision[]> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DECISION_STORE, 'readonly');
      const request = transaction.objectStore(DECISION_STORE).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async saveThread(thread: ChatThread): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(THREAD_STORE, 'readwrite');
      transaction.objectStore(THREAD_STORE).put(thread);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
  async getAllThreads(): Promise<ChatThread[]> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(THREAD_STORE, 'readonly');
      const request = transaction.objectStore(THREAD_STORE).getAll();
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(request.error);
    });
  },
  async deleteThread(id: string): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(THREAD_STORE, 'readwrite');
      transaction.objectStore(THREAD_STORE).delete(id);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
  async deleteDecision(id: string): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DECISION_STORE, 'readwrite');
      transaction.objectStore(DECISION_STORE).delete(id);
      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
  async deleteDecisionRoot(rootId: string): Promise<void> {
    const db = await initDB();
    return new Promise((resolve, reject) => {
      const transaction = db.transaction(DECISION_STORE, 'readwrite');
      const store = transaction.objectStore(DECISION_STORE);
      const index = store.index('rootId');

      // Get all keys for this rootId, then delete them in the same transaction.
      const keysReq = (index as any).getAllKeys
        ? (index as any).getAllKeys(rootId)
        : null;

      if (keysReq) {
        keysReq.onsuccess = () => {
          const keys: IDBValidKey[] = keysReq.result || [];
          for (const k of keys) store.delete(k);
        };
        keysReq.onerror = () => reject(keysReq.error);
      } else {
        // Fallback for older browsers: iterate cursor.
        const cursorReq = index.openKeyCursor(IDBKeyRange.only(rootId));
        cursorReq.onsuccess = () => {
          const cursor = cursorReq.result;
          if (!cursor) return;
          store.delete(cursor.primaryKey);
          cursor.continue();
        };
        cursorReq.onerror = () => reject(cursorReq.error);
      }

      transaction.oncomplete = () => resolve();
      transaction.onerror = () => reject(transaction.error);
    });
  },
};

// --- AI Service ---

const getAI = () => new GoogleGenAI({ apiKey: process.env.GEMINI_API_KEY || process.env.API_KEY || '' });

const getGeminiApiKey = () => (process.env.GEMINI_API_KEY || process.env.API_KEY || '').trim();
const hasGeminiApiKey = () => getGeminiApiKey().length > 0;

type ChatMessage = {
  role: 'system' | 'user' | 'assistant';
  content: string;
};

const getLLMProvider = () => (process.env.LLM_PROVIDER || 'gemini').toLowerCase();
const getGroqModel = () => process.env.GROQ_MODEL || 'openai/gpt-oss-120b';
const getGroqApiKey = () => process.env.GROQ_API_KEY || '';

const isGroqEnabled = () => getLLMProvider() === 'groq' && getGroqApiKey().trim().length > 0;

const extractFirstJsonObject = (text: string): string | null => {
  const start = text.indexOf('{');
  const end = text.lastIndexOf('}');
  if (start === -1 || end === -1 || end <= start) return null;
  return text.slice(start, end + 1);
};

const groqChat = async (messages: ChatMessage[], model = getGroqModel()): Promise<string> => {
  const apiKey = getGroqApiKey().trim();
  if (!apiKey) throw new Error('GROQ_API_KEY is missing');

  const resp = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: {
      'Authorization': `Bearer ${apiKey}`,
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.2,
    }),
  });

  if (!resp.ok) {
    const body = await resp.text().catch(() => '');
    throw new Error(`Groq error ${resp.status}: ${body}`);
  }

  const data = await resp.json();
  return data?.choices?.[0]?.message?.content ?? '';
};

const normalizeForMatch = (value: string) =>
  value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, ' ')
    .trim();

const tokenizeForMatch = (value: string) => {
  const tokens = normalizeForMatch(value).split(/\s+/g);
  return tokens.filter(t => t.length >= 3);
};

const estimateMemoryRelevance = (query: string, decisions: Decision[]) => {
  if (!decisions.length) return { relevant: false, bestScore: 0, matchedTitle: '' };

  const qTokens = new Set(tokenizeForMatch(query));
  if (qTokens.size === 0) return { relevant: false, bestScore: 0, matchedTitle: '' };

  let bestScore = 0;
  let matchedTitle = '';

  for (const d of decisions) {
    const haystack = [
      d.title,
      d.category,
      d.intent,
      d.finalDecision,
      d.reasoning,
      ...(d.constraints || []),
      ...(d.alternatives || []),
      ...(d.rejectedReasons || []),
    ].join(' ');

    const dTokens = tokenizeForMatch(haystack);
    let score = 0;
    for (const t of dTokens) {
      if (qTokens.has(t)) score++;
    }

    // Strong boost if query directly mentions the title phrase
    const titleNorm = normalizeForMatch(d.title);
    if (titleNorm && normalizeForMatch(query).includes(titleNorm)) score += 3;

    if (score > bestScore) {
      bestScore = score;
      matchedTitle = d.title;
    }
  }

  // Heuristic threshold: 2 overlapping keywords or a direct title mention
  return { relevant: bestScore >= 2, bestScore, matchedTitle };
};

const selectMemoryPool = (allDecisions: Decision[]) => {
  const active = allDecisions.filter(d => d.status === 'active');
  if (active.length > 0) return active;

  // Fallback: if user archived everything, still use the latest version of each rootId.
  const latestByRoot = new Map<string, Decision>();
  for (const d of allDecisions) {
    const prev = latestByRoot.get(d.rootId);
    if (!prev || d.updatedAt > prev.updatedAt || d.version > prev.version) {
      latestByRoot.set(d.rootId, d);
    }
  }
  return Array.from(latestByRoot.values());
};

const shouldUseExternalKnowledge = (query: string) => {
  return /(latest|current|today|202\d|news|release|pricing|cost|compare|benchmark|best practice|research|source|cite|link|web|online)/i.test(query);
};

const isPersonalDocRequest = (query: string) => {
  return /(application|cover letter|resume|cv|statement of purpose|sop|email to|letter to|job|internship|scholarship)/i.test(query);
};

const mentionsPersonalDetails = (query: string) => {
  return /\b(my|me|mine|i am|i'm|my name|my email|my phone|my address|my linkedin|my github)\b/i.test(query);
};

type UserProfile = {
  name?: string;
  email?: string;
  phone?: string;
  address?: string;
  linkedin?: string;
  github?: string;
  website?: string;
  other?: string[];
};

const extractUserProfileFromDecisions = (decisions: Decision[]): UserProfile => {
  const text = decisions
    .map(d => [
      d.title,
      d.category,
      d.intent,
      d.finalDecision,
      d.reasoning,
      ...(d.constraints || []),
      ...(d.alternatives || []),
      ...(d.rejectedReasons || []),
    ].join(' '))
    .join('\n');

  const profile: UserProfile = {};

  const email = text.match(/[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}/i)?.[0];
  if (email) profile.email = email;

  const phone = text.match(/(\+?\d{1,3}[\s.-]?)?(\(?\d{3}\)?[\s.-]?)\d{3}[\s.-]?\d{4}/)?.[0];
  if (phone) profile.phone = phone;

  const linkedin = text.match(/https?:\/\/(www\.)?linkedin\.com\/[A-Za-z0-9_\-/?=&.%]+/i)?.[0];
  if (linkedin) profile.linkedin = linkedin;

  const github = text.match(/https?:\/\/(www\.)?github\.com\/[A-Za-z0-9_\-]+/i)?.[0];
  if (github) profile.github = github;

  const website = text.match(/https?:\/\/(?!www\.)?[A-Za-z0-9\-]+\.[A-Za-z]{2,}(\/[A-Za-z0-9_\-/?=&.%]*)?/i)?.[0];
  if (website) profile.website = website;

  // Lightweight name extraction: prefer decisions explicitly about name.
  const nameDecision = decisions.find(d => /\bname\b/i.test(d.title) || /\bname\b/i.test(d.category));
  const nameCandidate = (nameDecision?.finalDecision || nameDecision?.reasoning || '').trim();
  if (nameCandidate && nameCandidate.length <= 120) {
    // Attempt to strip common phrasing like "My name is ...".
    const match = nameCandidate.match(/\b(?:my name is|i am|i'm)\s+([A-Za-z][A-Za-z .,'-]{1,80})\b/i);
    profile.name = (match?.[1] || nameCandidate).trim();
  }

  // Capture a few "profile-ish" decision finalDecision snippets.
  const other: string[] = [];
  for (const d of decisions) {
    if (other.length >= 6) break;
    if (/profile|personal|identity|contact|bio/i.test(`${d.title} ${d.category}`) && d.finalDecision?.trim()) {
      other.push(`${d.title}: ${d.finalDecision.trim()}`);
    }
  }
  if (other.length) profile.other = other;

  return profile;
};

const containsPlaceholderTokens = (text: string) => {
  // Detect common template placeholders that should never be shown to the user.
  return /(\[(your|your full|full)\s*(name|details|email|phone|address)\])|(\[your\s+linkedin\])|(\[your\s+github\])|(\[your\s+website\])|(<\s*your\s*(name|email|phone|address)\s*>)/i.test(text);
};

const applyProfileToPlaceholders = (text: string, profile: UserProfile) => {
  let out = text;
  const replaceIf = (pattern: RegExp, value?: string) => {
    if (!value?.trim()) return;
    out = out.replace(pattern, value.trim());
  };

  replaceIf(/\[\s*your\s+name\s*\]/gi, profile.name);
  replaceIf(/<\s*your\s+name\s*>/gi, profile.name);

  replaceIf(/\[\s*your\s+email\s*\]/gi, profile.email);
  replaceIf(/<\s*your\s+email\s*>/gi, profile.email);

  replaceIf(/\[\s*your\s+phone\s*\]/gi, profile.phone);
  replaceIf(/<\s*your\s+phone\s*>/gi, profile.phone);

  replaceIf(/\[\s*your\s+address\s*\]/gi, profile.address);
  replaceIf(/<\s*your\s+address\s*>/gi, profile.address);

  replaceIf(/\[\s*your\s+linkedin\s*\]/gi, profile.linkedin);
  replaceIf(/\[\s*your\s+github\s*\]/gi, profile.github);
  replaceIf(/\[\s*your\s+website\s*\]/gi, profile.website);

  // Generic catch-alls (only when we have a clear field).
  replaceIf(/\[\s*your\s+details\s*\]/gi, [profile.email, profile.phone, profile.linkedin].filter(Boolean).join(' | '));
  return out;
};

const stripRemainingPlaceholders = (text: string) => {
  // Remove leftover placeholders rather than showing them verbatim.
  return text
    .replace(/\[\s*(your|your full|full)\s*(name|details|email|phone|address)\s*\]/gi, '')
    .replace(/\[\s*your\s+(linkedin|github|website)\s*\]/gi, '')
    .replace(/<\s*your\s*(name|details|email|phone|address)\s*>/gi, '')
    .replace(/\n{3,}/g, '\n\n')
    .trim();
};

const buildPersonalizationLayer = (opts: {
  query: string;
  contextStr: string;
  profileBlock: string;
  includeMemoryFacts: boolean;
}) => {
  const { query, contextStr, profileBlock, includeMemoryFacts } = opts;

  // Personalization layer: only user-specific facts and prior decisions.
  // Content layer: general writing/explanation skill.
  // IMPORTANT: never mention memory to the user.
  return `
PERSONALIZATION LAYER (silent):
- Use USER PROFILE only when relevant to the task (letters/emails/bio/career explanation/etc).
- If a personal field is required and missing, ask for it briefly instead of inserting placeholders.
- Never output template placeholders like [your name], [your details], <your email>.
- Never say "based on memory" or "stored records".

USER PROFILE (user-specific facts):\n${profileBlock}

${includeMemoryFacts ? `DECISIONS (user-specific context): ${contextStr}` : ''}

USER REQUEST: ${query}`.trim();
};

const formatUserProfileBlock = (profile: UserProfile) => {
  const lines: string[] = [];
  if (profile.name) lines.push(`Name: ${profile.name}`);
  if (profile.email) lines.push(`Email: ${profile.email}`);
  if (profile.phone) lines.push(`Phone: ${profile.phone}`);
  if (profile.address) lines.push(`Address: ${profile.address}`);
  if (profile.linkedin) lines.push(`LinkedIn: ${profile.linkedin}`);
  if (profile.github) lines.push(`GitHub: ${profile.github}`);
  if (profile.website) lines.push(`Website: ${profile.website}`);
  if (profile.other?.length) lines.push(...profile.other.map(x => `Other: ${x}`));
  return lines.length ? lines.join('\n') : 'No profile details found in stored memory.';
};

const isLikelyUpdateRequest = (query: string) => {
  return /(update|change|modify|edit|revise|replace|remove|delete|overwrite|correct|fix)\b/i.test(query);
};

const buildPlainTranscript = (history: Interaction[], query: string) => {
  const lines: string[] = [];
  for (const h of history) {
    lines.push(`User: ${h.query}`);
    if (h.response?.explanation) lines.push(`Assistant: ${h.response.explanation}`);
  }
  lines.push(`User: ${query}`);
  return lines.join('\n\n');
};

const llmGenerateText = async (opts: { system: string; user: string; allowSearch?: boolean }) => {
  const { system, user, allowSearch = false } = opts;

  if (isGroqEnabled()) {
    return (await groqChat([
      { role: 'system', content: system },
      { role: 'user', content: user },
    ])).trim();
  }

  const genAI = getAI();
  const config: any = { systemInstruction: system };
  if (allowSearch) config.tools = [{ googleSearch: {} }];

  const response = await genAI.models.generateContent({
    model: 'gemini-3-pro-preview',
    contents: user,
    config,
  });

  return (response.text || '').trim();
};

// Global audio state to manage interruptions
let activeAudioSource: AudioBufferSourceNode | null = null;
let activeAudioCtx: AudioContext | null = null;

const stopSpeaking = () => {
  if (activeAudioSource) {
    try {
      activeAudioSource.stop();
    } catch (e) {
      // Audio already stopped
    }
    activeAudioSource = null;
  }

  // Also stop any browser TTS
  try {
    if (typeof window !== 'undefined' && 'speechSynthesis' in window) {
      window.speechSynthesis.cancel();
    }
  } catch {
    // ignore
  }
};

const speakText = async (text: string) => {
  // Always clear existing speech before starting new logic
  stopSpeaking();

  // If Gemini key isn't configured, fall back to the browser SpeechSynthesis API.
  // This keeps "talking" working even when LLM_PROVIDER=groq.
  if (!hasGeminiApiKey()) {
    try {
      if (typeof window === 'undefined' || !('speechSynthesis' in window)) return;
      const utter = new SpeechSynthesisUtterance(text);
      utter.rate = 1;
      utter.pitch = 1;
      utter.lang = 'en-US';
      window.speechSynthesis.speak(utter);
    } catch (e) {
      console.error('Browser TTS failed', e);
    }
    return;
  }
  
  const ai = getAI();
  try {
    const response = await ai.models.generateContent({
      model: "gemini-2.5-flash-preview-tts",
      contents: [{ parts: [{ text: `Respond naturally to: ${text}` }] }],
      config: {
        responseModalities: [Modality.AUDIO],
        speechConfig: {
          voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } },
        },
      },
    });
    
    const base64Audio = response.candidates?.[0]?.content?.parts?.[0]?.inlineData?.data;
    if (base64Audio) {
      if (!activeAudioCtx) {
        activeAudioCtx = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      }
      
      const buffer = await decodeAudioData(decode(base64Audio), activeAudioCtx, 24000, 1);
      const source = activeAudioCtx.createBufferSource();
      source.buffer = buffer;
      source.connect(activeAudioCtx.destination);
      
      activeAudioSource = source;
      source.start();
      
      source.onended = () => {
        if (activeAudioSource === source) activeAudioSource = null;
      };
    }
  } catch (e) {
    console.error("Audio generation failed", e);
  }
};

const generateThreadTitle = async (firstQuery: string): Promise<string> => {
  if (isGroqEnabled()) {
    try {
      const content = await groqChat([
        {
          role: 'system',
          content: 'Generate a very concise 2-3 word title. Respond ONLY with the title.'
        },
        {
          role: 'user',
          content: `Chat thread starts with: "${firstQuery}"`
        }
      ]);
      return content.trim() || 'Logic Inquiry';
    } catch (e) {
      console.warn('Groq title generation failed; falling back to Gemini.', e);
    }
  }

  const genAI = getAI();
  const response = await genAI.models.generateContent({
    model: 'gemini-3-flash-preview',
    contents: `Generate a very concise 2-3 word title for a chat thread starting with: "${firstQuery}". Respond ONLY with the title.`,
  });
  return response.text?.trim() || "Logic Inquiry";
};

const parseIntent = async (query: string, context: Decision[], history: Interaction[]): Promise<AIResponse> => {
  const latestDecisions = selectMemoryPool(context);
  const attachRootIdIfMissing = (res: AIResponse): AIResponse => {
    if (res?.intent !== 'update' || !res?.proposedChange) return res;

    const proposed: any = res.proposedChange as any;
    if (typeof proposed.rootId === 'string' && proposed.rootId.trim()) return res;

    let inferredRootId: string | null = null;

    if (latestDecisions.length === 1) {
      inferredRootId = latestDecisions[0].rootId;
    } else {
      const rel = estimateMemoryRelevance(query, latestDecisions);
      if (rel?.matchedTitle) {
        inferredRootId = latestDecisions.find(d => d.title === rel.matchedTitle)?.rootId || null;
      }
    }

    if (!inferredRootId) {
      return {
        intent: 'clarify',
        explanation: "I can update your stored memory, but I couldn't tell which memory path you meant. Tell me the exact title from Timeline (or open Timeline and click the one you want to update), then repeat what to change.",
        confidence: 0,
      };
    }

    return {
      ...res,
      proposedChange: { ...res.proposedChange, rootId: inferredRootId },
    };
  };
  
  const contextStr = JSON.stringify(latestDecisions.map(d => ({
    title: d.title,
    finalDecision: d.finalDecision,
    reasoning: d.reasoning,
    category: d.category,
    constraints: d.constraints,
    alternatives: d.alternatives,
    rootId: d.rootId,
    version: d.version
  })));

  const relevance = estimateMemoryRelevance(query, latestDecisions);
  const wantsExternal = shouldUseExternalKnowledge(query);
  const wantsUpdate = isLikelyUpdateRequest(query);
  const wantsPersonal = isPersonalDocRequest(query) || mentionsPersonalDetails(query);

  const profile = extractUserProfileFromDecisions(latestDecisions);
  const profileBlock = formatUserProfileBlock(profile);

  // If the user is asking for a personal document but we have no memory, prompt for details instead of generating random credentials.
  if (wantsPersonal && latestDecisions.length === 0) {
    return {
      intent: 'clarify',
      explanation: "I don't have your personal details stored in memory yet (name/email/phone/links). Add them in the Memory tab, then ask again and I'll include them automatically.",
      confidence: 0,
    };
  }

  // Decide routing deterministically so general questions don't get "memory-only" answers.
  const routedIntent: AIResponse['intent'] =
    wantsUpdate && latestDecisions.length
      ? 'update'
      : (relevance.relevant || (wantsPersonal && latestDecisions.length)) && wantsExternal
        ? 'mixed'
        : (relevance.relevant || (wantsPersonal && latestDecisions.length))
          ? 'assist'
          : wantsExternal
            ? 'mixed'
            : 'general';

  const transcript = buildPlainTranscript(history, query);

  // UPDATE: keep the existing JSON-based behavior so the UI can offer "Commit Memory Evolution".
  if (routedIntent === 'update') {
    try {
      const systemInstruction = `You are SMRUTI, the Decision Memory Continuity System.
Maintain logical continuity.

HIGHLIGHTING RULE:
Wrap key decisions, primary logic points, and the single most important summary sentence in **double asterisks**.

SOURCE (INTERNAL MEMORY): ${contextStr}

TASK:
The user is requesting a modification to stored decision memory.
Return ONLY JSON with:
- intent: "update"
- explanation: what will change and why
- proposedChange: a partial Decision object with the fields that should be updated
- confidence: number 0-1
If the request is not specific enough, set intent to "clarify" and ask follow-ups.`;

      if (isGroqEnabled()) {
        const content = await groqChat([
          { role: 'system', content: systemInstruction },
          { role: 'user', content: transcript },
        ]);
        const jsonText = extractFirstJsonObject(content) ?? content;
        return attachRootIdIfMissing(JSON.parse(jsonText) as AIResponse);
      }

      const genAI = getAI();
      const response = await genAI.models.generateContent({
        model: 'gemini-3-pro-preview',
        contents: transcript,
        config: {
          systemInstruction,
          responseMimeType: 'application/json',
        },
      });

      return attachRootIdIfMissing(JSON.parse(response.text || '{}') as AIResponse);
    } catch (e) {
      console.warn('Update routing failed; falling back to clarify.', e);
      return { intent: 'clarify', explanation: "I couldn't safely update memory from that request. What exactly should change?", confidence: 0 };
    }
  }

  // ASSIST: memory-forward answer.
  if (routedIntent === 'assist') {
    const personalizationLayer = buildPersonalizationLayer({
      query,
      contextStr,
      profileBlock,
      includeMemoryFacts: true,
    });

    const system = `You are SMRUTI.

HIGHLIGHTING RULE:
Wrap key decisions and the single most important summary sentence in **double asterisks**.

CONTENT LAYER:
- Provide a complete, polished answer (applications/emails/explanations/etc).
- Use general writing skill and structure; do not be "memory-only".

${personalizationLayer}
`;

    const user = isPersonalDocRequest(query)
      ? (transcript + "\n\nIMPORTANT: This is a personal document request. You MUST use the USER PROFILE fields when available (name/email/phone/links). If any are missing, ask for them rather than making them up.")
      : transcript;

    let explanation = await llmGenerateText({ system, user, allowSearch: false });
    explanation = applyProfileToPlaceholders(explanation, profile);
    if (containsPlaceholderTokens(explanation)) {
      explanation = stripRemainingPlaceholders(explanation);
      if (wantsPersonal) {
        explanation += "\n\nTo finalize this, share any missing details you want included (e.g., full name, email, phone, LinkedIn).";
      }
    }
    return {
      intent: 'assist',
      explanation: explanation || "I don't have enough stored memory on that yet. What details should I remember?",
      confidence: 0.7,
    };
  }

  // GENERAL: normal answer, do NOT use memory context.
  if (routedIntent === 'general') {
    const personalizationRelevant = wantsPersonal || relevance.relevant || mentionsPersonalDetails(query);
    const personalizationLayer = buildPersonalizationLayer({
      query,
      contextStr,
      profileBlock,
      includeMemoryFacts: personalizationRelevant,
    });

    const system = `You are a helpful assistant.

HIGHLIGHTING RULE:
Wrap the single most important summary sentence in **double asterisks**.

CONTENT LAYER:
- Answer the user's question normally.
- For task-based outputs (emails, applications, explanations), output a complete result (not a template).

${personalizationRelevant ? personalizationLayer : ''}
`;

    let explanation = await llmGenerateText({ system, user: transcript, allowSearch: wantsExternal });
    explanation = applyProfileToPlaceholders(explanation, profile);
    if (containsPlaceholderTokens(explanation)) {
      explanation = stripRemainingPlaceholders(explanation);
      if (wantsPersonal) {
        explanation += "\n\nTo finalize this, share any missing details you want included (e.g., full name, email, phone, LinkedIn).";
      }
    }
    return {
      intent: 'general',
      explanation: explanation || "I'm not sure I understood—can you rephrase your question?",
      confidence: 0.7,
    };
  }

  // MIXED: single pass for speed. Memory is used for user-specific facts; web search (Gemini) is used for external facts.

  const system = `You are SMRUTI (mixed mode).

HIGHLIGHTING RULE:
Wrap key decisions and the single most important summary sentence in **double asterisks**.

INTERNAL MEMORY (user-specific facts): ${contextStr}

USER PROFILE (extracted from memory):\n${profileBlock}

GOAL:
Answer the user using the best combination of:
- internal memory for personal details and prior decisions
- general knowledge and (if available) web search for up-to-date external facts

RULES:
- Never invent personal credentials. If missing (e.g. name/email/phone), ask for them.
- Prefer INTERNAL MEMORY for anything about the user.
- End with a short section titled "Best suggestion" with 2-4 bullets.`;

  const user = isPersonalDocRequest(query)
    ? (transcript + "\n\nIMPORTANT: This is a personal document request. Ensure the output includes the user's stored credentials (Name/Email/Phone/Links) if present in USER PROFILE. If any required fields are missing, ask for them explicitly.")
    : transcript;

  let explanation = await llmGenerateText({ system, user, allowSearch: wantsExternal });
  explanation = applyProfileToPlaceholders(explanation, profile);
  if (containsPlaceholderTokens(explanation)) {
    explanation = stripRemainingPlaceholders(explanation);
    if (wantsPersonal) {
      explanation += "\n\nTo finalize this, share any missing details you want included (e.g., full name, email, phone, LinkedIn).";
    }
  }

  return {
    intent: 'mixed',
    explanation: explanation || "I couldn't generate a combined answer. Can you rephrase?",
    confidence: 0.75,
  };
};

// --- Custom Components ---

const AnimatedMarkdown = ({ content, intent }: { content: string, intent: string }) => {
  const getHighlightClass = () => {
    switch(intent) {
      case 'update': return 'highlight-amber';
      case 'general':
      case 'mixed': return 'highlight-indigo';
      default: return 'highlight-slate';
    }
  };

  return (
    <ReactMarkdown
      components={{
        strong: ({ node, ...props }) => (
          <strong className={`animated-highlight animate-highlightReveal ${getHighlightClass()}`} {...props} />
        )
      }}
    >
      {content}
    </ReactMarkdown>
  );
};

const Button = ({ children, onClick, variant = 'primary', className = '', icon: Icon, disabled = false, type = "button" }: any) => {
  const variants = {
    primary: 'bg-slate-900 text-white hover:bg-slate-800 dark:bg-slate-50 dark:text-slate-950 dark:hover:bg-slate-200 disabled:opacity-50 active:scale-95 shadow-sm',
    secondary: 'bg-white text-slate-900 border border-slate-200 hover:bg-slate-50 dark:bg-slate-900 dark:text-slate-50 dark:border-slate-800 dark:hover:bg-slate-800 active:scale-95 shadow-sm',
    ghost: 'text-slate-600 hover:bg-slate-100 dark:text-slate-400 dark:hover:bg-slate-800 active:scale-95',
    danger: 'text-red-600 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-950/30 active:scale-95'
  };
  return (
    <button type={type} disabled={disabled} onClick={onClick} className={`px-4 py-2.5 rounded-xl font-semibold transition-all flex items-center justify-center gap-2 touch-manipulation ${variants[variant as keyof typeof variants]} ${className}`}>
      {Icon && <Icon size={18} />}
      {children}
    </button>
  );
};

const Card = ({ children, className = '' }: any) => (
  <div className={`bg-white dark:bg-slate-900 border border-slate-200 dark:border-slate-800 rounded-2xl overflow-hidden shadow-sm ${className}`}>
    {children}
  </div>
);

// --- Live Assistant Mode ---

const LiveAssistantOverlay = ({
  onClose,
  decisions,
  audioContexts,
}: {
  onClose: () => void;
  decisions: Decision[];
  audioContexts?: { input: AudioContext; output: AudioContext } | null;
}) => {
  const [isConnecting, setIsConnecting] = useState(true);
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const nextStartTimeRef = useRef(0);
  const sourcesRef = useRef(new Set<AudioBufferSourceNode>());
  const sessionRef = useRef<any>(null);
  const audioContextsRef = useRef<{ input?: AudioContext; output?: AudioContext }>({});

  const recognitionRef = useRef<any>(null);
  const liveHistoryRef = useRef<Interaction[]>([]);
  const isClosingRef = useRef(false);

  useEffect(() => {
    let scriptProcessor: ScriptProcessorNode;
    let stream: MediaStream;

    const startSession = async () => {
      setError(null);
      setIsConnecting(true);

      // If no Gemini key is configured, run a browser-voice live loop (SpeechRecognition + Groq/Gemini text).
      if (!hasGeminiApiKey()) {
        try {
          const SpeechRecognition =
            (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
          if (!SpeechRecognition) {
            setError('Live voice needs Chrome/Edge SpeechRecognition (works on localhost/https).');
            setIsConnecting(false);
            setIsActive(false);
            return;
          }

          setIsConnecting(false);
          setIsActive(true);

          const recognition = new SpeechRecognition();
          recognition.continuous = true;
          recognition.interimResults = false;
          recognition.lang = 'en-US';

          recognition.onresult = async (event: any) => {
            try {
              const last = event.results?.[event.results.length - 1];
              const transcript = (last?.[0]?.transcript || '').trim();
              if (!transcript) return;

              // Pause recognition while we respond so we don't transcribe the assistant voice.
              try { recognition.stop(); } catch {}

              const res = await parseIntent(transcript, decisions, liveHistoryRef.current);
              const interaction: Interaction = {
                id: Math.random().toString(36).substring(2),
                query: transcript,
                response: res,
                timestamp: Date.now(),
              };
              liveHistoryRef.current = [...liveHistoryRef.current, interaction].slice(-12);

              if (res?.explanation) {
                // Use the same TTS helper; it will use browser SpeechSynthesis here.
                await speakText(res.explanation);
              }
            } catch (e) {
              console.error('Browser live loop failed', e);
              setError('Live voice failed. Check console for details.');
            } finally {
              // Restart recognition once speaking is done (or immediately if speaking is off).
              if (!isClosingRef.current) {
                setTimeout(() => {
                  try { recognition.start(); } catch {}
                }, 350);
              }
            }
          };

          recognition.onerror = (e: any) => {
            console.error('SpeechRecognition error', e);
            if (!isClosingRef.current) setError('Speech recognition error. Try reloading or using Chrome/Edge.');
          };

          recognition.onend = () => {
            if (!isClosingRef.current && isActive) {
              // Try to keep it running.
              setTimeout(() => {
                try { recognition.start(); } catch {}
              }, 400);
            }
          };

          recognitionRef.current = recognition;
          try { recognition.start(); } catch {}
          return;
        } catch (e) {
          console.error('Failed to start browser live loop', e);
          setError('Could not start live voice on this browser.');
          setIsConnecting(false);
          setIsActive(false);
          return;
        }
      }

      const ai = getAI();
      const inputCtx = audioContexts?.input || new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
      const outputCtx = audioContexts?.output || new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
      audioContextsRef.current = { input: inputCtx, output: outputCtx };

      try {
        // Autoplay policy compliance: try resuming in case contexts were created before.
        try { await inputCtx.resume(); } catch {}
        try { await outputCtx.resume(); } catch {}

        if (!navigator?.mediaDevices?.getUserMedia) {
          throw new Error('getUserMedia is not supported in this browser.');
        }
        stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        
        const sessionPromise = ai.live.connect({
          model: 'gemini-2.5-flash-native-audio-preview-09-2025',
          config: {
            responseModalities: [Modality.AUDIO],
            speechConfig: { voiceConfig: { prebuiltVoiceConfig: { voiceName: 'Zephyr' } } },
            systemInstruction: `You are SMRUTI, the decision memory system. You communicate via voice in a natural, friendly, and helpful manner. You have access to the user's decision context: ${JSON.stringify(decisions.map(d => d.title))}. Assist the user with their memory and reasoning.`
          },
          callbacks: {
            onopen: () => {
              setIsConnecting(false);
              setIsActive(true);
              const source = inputCtx.createMediaStreamSource(stream);
              scriptProcessor = inputCtx.createScriptProcessor(4096, 1, 1);
              scriptProcessor.onaudioprocess = (e) => {
                const inputData = e.inputBuffer.getChannelData(0);
                const pcmBlob = createAudioBlob(inputData);
                sessionPromise.then(session => session.sendRealtimeInput({ media: pcmBlob }));
              };
              source.connect(scriptProcessor);
              scriptProcessor.connect(inputCtx.destination);
            },
            onmessage: async (msg: LiveServerMessage) => {
              const base64Audio = msg.serverContent?.modelTurn?.parts[0]?.inlineData?.data;
              if (base64Audio && outputCtx) {
                nextStartTimeRef.current = Math.max(nextStartTimeRef.current, outputCtx.currentTime);
                const buffer = await decodeAudioData(decode(base64Audio), outputCtx, 24000, 1);
                const source = outputCtx.createBufferSource();
                source.buffer = buffer;
                source.connect(outputCtx.destination);
                source.start(nextStartTimeRef.current);
                nextStartTimeRef.current += buffer.duration;
                sourcesRef.current.add(source);
                source.onended = () => sourcesRef.current.delete(source);
              }
              if (msg.serverContent?.interrupted) {
                sourcesRef.current.forEach(s => { try { s.stop(); } catch(e) {} });
                sourcesRef.current.clear();
                nextStartTimeRef.current = 0;
              }
            },
            onclose: () => onClose(),
            onerror: (e) => {
              console.error('Live Error', e);
              setError('Live session error. Verify your Gemini key and model access.');
              setIsConnecting(false);
              setIsActive(false);
            }
          }
        });

        sessionRef.current = await sessionPromise;
      } catch (e) {
        console.error("Failed to start live session", e);
        setError((e as any)?.message || 'Failed to start live session.');
        setIsConnecting(false);
        setIsActive(false);
      }
    };

    startSession();

    return () => {
      isClosingRef.current = true;
      if (scriptProcessor) scriptProcessor.disconnect();
      if (stream) stream.getTracks().forEach(t => t.stop());
      if (audioContextsRef.current.input) audioContextsRef.current.input.close();
      if (audioContextsRef.current.output) audioContextsRef.current.output.close();
      if (sessionRef.current) sessionRef.current.close();

      // Cleanup browser speech recognition (fallback mode)
      if (recognitionRef.current) {
        try { recognitionRef.current.onresult = null; } catch {}
        try { recognitionRef.current.onerror = null; } catch {}
        try { recognitionRef.current.onend = null; } catch {}
        try { recognitionRef.current.stop(); } catch {}
      }
    };
  }, []);

  return (
    <div className="fixed inset-0 bg-slate-900/90 backdrop-blur-3xl z-[100] flex flex-col items-center justify-center p-4 sm:p-8 animate-fadeIn text-white">
      <div className="absolute top-4 right-4 sm:top-8 sm:right-8">
        <Button variant="ghost" onClick={onClose} className="text-white hover:bg-white/10 p-4 rounded-full">
          <PhoneOff size={32} />
        </Button>
      </div>

      <div className="relative flex items-center justify-center w-48 h-48 sm:w-64 sm:h-64 md:w-80 md:h-80">
        <div className={`absolute inset-0 rounded-full border-4 border-dashed border-white/20 ${isActive ? 'animate-[spin_20s_linear_infinite]' : ''}`}></div>
        <div className={`absolute inset-4 rounded-full border-2 border-dotted border-white/40 ${isActive ? 'animate-[spin_10s_linear_infinite_reverse]' : ''}`}></div>
        
        <div className={`relative z-10 w-24 h-24 sm:w-32 sm:h-32 md:w-40 md:h-40 bg-white/10 backdrop-blur-xl rounded-full flex items-center justify-center shadow-2xl ${isActive ? 'animate-pulse' : ''}`}>
          <BrainCircuit size={isActive ? 64 : 48} className={`transition-all duration-500 ${isActive ? 'text-white' : 'text-white/40'}`} />
        </div>
        
        {isActive && (
          <>
            <div className="absolute inset-[-20px] rounded-full border border-white/10 animate-[ping_3s_linear_infinite]"></div>
            <div className="absolute inset-[-40px] rounded-full border border-white/5 animate-[ping_4s_linear_infinite]"></div>
          </>
        )}
      </div>

      <div className="mt-16 text-center space-y-4">
        <h2 className="text-3xl font-black tracking-tighter uppercase">
          {error ? 'Error' : (isConnecting ? 'Initializing Synapse...' : 'Active')}
        </h2>
        <p className="text-slate-400 font-medium tracking-widest text-sm uppercase opacity-80">
          {error ? error : (isConnecting ? 'Calibrating neural frequency...' : 'Listening to your logic. Speak naturally.')}
        </p>
      </div>

      {!isConnecting && (
        <div className="absolute bottom-12 flex gap-1 items-end h-8">
          {[1,2,3,4,5,6,7,8].map(i => (
             <div key={i} className={`w-1 bg-white rounded-full transition-all duration-300 ${isActive ? 'animate-[bounce_1s_infinite]' : 'h-2'}`} style={{ animationDelay: `${i * 0.1}s`, height: isActive ? `${Math.random() * 20 + 10}px` : '4px' }}></div>
          ))}
        </div>
      )}
    </div>
  );
};

// --- Ask SMRUTI (with History) ---

const AskSmruti = ({ decisions, onUpdate, onDecisionSaved }: any) => {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [threads, setThreads] = useState<ChatThread[]>([]);
  const [activeThreadId, setActiveThreadId] = useState<string | null>(null);
  const [savingInteractionIds, setSavingInteractionIds] = useState<Record<string, boolean>>({});
  const [savedInteractionIds, setSavedInteractionIds] = useState<Record<string, boolean>>({});
  const [isListening, setIsListening] = useState(false);
  const [isVoiceOutputEnabled, setIsVoiceOutputEnabled] = useState(true);
  const [isLiveOverlayOpen, setIsLiveOverlayOpen] = useState(false);
  const [liveAudioContexts, setLiveAudioContexts] = useState<{ input: AudioContext; output: AudioContext } | null>(null);
  const [isSidebarOpen, setIsSidebarOpen] = useState(true);
  const scrollRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    loadThreads();
  }, []);

  useEffect(() => {
    // On small screens, default the history sidebar closed so the chat has room.
    try {
      if (typeof window !== 'undefined' && window.innerWidth < 768) {
        setIsSidebarOpen(false);
      }
    } catch {
      // ignore
    }
  }, []);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTo({ top: scrollRef.current.scrollHeight, behavior: 'smooth' });
    }
  }, [threads, activeThreadId, loading]);

  const loadThreads = async () => {
    const all = await storage.getAllThreads();
    setThreads(all.sort((a, b) => b.updatedAt - a.updatedAt));
  };

  const activeThread = useMemo(() => 
    threads.find(t => t.id === activeThreadId) || null,
  [threads, activeThreadId]);

  const toggleVoiceOutput = () => {
    const nextValue = !isVoiceOutputEnabled;
    setIsVoiceOutputEnabled(nextValue);
    // If sound is turned off, kill current speech immediately
    if (!nextValue) {
      stopSpeaking();
    }
  };

  const openLiveOverlay = async () => {
    // Prepare AudioContexts under a user gesture for better reliability (Gemini Live / autoplay policies).
    // If Gemini isn't configured, overlay falls back to browser voice mode and won't use these.
    try {
      if (hasGeminiApiKey()) {
        const input = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 16000 });
        const output = new (window.AudioContext || (window as any).webkitAudioContext)({ sampleRate: 24000 });
        try { await input.resume(); } catch {}
        try { await output.resume(); } catch {}
        setLiveAudioContexts({ input, output });
      } else {
        setLiveAudioContexts(null);
      }
    } catch (e) {
      console.warn('Failed to precreate AudioContexts for Live overlay', e);
      setLiveAudioContexts(null);
    }
    setIsLiveOverlayOpen(true);
  };

  const closeLiveOverlay = () => {
    setIsLiveOverlayOpen(false);
    setLiveAudioContexts(null);
  };

  const handleAsk = async () => {
    if (!query.trim() || loading) return;
    const currentQuery = query;
    setQuery('');
    setLoading(true);
    
    if (textareaRef.current) textareaRef.current.style.height = 'auto';

    try {
      const history = activeThread?.interactions || [];
      const latestDecisions = await storage.getAllDecisions();
      const res = await parseIntent(currentQuery, latestDecisions, history);
      
      const newInteraction: Interaction = {
        id: Math.random().toString(36).substring(2),
        query: currentQuery,
        response: res,
        timestamp: Date.now()
      };

      let updatedThread: ChatThread;

      if (!activeThreadId) {
        const title = await generateThreadTitle(currentQuery);
        updatedThread = {
          id: Math.random().toString(36).substring(2, 11),
          title,
          interactions: [newInteraction],
          createdAt: Date.now(),
          updatedAt: Date.now()
        };
        setActiveThreadId(updatedThread.id);
      } else {
        updatedThread = {
          ...activeThread!,
          interactions: [...activeThread!.interactions, newInteraction],
          updatedAt: Date.now()
        };
      }

      await storage.saveThread(updatedThread);
      await loadThreads();
      
      // Auto-narrate only if enabled
      if (isVoiceOutputEnabled && res.explanation) {
        speakText(res.explanation);
      }
    } catch (e) { 
      console.error(e); 
    }
    setLoading(false);
  };

  const startNewThread = () => {
    stopSpeaking(); // Stop any speech when clearing the context
    setActiveThreadId(null);
    setQuery('');
  };

  const deleteThread = async (e: React.MouseEvent, id: string) => {
    e.stopPropagation();
    await storage.deleteThread(id);
    if (activeThreadId === id) {
      stopSpeaking();
      setActiveThreadId(null);
    }
    await loadThreads();
  };

  const toggleMic = () => {
    if (!('webkitSpeechRecognition' in window)) return alert("Speech recognition not available.");
    const recognition = new (window as any).webkitSpeechRecognition();
    recognition.onstart = () => setIsListening(true);
    recognition.onresult = (e: any) => { setQuery(e.results[0][0].transcript); setIsListening(false); };
    recognition.onerror = () => setIsListening(false);
    recognition.start();
  };

  const shouldOfferSaveForInteraction = (interaction: Interaction) => {
    // Update intent already has a dedicated "Commit Memory Evolution" action.
    if (interaction.response.intent === 'update') return false;
    return isImportantOrDecisionRelated(interaction.response.explanation);
  };

  const saveInteractionToTimeline = async (interaction: Interaction) => {
    if (savingInteractionIds[interaction.id] || savedInteractionIds[interaction.id]) return;
    setSavingInteractionIds(prev => ({ ...prev, [interaction.id]: true }));
    try {
      const decision = createDecisionFromInteraction(interaction);
      await storage.saveDecision(decision);
      setSavedInteractionIds(prev => ({ ...prev, [interaction.id]: true }));
      await onDecisionSaved?.();
    } catch (e) {
      console.error('Failed to save to timeline', e);
    } finally {
      setSavingInteractionIds(prev => ({ ...prev, [interaction.id]: false }));
    }
  };

  return (
    <div className="h-full min-h-0 flex relative w-full overflow-hidden bg-white dark:bg-slate-950">
      {/* Mobile backdrop to close history drawer */}
      {isSidebarOpen && (
        <button
          type="button"
          aria-label="Close history"
          onClick={() => setIsSidebarOpen(false)}
          className="md:hidden fixed inset-0 z-30 bg-slate-900/40 backdrop-blur-[2px]"
        />
      )}

      {/* Sidebar for History */}
      <aside
        className={`fixed md:static inset-y-0 left-0 z-40 w-[86vw] max-w-[20rem] md:w-72 min-h-0 ${
          isSidebarOpen
            ? 'translate-x-0 md:w-72'
            : '-translate-x-full pointer-events-none md:pointer-events-auto md:translate-x-0 md:w-0'
        } transition-all duration-300 border-r border-slate-100 dark:border-slate-800 flex flex-col bg-slate-50/90 dark:bg-slate-900/90 backdrop-blur-md md:backdrop-blur-0 overflow-hidden shrink-0 shadow-2xl md:shadow-none`}
      >
        <div className="p-4 border-b border-slate-100 dark:border-slate-800 flex items-center gap-2">
          <Button variant="primary" onClick={startNewThread} className="flex-1 gap-2 text-sm" icon={Plus}>
            New Inquiry
          </Button>
          <button
            type="button"
            aria-label="Close history"
            onClick={() => setIsSidebarOpen(false)}
            className="md:hidden p-3 rounded-xl bg-white/80 dark:bg-slate-950/40 border border-slate-100 dark:border-slate-800 text-slate-700 dark:text-slate-200"
          >
            <X size={18} />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto p-2 space-y-1 touch-pan-y">
          {threads.map(t => (
            <div 
              key={t.id} 
              onClick={() => {
                stopSpeaking(); // Stop speech when switching threads
                setActiveThreadId(t.id);
                try {
                  if (typeof window !== 'undefined' && window.innerWidth < 768) setIsSidebarOpen(false);
                } catch {
                  // ignore
                }
              }}
              className={`group flex items-center justify-between p-3 rounded-xl cursor-pointer transition-all ${activeThreadId === t.id ? 'bg-slate-200 dark:bg-slate-800 text-slate-900 dark:text-slate-50' : 'text-slate-500 hover:bg-slate-100 dark:hover:bg-slate-800/50'}`}
            >
              <div className="flex items-center gap-3 min-w-0">
                <MessageSquare size={16} className="shrink-0 opacity-50" />
                <span className="text-sm font-semibold truncate pr-2">{t.title}</span>
              </div>
              <button 
                onClick={(e) => deleteThread(e, t.id)}
                className="opacity-0 group-hover:opacity-100 p-1.5 hover:bg-red-100 hover:text-red-500 rounded-lg transition-all"
              >
                <Trash2 size={14} />
              </button>
            </div>
          ))}
          {threads.length === 0 && (
            <div className="py-12 text-center">
               <Clock size={32} className="mx-auto text-slate-200 mb-2" />
               <p className="text-[10px] font-black uppercase text-slate-300 tracking-widest">No History</p>
            </div>
          )}
        </div>
      </aside>

      {/* Main Chat Area */}
      <div className="flex-1 min-h-0 flex flex-col relative min-w-0 h-full">
        <header className="py-3 sm:py-4 px-4 sm:px-6 shrink-0 flex justify-between items-center bg-white/80 dark:bg-slate-950/80 backdrop-blur-md sticky top-0 z-20 border-b border-slate-50 dark:border-slate-800">
          <div className="flex items-center gap-3 sm:gap-4 min-w-0">
            <button onClick={() => setIsSidebarOpen(!isSidebarOpen)} className="p-2 text-slate-400 hover:text-slate-900 dark:hover:text-slate-50 transition-colors">
              {isSidebarOpen ? <PanelLeftClose size={20} /> : <PanelLeftOpen size={20} />}
            </button>
            <div className="min-w-0">
              <h2 className="text-lg sm:text-xl font-black text-slate-900 dark:text-slate-50 tracking-tighter truncate">
                {activeThread ? activeThread.title : 'Cognitive Synthesis'}
              </h2>
              <p className="text-[9px] font-black text-slate-400 uppercase tracking-widest font-mono">Status: Neural Link Active</p>
            </div>
          </div>
          <div className="flex items-center gap-1.5 sm:gap-2 shrink-0">
             <Button variant="ghost" onClick={toggleVoiceOutput} className="p-2.5">
               {isVoiceOutputEnabled ? <Volume2 size={18} className="text-slate-900 dark:text-white" /> : <VolumeX size={18} className="text-slate-400" />}
             </Button>
             <Button variant="secondary" onClick={openLiveOverlay} className="gap-2 bg-slate-900 text-white dark:bg-white dark:text-slate-950 border-none px-3 sm:px-4 py-1.5 text-xs" icon={Zap}>
               Go Live
             </Button>
          </div>
        </header>

        <div ref={scrollRef} className="flex-1 overflow-y-auto px-4 sm:px-6 pt-6 sm:pt-8 pb-44 sm:pb-56 space-y-12 no-scrollbar touch-pan-y overscroll-contain">
          {!activeThread && !loading && (
            <div className="h-full flex flex-col items-center justify-center text-center py-20 animate-fadeIn opacity-50">
              <Sparkles size={64} className="mb-6 text-slate-100 dark:text-slate-800" />
              <h3 className="text-xl font-bold font-serif italic mb-2 text-slate-400">Initialize logic inquiry.</h3>
              <p className="text-sm text-slate-400 max-w-xs">Ask SMRUTI to synthesize memory or search external logic.</p>
            </div>
          )}

          {activeThread?.interactions.map((interaction) => (
            <div key={interaction.id} className="animate-fadeIn space-y-6">
              <div className="flex flex-col items-end">
                <div className="bg-slate-100 dark:bg-slate-800 px-5 py-3.5 rounded-2xl rounded-tr-none max-w-[92%] sm:max-w-[85%] shadow-sm border border-slate-200 dark:border-slate-700">
                  <p className="text-slate-900 dark:text-slate-100 font-semibold text-base md:text-lg">{interaction.query}</p>
                </div>
              </div>

              <Card className={`p-8 md:p-12 border-l-[8px] md:border-l-[16px] overflow-visible transition-all duration-700 ${interaction.response.intent === 'update' ? 'border-l-amber-500' : (interaction.response.intent === 'general' || interaction.response.intent === 'mixed') ? 'border-l-indigo-500' : 'border-l-slate-900 dark:border-l-slate-50'}`}>
                <div className="mb-6 flex justify-between items-center">
                  <span className={`px-2.5 py-1 rounded-lg text-[9px] font-black uppercase tracking-[0.2em] flex items-center gap-1.5 ${interaction.response.intent === 'update' ? 'bg-amber-100 dark:bg-amber-950 text-amber-700 dark:text-amber-400' : (interaction.response.intent === 'general' || interaction.response.intent === 'mixed') ? 'bg-indigo-100 dark:bg-indigo-950 text-indigo-700 dark:text-indigo-400' : 'bg-slate-100 dark:bg-slate-800 text-slate-700 dark:text-slate-300'}`}>
                    {interaction.response.intent}
                  </span>
                  <span className="text-[9px] font-bold text-slate-400 uppercase tracking-widest font-mono">{new Date(interaction.timestamp).toLocaleTimeString()}</span>
                </div>
                
                <div className="text-slate-800 dark:text-slate-100 text-lg md:text-xl leading-relaxed prose dark:prose-invert max-w-none font-medium">
                  <AnimatedMarkdown content={interaction.response.explanation} intent={interaction.response.intent} />
                </div>

                {shouldOfferSaveForInteraction(interaction) && (
                  <div className="mt-8 flex items-center justify-between gap-4">
                    <div className="flex items-center gap-2 text-[10px] font-black uppercase tracking-widest text-slate-400">
                      <AlertCircle size={14} className="opacity-70" />
                      <span>Important • Save to Timeline</span>
                    </div>
                    <Button
                      variant="secondary"
                      icon={Save}
                      disabled={!!savingInteractionIds[interaction.id] || !!savedInteractionIds[interaction.id]}
                      onClick={() => saveInteractionToTimeline(interaction)}
                      className="text-[10px] py-1.5 px-3 rounded-xl"
                    >
                      {savedInteractionIds[interaction.id] ? 'Saved' : (savingInteractionIds[interaction.id] ? 'Saving…' : 'Save')}
                    </Button>
                  </div>
                )}

                {interaction.response.intent === 'update' && interaction.response.proposedChange && (
                  <div className="mt-10 bg-amber-50/50 dark:bg-amber-950/20 p-6 md:p-8 rounded-3xl border border-amber-200 dark:border-amber-900 shadow-inner group">
                    <div className="flex items-center gap-4 mb-6">
                      <div className="p-3 bg-amber-100 dark:bg-amber-900 rounded-xl text-amber-600 dark:text-amber-400 group-hover:scale-110 transition-transform">
                        <RefreshCw size={24} className="animate-spin-slow"/>
                      </div>
                      <div>
                        <h4 className="font-black text-amber-900 dark:text-amber-100 text-lg">Evolve Path</h4>
                        <p className="text-[9px] font-bold text-amber-600 dark:text-amber-400 uppercase tracking-widest">Logic Modification</p>
                      </div>
                    </div>
                    
                    <div className="space-y-4 mb-8">
                       {interaction.response.proposedChange.finalDecision && (
                         <div className="p-4 bg-white/80 dark:bg-slate-900/80 rounded-xl border border-amber-200/50">
                            <p className="text-[8px] font-black text-amber-500 uppercase tracking-widest mb-1">New Path</p>
                            <p className="text-amber-950 dark:text-amber-50 font-bold italic">"{interaction.response.proposedChange.finalDecision}"</p>
                         </div>
                       )}
                    </div>

                    <Button
                      onClick={() => {
                        const change: any = interaction.response.proposedChange as any;
                        if (!change?.rootId) {
                          alert('Missing target memory path. Please ask again and mention the exact Timeline title you want to update.');
                          return;
                        }
                        onUpdate(change);
                      }}
                      className="bg-amber-600 dark:bg-amber-500 hover:bg-amber-700 dark:hover:bg-amber-400 text-white dark:text-slate-950 border-none w-full py-4 font-black rounded-xl"
                    >
                      Commit Memory Evolution
                    </Button>
                  </div>
                )}
              </Card>
            </div>
          ))}

          {loading && (
            <div className="flex flex-col items-center py-12 gap-5 animate-fadeIn">
              <div className="w-10 h-10 border-[4px] border-slate-100 dark:border-slate-800 border-t-slate-900 dark:border-t-slate-50 rounded-full animate-spin"></div>
              <p className="text-[10px] font-black text-slate-400 dark:text-slate-600 uppercase tracking-[0.4em] animate-pulse">Scanning Logic Corridors...</p>
            </div>
          )}
        </div>

        {/* INPUT AREA */}
        <div className="absolute bottom-20 md:bottom-8 left-0 right-0 px-3 sm:px-6 z-40">
          <div className="max-w-3xl mx-auto bg-white/95 dark:bg-slate-900/95 backdrop-blur-xl border-2 border-slate-100 dark:border-slate-800 rounded-[2rem] shadow-2xl p-2 md:p-3 ring-8 ring-slate-900/[0.03] dark:ring-white/[0.03]">
            <div className="flex items-end gap-2">
              <button 
                onClick={toggleMic} 
                className={`p-3.5 rounded-full transition-all shrink-0 mb-0.5 ${isListening ? 'bg-red-500 text-white animate-pulse' : 'bg-slate-50 dark:bg-slate-800 text-slate-600 dark:text-slate-300 hover:bg-slate-100 dark:hover:bg-slate-700'}`}
              >
                <Mic size={20} strokeWidth={2.5} />
              </button>
              
              <textarea 
                ref={textareaRef}
                className="flex-1 bg-transparent px-2 md:px-4 py-3 text-base md:text-lg font-medium text-slate-900 dark:text-slate-50 placeholder:text-slate-400 dark:placeholder:text-slate-600 focus:outline-none resize-none max-h-40 overflow-y-auto no-scrollbar min-h-[44px]"
                placeholder="Engage SMRUTI..."
                rows={1}
                value={query}
                onChange={e => {
                  setQuery(e.target.value);
                  e.target.style.height = 'auto';
                  e.target.style.height = `${Math.min(e.target.scrollHeight, 160)}px`;
                }}
                onKeyDown={e => {
                  if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    handleAsk();
                  }
                }}
              />
              
              <button 
                onClick={handleAsk} 
                disabled={loading || !query.trim()} 
                className={`rounded-full p-3.5 h-[48px] w-[48px] flex items-center justify-center transition-all shadow-lg active:scale-90 disabled:opacity-30 mb-0.5 ${query.trim() ? 'bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-950' : 'bg-slate-100 dark:bg-slate-800 text-slate-400'}`}
              >
                <SendHorizontal size={22} />
              </button>
            </div>
          </div>
        </div>
      </div>

      {isLiveOverlayOpen && (
        <LiveAssistantOverlay
          onClose={closeLiveOverlay}
          decisions={decisions}
          audioContexts={liveAudioContexts}
        />
      )}
    </div>
  );
};

// --- Theme Toggle ---

const ThemeToggle = ({ theme, toggleTheme }: { theme: string, toggleTheme: () => void }) => (
  <button onClick={toggleTheme} className="p-3 bg-slate-100 dark:bg-slate-800 text-slate-900 dark:text-slate-100 rounded-2xl hover:scale-110 transition-all duration-500 shadow-inner group relative overflow-hidden">
    <div className="relative z-10">{theme === 'light' ? <Moon size={20} strokeWidth={2.5} /> : <Sun size={20} strokeWidth={2.5} />}</div>
    <div className="absolute inset-0 bg-gradient-to-tr from-slate-200 to-white dark:from-slate-900 dark:to-slate-700 opacity-0 group-hover:opacity-100 transition-opacity"></div>
  </button>
);

// --- Navbar & Stats ---

const Navbar = ({ activeTab, setActiveTab, theme, toggleTheme }: { activeTab: string, setActiveTab: (t: string) => void, theme: string, toggleTheme: () => void }) => (
  <nav className="hidden md:flex w-64 bg-slate-50 dark:bg-slate-950 border-r border-slate-200 dark:border-slate-800 flex-col h-[100dvh] sticky top-0 shrink-0">
    <div className="p-8 flex flex-col gap-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-black tracking-tighter text-slate-900 dark:text-slate-50 flex items-center gap-2">
          <BrainCircuit className="text-slate-900 dark:text-slate-50" strokeWidth={3} /> SMRUTI
        </h1>
      </div>
      <ThemeToggle theme={theme} toggleTheme={toggleTheme} />
    </div>
    <div className="flex-1 px-4 space-y-1.5">
      <NavItem icon={LayoutDashboard} label="Dashboard" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
      <NavItem icon={Plus} label="New Decision" active={activeTab === 'new'} onClick={() => setActiveTab('new')} />
      <NavItem icon={History} label="Timeline" active={activeTab === 'timeline'} onClick={() => setActiveTab('timeline')} />
      <NavItem icon={BrainCircuit} label="Inquiry" active={activeTab === 'ask'} onClick={() => setActiveTab('ask')} />
    </div>
    <div className="p-8 text-[10px] text-slate-400 dark:text-slate-600 border-t border-slate-200 dark:border-slate-800 uppercase tracking-[0.2em] font-bold">Encrypted Local Core</div>
  </nav>
);

const MobileNavbar = ({ activeTab, setActiveTab }: { activeTab: string, setActiveTab: (t: string) => void }) => (
  <nav className="md:hidden fixed bottom-0 left-0 right-0 bg-white/90 dark:bg-slate-900/90 backdrop-blur-md border-t border-slate-200 dark:border-slate-800 flex justify-around items-center px-4 pb-safe z-50 h-20 shadow-[0_-8px_30px_rgb(0,0,0,0.04)]">
    <MobileNavItem icon={LayoutDashboard} label="Dash" active={activeTab === 'dashboard'} onClick={() => setActiveTab('dashboard')} />
    <MobileNavItem icon={Plus} label="Add" active={activeTab === 'new'} onClick={() => setActiveTab('new')} />
    <MobileNavItem icon={History} label="Paths" active={activeTab === 'timeline'} onClick={() => setActiveTab('timeline')} />
    <MobileNavItem icon={BrainCircuit} label="Ask" active={activeTab === 'ask'} onClick={() => setActiveTab('ask')} />
  </nav>
);

const NavItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button onClick={onClick} className={`w-full flex items-center gap-3 px-4 py-3.5 rounded-xl transition-all ${active ? 'bg-slate-900 text-white dark:bg-slate-50 dark:text-slate-950 shadow-lg shadow-slate-900/10 dark:shadow-white/5' : 'text-slate-600 dark:text-slate-400 hover:bg-slate-200/50 dark:hover:bg-slate-800/50'}`}>
    <Icon size={20} strokeWidth={active ? 2.5 : 2} />
    <span className="font-bold text-sm">{label}</span>
  </button>
);

const MobileNavItem = ({ icon: Icon, label, active, onClick }: any) => (
  <button onClick={onClick} className={`flex flex-col items-center justify-center gap-1 min-w-[64px] transition-all ${active ? 'text-slate-900 dark:text-slate-50' : 'text-slate-400'}`}>
    <div className={`p-2 rounded-xl transition-all ${active ? 'bg-slate-100 dark:bg-slate-800 scale-110' : ''}`}><Icon size={22} strokeWidth={active ? 2.5 : 2} /></div>
    <span className="text-[10px] font-bold uppercase tracking-widest">{label}</span>
  </button>
);

// --- Landing Page ---

const LandingPage = ({ onStart, theme, toggleTheme }: { onStart: () => void, theme: string, toggleTheme: () => void }) => (
  <div className="min-h-[100dvh] bg-white dark:bg-slate-950 flex flex-col items-center justify-center p-6 sm:p-8 text-center animate-fadeIn relative">
    <div className="absolute top-4 right-4 sm:top-8 sm:right-8"><ThemeToggle theme={theme} toggleTheme={toggleTheme} /></div>
    <div className="max-w-xl w-full">
      <div className="inline-flex p-5 bg-slate-50 dark:bg-slate-900 rounded-3xl mb-10 shadow-inner"><BrainCircuit size={48} className="text-slate-900 dark:text-slate-50" strokeWidth={2.5} /></div>
      <h1 className="text-4xl sm:text-5xl md:text-7xl font-black tracking-tighter text-slate-900 dark:text-slate-50 mb-6 uppercase">SMRUTI</h1>
      <p className="text-lg md:text-xl text-slate-600 dark:text-slate-400 mb-14 leading-relaxed font-medium">The Human–AI Decision Memory Continuity System. <br className="hidden md:block"/><span className="text-slate-400 dark:text-slate-500 italic">Capturing logic paths before they fade.</span></p>
      <Button onClick={onStart} className="px-12 py-5 text-xl mx-auto rounded-2xl w-full sm:w-auto">Initialize Core</Button>
    </div>
  </div>
);

// --- Dashboard ---

const Dashboard = ({ decisions, onRecord, onAsk }: any) => {
  const activeDecisions = decisions.filter((d: any) => d.status === 'active');
  const recent = [...activeDecisions].sort((a, b) => b.updatedAt - a.updatedAt).slice(0, 4);
  return (
    <div className="space-y-8 max-w-5xl mx-auto w-full pb-32 md:pb-0">
      <header className="px-2"><h2 className="text-3xl md:text-4xl font-black tracking-tight text-slate-900 dark:text-slate-50">Memory Hub</h2><p className="text-slate-500 dark:text-slate-400 mt-2 text-base font-medium">Tracking the logic of your path.</p></header>
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-4 md:gap-6 px-2">
        <StatCard label="Active Logic" value={activeDecisions.length} icon={BrainCircuit} />
        <StatCard label="Archived Paths" value={decisions.length - activeDecisions.length} icon={History} />
        <StatCard label="Core Integrity" value="Safe" icon={CheckCircle2} color="text-emerald-600 dark:text-emerald-400" />
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 md:gap-8 px-2">
        <Card className="p-6 md:p-8">
          <div className="flex justify-between items-center mb-8"><h3 className="font-black text-slate-900 dark:text-slate-50 uppercase tracking-[0.2em] text-[10px]">Logical Trace</h3><Button variant="ghost" onClick={onRecord} icon={Plus} className="text-[10px] py-1 px-2.5">Add Path</Button></div>
          <div className="space-y-4">{recent.length > 0 ? recent.map((d: any) => (
            <div key={d.id} className="flex items-center justify-between p-4 rounded-xl bg-slate-50 dark:bg-slate-900 border border-slate-100 dark:border-slate-800 hover:border-slate-300 dark:hover:border-slate-700 transition-all cursor-pointer group">
              <div className="min-w-0 flex-1"><h4 className="font-bold text-slate-900 dark:text-slate-50 truncate pr-4 text-sm">{d.title}</h4><p className="text-[9px] text-slate-400 uppercase tracking-widest mt-1">v{d.version} • {new Date(d.updatedAt).toLocaleDateString()}</p></div>
              <ChevronRight size={16} className="text-slate-300 group-hover:text-slate-900 dark:group-hover:text-slate-50" />
            </div>
          )) : <p className="text-slate-400 text-center py-12 text-sm italic">Core empty.</p>}</div>
        </Card>
        <Card className="p-8 md:p-10 bg-slate-900 dark:bg-slate-50 text-white dark:text-slate-950 flex flex-col justify-between relative overflow-hidden shadow-2xl">
          <div className="absolute top-0 right-0 p-10 opacity-10 dark:opacity-20"><BrainCircuit size={160} /></div>
          <div className="relative z-10"><h3 className="font-black uppercase tracking-[0.3em] text-[10px] mb-8 opacity-50">Inquiry Engine</h3><p className="text-slate-200 dark:text-slate-800 text-xl md:text-2xl leading-relaxed mb-10 font-serif italic">"Ask to find the reasoning hidden in your past."</p></div>
          <Button onClick={onAsk} variant="secondary" className="w-full py-4 relative z-10 font-bold rounded-xl" icon={BrainCircuit}>Engage Inquiry</Button>
        </Card>
      </div>
    </div>
  );
};

const StatCard = ({ label, value, icon: Icon, color = 'text-slate-900 dark:text-slate-50' }: any) => (
  <Card className="p-6 flex items-center gap-5 border-b-4 border-b-slate-900/5 dark:border-b-white/5"><div className="p-3 bg-slate-50 dark:bg-slate-950 rounded-xl shrink-0"><Icon size={20} className={`${color}`} strokeWidth={2.5} /></div><div><p className="text-[9px] font-black uppercase tracking-[0.2em] text-slate-400 mb-1">{label}</p><p className={`text-2xl font-black tracking-tight ${color}`}>{value}</p></div></Card>
);

const RecordDecision = ({ onSave }: any) => {
  const [formData, setFormData] = useState({ title: '', category: '', intent: '', alternatives: '', finalDecision: '', rejectedReasons: '', constraints: '', confidence: 'Medium', reasoning: '' });
  const handleSubmit = (e: any) => {
    e.preventDefault();
    onSave({ ...formData, constraints: formData.constraints.split(',').map(s => s.trim()).filter(Boolean), alternatives: formData.alternatives.split(',').map(s => s.trim()).filter(Boolean), rejectedReasons: formData.rejectedReasons.split(',').map(s => s.trim()).filter(Boolean) });
  };
  return (
    <div className="max-w-2xl mx-auto pb-32 md:pb-16 px-2 animate-fadeIn">
      <header className="mb-10 text-center"><h2 className="text-3xl md:text-5xl font-black text-slate-900 dark:text-slate-50 tracking-tight uppercase">Trace Path</h2><p className="text-slate-500 dark:text-slate-400 mt-2 font-serif italic">Formalize the memory of your choice.</p></header>
      <form onSubmit={handleSubmit} className="space-y-6">
        <Input label="Title" value={formData.title} onChange={(v: string) => setFormData({...formData, title: v})} required placeholder="The decision subject" />
        <TextArea label="Context & Reasoning" value={formData.reasoning} onChange={(v: string) => setFormData({...formData, reasoning: v})} rows={4} placeholder="Why are you deciding this?" />
        <Input label="Final Path" value={formData.finalDecision} onChange={(v: string) => setFormData({...formData, finalDecision: v})} required placeholder="The choice made" />
        <Button type="submit" className="w-full py-5 text-lg font-black shadow-xl rounded-2xl" icon={Save}>Commit to Memory</Button>
      </form>
    </div>
  );
};

const Input = ({ label, value, onChange, placeholder, required }: any) => (
  <div className="flex flex-col gap-2"><label className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 ml-1">{label}</label><input required={required} className="px-5 py-3.5 bg-slate-50 dark:bg-slate-950/50 border border-slate-200 dark:border-slate-800 rounded-xl focus:border-slate-900 dark:focus:border-slate-100 focus:outline-none text-slate-900 dark:text-slate-50" placeholder={placeholder} value={value} onChange={e => onChange(e.target.value)} /></div>
);

const TextArea = ({ label, value, onChange, placeholder, required, rows = 3 }: any) => (
  <div className="flex flex-col gap-2"><label className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 ml-1">{label}</label><textarea required={required} rows={rows} className="px-5 py-3.5 bg-slate-50 dark:bg-slate-950/50 border border-slate-200 dark:border-slate-800 rounded-xl focus:border-slate-900 dark:focus:border-slate-100 focus:outline-none resize-none text-slate-900 dark:text-slate-50" placeholder={placeholder} value={value} onChange={e => onChange(e.target.value)} /></div>
);

const Timeline = ({ decisions, onViewHistory, onDelete, onEdit }: any) => {
  const active = useMemo(() => decisions.filter((d: any) => d.status === 'active').sort((a: any, b: any) => b.updatedAt - a.updatedAt), [decisions]);
  return (
    <div className="space-y-8 max-w-4xl mx-auto w-full px-2 pb-32 md:pb-12">
      <header><h2 className="text-3xl font-black text-slate-900 dark:text-slate-50 tracking-tight uppercase">Logic Timeline</h2></header>
      <div className="space-y-6">{active.length > 0 ? active.map((d: any) => (
          <Card key={d.id} className="p-6 md:p-8 hover:shadow-xl hover:-translate-y-1 transition-all cursor-pointer" onClick={() => onViewHistory(d.rootId)}>
            <div className="flex flex-col sm:flex-row sm:justify-between sm:items-start gap-4 mb-6">
              <div className="min-w-0"><h3 className="text-xl font-black truncate">{d.title}</h3><p className="text-[9px] font-black uppercase text-slate-400">PATH_v{d.version} • {new Date(d.updatedAt).toLocaleDateString()}</p></div>
              <div className="flex flex-wrap items-center justify-start sm:justify-end gap-2">
                <Button variant="secondary" onClick={(e: any) => { e.stopPropagation(); onViewHistory(d.rootId); }} icon={History} className="text-[9px] py-1.5">View Trace</Button>
                <Button variant="ghost" onClick={(e: any) => { e.stopPropagation(); onEdit?.(d.rootId); }} icon={RefreshCw} className="text-[9px] py-1.5">Update</Button>
                <Button variant="danger" onClick={(e: any) => { e.stopPropagation(); onDelete?.(d.rootId); }} icon={Trash2} className="text-[9px] py-1.5">Delete</Button>
              </div>
            </div>
            <div className="p-5 bg-slate-50 dark:bg-slate-950 rounded-xl border-l-4 border-slate-900 dark:border-slate-50 italic font-serif text-sm">"{d.finalDecision}"</div>
          </Card>
        )) : <div className="text-center py-20 italic text-slate-400 text-sm">No memory paths found.</div>}</div>
    </div>
  );
};

const EditDecisionModal = ({ decision, onClose, onSave }: { decision: Decision; onClose: () => void; onSave: (data: any) => void; }) => {
  const [formData, setFormData] = useState(() => ({
    title: decision.title || '',
    category: decision.category || '',
    intent: decision.intent || '',
    alternatives: (decision.alternatives || []).join(', '),
    finalDecision: decision.finalDecision || '',
    rejectedReasons: (decision.rejectedReasons || []).join(', '),
    constraints: (decision.constraints || []).join(', '),
    confidence: decision.confidence || 'Medium',
    reasoning: decision.reasoning || '',
  }));

  const handleSubmit = (e: any) => {
    e.preventDefault();
    onSave({
      rootId: decision.rootId,
      title: formData.title,
      category: formData.category,
      intent: formData.intent,
      alternatives: formData.alternatives.split(',').map((s: string) => s.trim()).filter(Boolean),
      finalDecision: formData.finalDecision,
      rejectedReasons: formData.rejectedReasons.split(',').map((s: string) => s.trim()).filter(Boolean),
      constraints: formData.constraints.split(',').map((s: string) => s.trim()).filter(Boolean),
      confidence: formData.confidence,
      reasoning: formData.reasoning,
    });
  };

  return (
    <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-xl z-[110] flex items-center justify-center p-4">
      <div className="w-full max-w-2xl max-h-[85dvh] flex flex-col bg-white dark:bg-slate-950 rounded-[2rem] shadow-2xl overflow-hidden">
        <div className="p-4 sm:p-6 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/50">
          <div>
            <h3 className="text-xl font-black text-slate-900 dark:text-slate-50">Update Path</h3>
            <p className="text-[10px] font-black uppercase tracking-[0.2em] text-slate-400 mt-1">This will create a new version</p>
          </div>
          <button onClick={onClose} className="p-3 bg-white dark:bg-slate-900 rounded-xl shadow-sm"><X size={20}/></button>
        </div>

        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-8 touch-pan-y overscroll-contain">
          <form onSubmit={handleSubmit} className="space-y-6">
            <Input label="Title" value={formData.title} onChange={(v: string) => setFormData({ ...formData, title: v })} required placeholder="The decision subject" />
            <Input label="Category" value={formData.category} onChange={(v: string) => setFormData({ ...formData, category: v })} required placeholder="Category" />
            <Input label="Intent" value={formData.intent} onChange={(v: string) => setFormData({ ...formData, intent: v })} required placeholder="Intent" />
            <TextArea label="Context & Reasoning" value={formData.reasoning} onChange={(v: string) => setFormData({ ...formData, reasoning: v })} rows={4} placeholder="Why are you deciding this?" />
            <Input label="Final Path" value={formData.finalDecision} onChange={(v: string) => setFormData({ ...formData, finalDecision: v })} required placeholder="The choice made" />
            <Input label="Constraints (comma-separated)" value={formData.constraints} onChange={(v: string) => setFormData({ ...formData, constraints: v })} placeholder="e.g., budget, time" />
            <Input label="Alternatives (comma-separated)" value={formData.alternatives} onChange={(v: string) => setFormData({ ...formData, alternatives: v })} placeholder="e.g., Option A, Option B" />
            <Input label="Rejected Reasons (comma-separated)" value={formData.rejectedReasons} onChange={(v: string) => setFormData({ ...formData, rejectedReasons: v })} placeholder="Why alternatives were rejected" />

            <div className="flex gap-3">
              <Button type="button" variant="secondary" onClick={onClose} className="flex-1 py-4">Cancel</Button>
              <Button type="submit" className="flex-1 py-4 font-black" icon={Save}>Save Update</Button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

const HistoryModal = ({ rootId, decisions, onClose }: any) => {
  const versions = decisions.filter((d: any) => d.rootId === rootId).sort((a: any, b: any) => b.version - a.version);
  return (
    <div className="fixed inset-0 bg-slate-900/80 backdrop-blur-xl z-[100] flex items-center justify-center p-4">
      <div className="w-full max-w-4xl max-h-[85dvh] flex flex-col bg-white dark:bg-slate-950 rounded-[2rem] shadow-2xl overflow-hidden">
        <div className="p-4 sm:p-6 border-b border-slate-100 dark:border-slate-800 flex justify-between items-center bg-slate-50/50 dark:bg-slate-900/50">
          <h3 className="text-xl font-black text-slate-900 dark:text-slate-50">Memory Evolution</h3>
          <button onClick={onClose} className="p-3 bg-white dark:bg-slate-900 rounded-xl shadow-sm"><X size={20}/></button>
        </div>
        <div className="flex-1 overflow-y-auto p-4 sm:p-6 md:p-12 space-y-12 touch-pan-y overscroll-contain">{versions.map((v: any) => (
            <div key={v.id} className="border-l-4 border-slate-100 dark:border-slate-800 pl-6 md:pl-10 relative">
              <div className="absolute left-[-11px] top-0 w-5 h-5 bg-slate-900 dark:bg-white rounded-full flex items-center justify-center text-[8px] font-black text-white dark:text-slate-900">{v.version}</div>
              <h4 className="font-black text-xl mb-4">Version {v.version}</h4>
              <div className="bg-slate-50/80 dark:bg-slate-900/80 p-6 rounded-2xl italic font-serif text-base md:text-lg mb-6 leading-relaxed">"{v.finalDecision}"</div>
              <div className="prose dark:prose-invert max-w-none text-sm md:text-base"><AnimatedMarkdown content={v.reasoning} intent="assist" /></div>
            </div>
          ))}</div>
      </div>
    </div>
  );
};

// --- Main App Root ---

const App = () => {
  const [started, setStarted] = useState(false);
  const [activeTab, setActiveTab] = useState('dashboard');
  const [decisions, setDecisions] = useState<Decision[]>([]);
  const [historyRootId, setHistoryRootId] = useState<string | null>(null);
  const [editRootId, setEditRootId] = useState<string | null>(null);
  const [theme, setTheme] = useState(() => localStorage.getItem('theme') || 'light');
  
  useEffect(() => { document.documentElement.classList.toggle('dark', theme === 'dark'); localStorage.setItem('theme', theme); }, [theme]);
  const toggleTheme = () => setTheme(prev => prev === 'light' ? 'dark' : 'light');
  useEffect(() => { loadDecisions(); }, []);
  
  const loadDecisions = async () => setDecisions(await storage.getAllDecisions());
  
  const handleSave = async (data: any) => { 
    const rootId = Math.random().toString(36).substring(2, 11); 
    const d: Decision = { ...data, id: `${rootId}_v1`, rootId, version: 1, status: 'active', createdAt: Date.now(), updatedAt: Date.now(), updatedBy: 'user' }; 
    await storage.saveDecision(d); await loadDecisions(); setActiveTab('timeline'); 
  };
  
  const handleUpdate = async (change: any) => { 
    const current = decisions.find(d => d.rootId === change.rootId && d.status === 'active'); 
    if (!current) return; 
    await storage.saveDecision({ ...current, status: 'archived', updatedAt: Date.now() }); 
    const next: Decision = { ...current, ...change, id: `${current.rootId}_v${current.version + 1}`, version: current.version + 1, status: 'active', updatedAt: Date.now(), updatedBy: 'ai' }; 
    await storage.saveDecision(next); await loadDecisions(); setActiveTab('timeline'); setHistoryRootId(current.rootId); 
  };

  const handleDelete = async (rootId: string) => {
    const ok = window.confirm('Delete this memory path? This will remove all its versions from your local database.');
    if (!ok) return;
    if (historyRootId === rootId) setHistoryRootId(null);
    if (editRootId === rootId) setEditRootId(null);
    await storage.deleteDecisionRoot(rootId);
    await loadDecisions();
  };

  const handleManualUpdate = async (data: any) => {
    const current = decisions.find(d => d.rootId === data.rootId && d.status === 'active');
    if (!current) return;

    await storage.saveDecision({ ...current, status: 'archived', updatedAt: Date.now() });
    const next: Decision = {
      ...current,
      ...data,
      id: `${current.rootId}_v${current.version + 1}`,
      version: current.version + 1,
      status: 'active',
      updatedAt: Date.now(),
      updatedBy: 'user',
    };
    await storage.saveDecision(next);
    await loadDecisions();
    setActiveTab('timeline');
    setEditRootId(null);
    setHistoryRootId(current.rootId);
  };
  
  if (!started) return <LandingPage onStart={() => setStarted(true)} theme={theme} toggleTheme={toggleTheme} />;
  
  return (
    <div className="flex flex-col md:flex-row bg-white dark:bg-slate-950 min-h-[100dvh] md:h-screen overflow-hidden font-sans transition-colors duration-500 selection:bg-slate-900 dark:selection:bg-slate-100 selection:text-white dark:selection:text-slate-900">
      <Navbar activeTab={activeTab} setActiveTab={setActiveTab} theme={theme} toggleTheme={toggleTheme} />
      <main className="flex-1 h-[100dvh] md:h-screen min-h-0 overflow-hidden flex flex-col relative bg-white dark:bg-slate-950">
        <div className="flex-1 overflow-y-auto w-full p-4 sm:p-10 md:p-16 no-scrollbar touch-pan-y overscroll-contain">
          <div className="max-w-7xl mx-auto h-full">
            {activeTab === 'dashboard' && <Dashboard decisions={decisions} onRecord={() => setActiveTab('new')} onAsk={() => setActiveTab('ask')} />}
            {activeTab === 'new' && <RecordDecision onSave={handleSave} />}
            {activeTab === 'timeline' && <Timeline decisions={decisions} onViewHistory={setHistoryRootId} onDelete={handleDelete} onEdit={setEditRootId} />}
            {activeTab === 'ask' && <AskSmruti decisions={decisions} onUpdate={handleUpdate} onDecisionSaved={loadDecisions} />}
          </div>
        </div>
      </main>
      <MobileNavbar activeTab={activeTab} setActiveTab={setActiveTab} />
      {historyRootId && <HistoryModal rootId={historyRootId} decisions={decisions} onClose={() => setHistoryRootId(null)} />}
      {editRootId && (
        <EditDecisionModal
          decision={decisions.find(d => d.rootId === editRootId && d.status === 'active') as Decision}
          onClose={() => setEditRootId(null)}
          onSave={handleManualUpdate}
        />
      )}
    </div>
  );
};

const rootEl = document.getElementById('root');
if (rootEl) createRoot(rootEl).render(<App />);
