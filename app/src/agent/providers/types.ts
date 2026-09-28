/** Provider configuration, as softn Studio's (`types/studio.ts`) defines it. */

export type ProviderType = 'anthropic' | 'openai' | 'local' | 'custom';

export type LocalServerKind = 'ollama' | 'lmstudio' | 'oaiy' | 'other';

export interface ProviderConfig {
  id: string;
  type: ProviderType;
  name: string;
  apiKey: string;
  /** The server's address or API base (`http://localhost:11434`, `https://example.com/v1`). */
  baseUrl?: string;
  /** The model every request uses. There is no default. */
  modelId?: string;
  orgId?: string;
  /** For a `local` provider: which server, for its address and its advice. */
  serverKind?: LocalServerKind;
  /** The context window in tokens, as the person set it: wins over anything detected. */
  contextTokens?: number;
  /** The window the server reported for `model` (or stated in an overflow error). */
  detectedContext?: { model: string; tokens: number; how: string; at: number };
  /** How many agents may use this provider at once (default 1 for a local server, 3 for an API). */
  parallelAgents?: number;
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}
