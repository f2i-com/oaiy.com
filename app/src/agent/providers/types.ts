/** Provider configuration, as softn Studio's (`types/studio.ts`) defines it. */

export type ProviderType = 'anthropic' | 'openai' | 'local' | 'custom';

export type LocalServerKind = 'ollama' | 'lmstudio' | 'other';

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
}

export interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
}
