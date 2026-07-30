// OAIY Engine Type Definitions

/**
 * Supported node types in the OAIY workflow builder
 */
/** Built-in node types */
export type BuiltinNodeType =
  | 'input_text'
  | 'input_file'
  | 'input_folder'
  | 'file_read'
  | 'file_write'
  | 'text_chunker'
  | 'ai_llm'
  | 'logic_block'
  | 'memory'
  | 'template'
  | 'image_gen'
  | 'image_view'
  | 'image_save'
  | 'image_combiner'
  | 'loop_start'
  | 'loop_end'
  | 'condition'
  | 'subflow'
  | 'browser_session'
  | 'browser_request'
  | 'browser_extract'
  | 'browser_control'
  | 'database'
  | 'video_frame_extractor'
  | 'output'
  // Macro nodes
  | 'macro'
  | 'macro_input'
  | 'macro_output'
  // Video nodes
  | 'video_gen'
  | 'video_append'
  | 'video_avatar'
  | 'video_pip'
  | 'video_save'
  | 'video_captions'
  | 'extend_videos'
  // Audio nodes
  | 'text_to_speech'
  | 'music_gen'
  | 'audio_mixer'
  | 'audio_append'
  // Utility nodes
  | 'comfyui_free_memory'
  // Terminal nodes
  | 'terminal_ai_control';

/**
 * Node type for workflow nodes
 */
export type NodeType = BuiltinNodeType;

/**
 * Represents a node in the workflow graph
 */
export interface GraphNode {
  id: string;
  type: NodeType;
  data: Record<string, unknown>;
  /** Optional position for UI persistence */
  position?: { x: number; y: number };
}

/**
 * Represents a connection between nodes
 */
export interface GraphEdge {
  id?: string;
  source: string;
  target: string;
  sourceHandle?: string;
  targetHandle?: string;
}

/**
 * Complete workflow graph structure
 */
export interface WorkflowGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
}

/**
 * Log entry for the execution console
 */
export interface LogEntry {
  id: string;
  source: string;
  message: string;
  timestamp: number;
  isStreaming?: boolean;
  type?: 'info' | 'error' | 'success' | 'node';
}

/**
 * Callback type for streaming token updates
 */
export type StreamCallback = (nodeId: string, token: string) => void;

/**
 * Callback type for log messages
 */
export type LogCallback = (entry: Omit<LogEntry, 'id' | 'timestamp'>) => void;

/**
 * Callback type for image preview updates (real-time during execution)
 */
export type ImageCallback = (nodeId: string, imageUrl: string) => void;

/**
 * Callback type for node execution status updates
 */
export type NodeStatusCallback = (nodeId: string, status: 'running' | 'completed' | 'error') => void;

/**
 * Database operation types
 */
export type DatabaseOperation = 'insert' | 'query' | 'update' | 'delete' | 'raw_sql';
export type DatabaseStorageType = 'collection' | 'table';

/**
 * Column definition for table schema
 */
export interface ColumnDefinition {
  name: string;
  type: 'TEXT' | 'INTEGER' | 'REAL' | 'BLOB' | 'JSON';
  primaryKey?: boolean;
  nullable?: boolean;
}

/**
 * Column mapping for transforming input data to table columns
 */
export interface ColumnMapping {
  sourceField: string;
  targetColumn: string;
}

/**
 * Database operation request
 */
export interface DatabaseRequest {
  operation: DatabaseOperation;
  storageType: DatabaseStorageType;
  collectionName?: string;
  tableName?: string;
  data?: Record<string, unknown> | Record<string, unknown>[];
  filter?: Record<string, unknown>;
  whereClause?: string;
  rawSql?: string;
  params?: (string | number | null)[];
  limit?: number;
  /** Auto-create table if it doesn't exist (for table storage type) */
  autoCreateTable?: boolean;
  /** Schema definition for auto-creating tables */
  tableSchema?: ColumnDefinition[];
  /** Column mappings to transform input data keys to table column names */
  columnMappings?: ColumnMapping[];

  // =============================================================================
  // Per-flow database context (added for flow-scoped data isolation)
  // =============================================================================

  /**
   * Flow ID for per-flow database isolation
   * When provided, operations use the flow's dedicated database file
   * When omitted, operations use the legacy shared database
   */
  flowId?: string;

  /**
   * Package ID if the flow is from a loaded package
   * Used with flowId to locate the package-specific database
   */
  packageId?: string;
}

/**
 * Database operation result
 */
export interface DatabaseResult {
  success: boolean;
  data?: Record<string, unknown>[];
  insertedId?: string | number;
  rowsAffected?: number;
  error?: string;
}

/**
 * Callback type for database operations
 */
export type DatabaseCallback = (request: DatabaseRequest) => Promise<DatabaseResult>;

/**
 * Node configuration for each node type
 */
export interface AILLMNodeData {
  model: string;
  systemPrompt: string;
  endpointId?: string; // Reference to LLMEndpoint.id for per-node endpoint selection
}

export interface HttpActionNodeData {
  method: 'GET' | 'POST';
  url: string;
  body?: string;
}

export interface LogicBlockNodeData {
  code: string;
}

export interface MemoryNodeData {
  mode: 'read' | 'write';
  key: string;
}

export interface OutputNodeData {
  label?: string;
}

export interface ImageGenNodeData {
  endpoint?: string;
  apiFormat?: 'openai' | 'gemini-flash' | 'gemini-2-flash' | 'gemini-3-pro' | 'comfyui' | 'custom';
  model?: string;
  size?: string;
  quality?: string;
  outputFormat?: string;
  background?: string;
  aspectRatio?: string;
  apiKey?: string;
  headers?: string;
}

export interface ImageViewNodeData {
  imageUrl?: string;
  label?: string;
}

export interface ImageSaveNodeData {
  imageUrl?: string;
  filename?: string;
  format?: 'png' | 'jpg' | 'webp';
}

export interface ImageCombinerNodeData {
  inputCount?: number;
}

export interface TemplateNodeData {
  template?: string;
  inputCount?: number;
  inputNames?: string[];
}

export interface LoopStartNodeData {
  iterations?: number;
}

export interface LoopEndNodeData {
  collectedResults?: unknown[];
}

export interface ConditionNodeData {
  operator?: 'equals' | 'not_equals' | 'contains' | 'not_contains' | 'greater' | 'less' | 'greater_eq' | 'less_eq' | 'is_empty' | 'not_empty';
  compareValue?: string;
}

// ============================================
// File System Node Types
// ============================================

/**
 * File information returned from folder scanning
 */
export interface FileInfo {
  path: string;
  name: string;
  nameWithoutExt: string;
  ext: string;
  size: number;
  modifiedAt: string;
  isDirectory: boolean;
}

/**
 * Folder input node - scans a folder and outputs file list
 */
export interface FolderInputNodeData {
  path?: string;
  recursive?: boolean;
  includePatterns?: string;  // Comma-separated patterns like "*.png, *.jpg"
  excludePatterns?: string;
  maxFiles?: number;
}

/**
 * File read node - reads file contents
 */
export interface FileReadNodeData {
  readAs?: 'text' | 'base64';
}

/**
 * File write node - writes content to a file
 */
export interface FileWriteNodeData {
  targetPath?: string;  // Template string with {{name}}, {{ext}}, etc.
  contentType?: 'text' | 'base64';
  createDirectories?: boolean;
  overwrite?: boolean;
}

/**
 * File reference for large files (content not loaded into memory)
 * Used when file size exceeds MAX_IN_MEMORY_SIZE (10MB)
 */
export interface FileRef {
  __type: 'file_ref';
  path: string;
  size: number;
  name: string;
  nameWithoutExt: string;
  ext: string;
}

/**
 * Chunk reference for streaming large text files
 * Points to a specific byte range in a file
 */
export interface ChunkRef {
  __type: 'chunk_ref';
  path: string;
  start: number;
  length: number;
  index: number;
  total: number;
}

/**
 * Text chunker node - splits large text/files into processable chunks
 */
export interface TextChunkerNodeData {
  chunkSize?: number;    // Characters per chunk (default: 2000)
  overlap?: number;      // Overlap between chunks (default: 200)
}

// ============================================
// Video Processing Types
// ============================================

/**
 * Information about an extracted video frame
 */
export interface FrameInfo {
  index: number;
  timestamp: number;
  path: string;
  dataUrl: string;
}

/**
 * Video metadata
 */
export interface VideoInfo {
  duration: number;
  width: number;
  height: number;
  fps: number;
  codec: string;
  format: string;
}

/**
 * Video frame extractor node - extracts frames from video files
 */
export interface VideoFrameExtractorNodeData {
  intervalSeconds?: number;  // Extract 1 frame every N seconds (default: 1.0)
  startTime?: number;        // Start timestamp in seconds (default: 0)
  endTime?: number;          // End timestamp in seconds (0 = full video)
  maxFrames?: number;        // Maximum frames to extract (default: 100)
  outputFormat?: 'png' | 'jpeg';  // Output image format (default: jpeg)
  scaleWidth?: number;       // Optional resize width
  scaleHeight?: number;      // Optional resize height
  batchSize?: number;        // Frames per batch for memory-efficient processing (0 = all at once)
}

/**
 * Result from batch frame extraction
 */
export interface BatchExtractResult {
  frames: FrameInfo[];
  batchIndex: number;
  totalBatches: number;
  totalFrames: number;
  hasMore: boolean;
  nextStartTime: number;
}

// ============================================
// Browser Node Types
// ============================================

export type BrowserProfile =
  | 'chrome_windows'
  | 'chrome_mac'
  | 'firefox_windows'
  | 'firefox_mac'
  | 'safari_mac'
  | 'edge_windows'
  | 'mobile_ios'
  | 'mobile_android'
  | 'custom';

export interface BrowserSessionNodeData {
  browserProfile?: BrowserProfile;
  customUserAgent?: string;
  customHeaders?: string;
  initialCookies?: string;
}

export interface BrowserRequestNodeData {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE';
  url?: string;
  bodyType?: 'none' | 'form_urlencoded' | 'json' | 'multipart' | 'raw';
  body?: string;
  responseFormat?: 'html' | 'json' | 'text' | 'full';
  followRedirects?: boolean;
  maxRedirects?: number;
}

export interface BrowserExtractNodeData {
  extractionType?: 'css_selector' | 'regex' | 'all_links' | 'all_forms' | 'form_fields';
  selector?: string;
  pattern?: string;
  extractTarget?: 'text' | 'html' | 'attribute';
  attributeName?: string;
  outputFormat?: 'first' | 'all_json' | 'all_newline';
}

export type BrowserControlAction = 'click' | 'type' | 'scroll' | 'screenshot' | 'evaluate' | 'wait';

export interface BrowserControlNodeData {
  action?: BrowserControlAction;
  selector?: string;
  value?: string;
  scrollDirection?: 'up' | 'down' | 'left' | 'right';
  scrollAmount?: number;
  waitTimeout?: number;
}

/**
 * Workflow execution result
 */
export interface ExecutionResult {
  success: boolean;
  output?: unknown;
  error?: string;
  logs: LogEntry[];
}

// ============================================
// Project & Flow Management Types
// ============================================

/**
 * Macro port definition (for inputs/outputs of a macro)
 */
export interface MacroPortDefinition {
  id: string;           // Node ID of the macro_input/macro_output node
  name: string;         // User-defined port name
  type: string;         // Data type (any, text, image, video, etc.)
  required?: boolean;   // For inputs only
  defaultValue?: string; // For inputs only
}

/**
 * Macro metadata (extracted from macro_input/macro_output nodes)
 */
export interface MacroMetadata {
  inputs: MacroPortDefinition[];
  outputs: MacroPortDefinition[];
  description?: string;
  icon?: string;
  color?: string;
}

/**
 * A single flow within a project
 */
export interface Flow {
  id: string;
  name: string;
  description?: string;
  tags?: string[];
  createdAt: string;
  updatedAt: string;
  graph: WorkflowGraph;
  localOnly?: boolean;
  /** If true, this flow is a macro and can be used as a node in other flows */
  isMacro?: boolean;
  /** Macro-specific metadata (only present if isMacro is true) */
  macroMetadata?: MacroMetadata;
  /** If true, this is a demo/example flow (shown in Demos category) */
  isDemo?: boolean;
  /** If true, this is a built-in macro (loaded from /macros folder) */
  isBuiltIn?: boolean;
}

/**
 * LLM endpoint configuration
 */
export interface LLMEndpoint {
  id: string;
  name: string;
  description?: string;
  endpoint: string;
  model?: string;
  apiKeyEnvVar?: string;
  requestFormat?: 'openai' | 'anthropic' | 'ollama';
  isLocal?: boolean;
}

/**
 * Image generation endpoint configuration
 */
export interface ImageGenEndpoint {
  id: string;
  name: string;
  description?: string;
  endpoint: string;
  apiKeyEnvVar?: string;
  isLocal?: boolean;
  /** API format determines how requests are structured */
  apiFormat?: 'openai' | 'gemini-flash' | 'gemini-2-flash' | 'gemini-3-pro' | 'comfyui' | 'custom';
  /** Default model for this endpoint */
  model?: string;
  /** Default size setting */
  defaultSize?: string;
  /** Default quality setting if applicable */
  defaultQuality?: string;
}


/**
 * HTTP preset for common API integrations
 */
export interface HttpPreset {
  id: string;
  name: string;
  description?: string;
  baseUrl: string;
  defaultHeaders?: Record<string, string>;
  authType?: 'none' | 'bearer' | 'api-key' | 'basic';
  authHeaderName?: string;
  apiKeyEnvVar?: string;
}

/**
 * A named constant/secret for use in workflows
 * These can be referenced by name in node configurations
 */
export interface ProjectConstant {
  id: string;
  name: string;
  /** The constant name as it appears in autocomplete (e.g., OPENAI_API_KEY) */
  key: string;
  /** The actual value (stored securely) */
  value: string;
  /** Category for organization */
  category: 'api_key' | 'endpoint' | 'model' | 'custom';
  /** Optional description */
  description?: string;
  /** If true, value should be masked in UI */
  isSecret?: boolean;
}

/**
 * Provider type for AI/LLM nodes
 */
export type AIProvider = 'openai' | 'anthropic' | 'google' | 'ollama' | 'lmstudio' | 'openrouter' | 'groq' | 'huggingface' | 'oaiy-local' | 'custom';

/**
 * Provider type for Image Generation nodes
 */
export type ImageProvider = 'openai' | 'gemini' | 'gemini-3-pro' | 'gemini-flash' | 'gemini-2-flash' | 'wan2gp' | 'comfyui' | 'custom';

/**
 * Per-provider data-flow descriptors. OAIY makes a deliberate effort to
 * tell the user *which data leaves their machine* before they pick a
 * provider — the badges rendered from this metadata appear in the AI
 * settings UI alongside every provider option.
 *
 * Categories:
 *   * `local-only`     — runs entirely on the user's hardware, no
 *                         outbound network calls beyond the local host.
 *   * `prompt`         — text prompts/messages cross the boundary.
 *   * `files`          — attached files (PDFs, audio, video, etc.) cross.
 *   * `images`         — base64-encoded images / vision inputs cross.
 *   * `screenshots`    — screenshots taken by the runtime cross.
 *   * `terminal`       — terminal session output may be sent to the model.
 *   * `browser`        — browser-automation page content may be sent.
 */
export type AIProviderDataFlow =
  | 'local-only'
  | 'prompt'
  | 'files'
  | 'images'
  | 'screenshots'
  | 'terminal'
  | 'browser';

/**
 * Static metadata about each AI provider, used to render privacy badges
 * in the settings UI. Local providers (Ollama, LM Studio) do not list
 * any other categories — by definition they don't send anything off-box.
 */
export interface AIProviderInfo {
  id: AIProvider;
  /** Human-readable name for the settings UI. */
  label: string;
  /** Two-line description of what the provider is. */
  description: string;
  /** Categories that flow OUT of the user's machine when this provider runs. */
  dataFlows: readonly AIProviderDataFlow[];
  /** True when the provider keeps everything on the user's hardware. */
  isLocal: boolean;
}

export const AI_PROVIDER_INFO: Readonly<Record<AIProvider, AIProviderInfo>> = {
  openai: {
    id: 'openai',
    label: 'OpenAI',
    description: 'Cloud LLM provider (api.openai.com).',
    dataFlows: ['prompt', 'images', 'files'],
    isLocal: false,
  },
  anthropic: {
    id: 'anthropic',
    label: 'Anthropic',
    description: 'Claude family. Cloud (api.anthropic.com).',
    dataFlows: ['prompt', 'images', 'files'],
    isLocal: false,
  },
  google: {
    id: 'google',
    label: 'Google Gemini',
    description: 'Cloud (generativelanguage.googleapis.com).',
    dataFlows: ['prompt', 'images', 'files'],
    isLocal: false,
  },
  openrouter: {
    id: 'openrouter',
    label: 'OpenRouter',
    description: 'Cloud aggregator that proxies many model providers.',
    dataFlows: ['prompt', 'images', 'files'],
    isLocal: false,
  },
  groq: {
    id: 'groq',
    label: 'Groq',
    description: 'Cloud (api.groq.com).',
    dataFlows: ['prompt'],
    isLocal: false,
  },
  huggingface: {
    id: 'huggingface',
    label: 'Hugging Face',
    description: 'Cloud inference endpoints (api-inference.huggingface.co).',
    dataFlows: ['prompt', 'images', 'files'],
    isLocal: false,
  },
  ollama: {
    id: 'ollama',
    label: 'Ollama',
    description: 'Local model server (default http://localhost:11434).',
    dataFlows: ['local-only'],
    isLocal: true,
  },
  lmstudio: {
    id: 'lmstudio',
    label: 'LM Studio',
    description: 'Local model server (default http://localhost:1234).',
    dataFlows: ['local-only'],
    isLocal: true,
  },
  'oaiy-local': {
    id: 'oaiy-local',
    label: 'OAIY Local (GGUF, Rust)',
    description: 'Pure-Rust llama_cpp_rewrite engine running locally — no network. CUDA accelerated on supported GPUs.',
    dataFlows: ['local-only'],
    isLocal: true,
  },
  custom: {
    id: 'custom',
    label: 'Custom endpoint',
    description: 'A user-configured endpoint URL — verify whether it is local before sharing sensitive data.',
    dataFlows: ['prompt', 'images', 'files'],
    isLocal: false,
  },
};

/**
 * Site-wide default settings for AI and Image Generation nodes
 */
export interface ProjectSettings {
  /** Default AI provider for new AI/LLM nodes */
  defaultAIProvider?: AIProvider;
  /** Default endpoint URL for AI (can be overridden per node) */
  defaultAIEndpoint?: string;
  /** Default model for AI (can be overridden per node) */
  defaultAIModel?: string;
  /** Default API key constant name for AI (e.g., "OPENAI_API_KEY") */
  defaultAIApiKeyConstant?: string;

  /** Ollama server endpoint (default: http://localhost:11434) */
  ollamaEndpoint?: string;
  /** LM Studio server endpoint (default: http://localhost:1234) */
  lmstudioEndpoint?: string;

  /** Default provider for new Image Gen nodes */
  defaultImageProvider?: ImageProvider;
  /** Default endpoint URL for Image Gen */
  defaultImageEndpoint?: string;
  /** Default model for Image Gen */
  defaultImageModel?: string;
  /** Default API key constant name for Image Gen */
  defaultImageApiKeyConstant?: string;

  /** Default provider for new Video Gen nodes */
  defaultVideoProvider?: string;
  /** Default endpoint URL for Video Gen */
  defaultVideoEndpoint?: string;
  /** Default model for Video Gen */
  defaultVideoModel?: string;

  // ============================================
  // Default Service (unified Services registry)
  // ============================================
  //
  // Picked in Settings → Providers, referenced from a node's `service`
  // field when the node leaves it blank. Single source of truth for
  // "what engine should new nodes of this kind talk to" — replaces
  // the per-category provider/endpoint/model triplets above.
  /** Default service ID for new AI LLM nodes (from oaiy.customServices + built-ins) */
  defaultAIServiceId?: string;
  /** Default service ID for new Image Gen nodes */
  defaultImageServiceId?: string;
  /** Default service ID for new Video Gen nodes */
  defaultVideoServiceId?: string;

  // ============================================
  // Local Network Security Settings
  // ============================================

  /**
   * Override switch - if true, allows ALL local network requests without prompts
   * Use with caution: disables all local network security checks
   */
  allowAllLocalNetwork?: boolean;

  /**
   * Whitelist of allowed local network addresses
   * Format: "hostname:port" or "ip:port" (e.g., "localhost:11434", "192.168.1.100:8188")
   * Requests to these addresses are allowed without prompting
   */
  localNetworkWhitelist?: string[];

  // ============================================
  // App Data Settings
  // ============================================

  /**
   * Custom path to the app data directory (root folder for all OAIY data)
   * Plugins are stored in {appDataPath}/plugins
   * If not set, uses the default: %APPDATA%/oaiy (Windows) or ~/.local/share/oaiy (Linux)
   * Setting this allows data persistence between app installs/upgrades
   */
  appDataPath?: string;

  /**
   * @deprecated Use appDataPath instead. This is kept for backwards compatibility.
   */
  pluginsPath?: string;

  // ============================================
  // UI Settings
  // ============================================

  /**
   * Show demo/example flows in the sidebar (default: true)
   */
  showDemoFlows?: boolean;

  // ============================================
  // Service Lifecycle Settings
  // ============================================

  /**
   * Default idle timeout for services in seconds (0 = never auto-stop)
   * Services that are idle for this duration will be automatically stopped.
   * Default: 900 (15 minutes)
   */
  serviceIdleTimeoutSecs?: number;

  /**
   * Per-service idle timeout overrides
   * Key is service ID (e.g., "playwright-browser"), value is timeout in seconds
   * Use 0 to disable auto-stop for a specific service
   */
  serviceIdleTimeoutOverrides?: Record<string, number>;

  /**
   * Maximum time to wait for a service to start and become healthy (in seconds)
   * Default: 60
   */
  serviceStartupTimeoutSecs?: number;

  // ============================================
  // Log Privacy Settings
  // ============================================

  /**
   * How filesystem paths appear in workflow logs.
   *
   *   * `full`     — print the path verbatim (no privacy).
   *   * `basename` — only the file name, drop every directory component.
   *   * `tilde`    — keep the layout but rewrite `$HOME` as `~/…` and
   *                  truncate everything else to its basename. **Default.**
   *   * `none`     — replace the path with `<path>`.
   *
   * Affects log output only; the underlying file operations always use
   * the real path. Adjust this when sharing logs / screen-recordings of
   * workflows that touch named files.
   */
  logPathPolicy?: LogPathPolicy;
}

/**
 * Privacy policy for filesystem paths in workflow logs. See
 * `ProjectSettings.logPathPolicy`.
 */
export type LogPathPolicy = 'full' | 'basename' | 'tilde' | 'none';

/**
 * A OAIY project containing multiple flows
 */
export interface OAIYProject {
  version: string;
  name: string;
  description?: string;
  createdAt: string;
  updatedAt: string;
  flows: Flow[];
  llmEndpoints: LLMEndpoint[];
  imageGenEndpoints: ImageGenEndpoint[];
  httpPresets: HttpPreset[];
  /** Project-level constants/secrets */
  constants?: ProjectConstant[];
  /** Site-wide default settings */
  settings?: ProjectSettings;
  /**
   * Set when the project was IMPORTED from an untrusted source (a shared `?flow=` link or
   * AI-designer output). The web run path runs such projects HARDENED (worker isolation +
   * permission gate, no code:execute/secrets:read) so a malicious imported flow can't run
   * arbitrary JS / steal API keys in the user's origin. Absent/false = the user's own (trusted)
   * project. Set by the importer (App.tsx), never copied from the imported flow's own data.
   */
  untrusted?: boolean;
}

// ============================================
// Run History & Observability Types
// ============================================

/**
 * Status of a single node execution
 */
export interface NodeRunStatus {
  nodeId: string;
  nodeType: NodeType;
  status: 'pending' | 'running' | 'completed' | 'error' | 'skipped';
  startedAt?: number;
  completedAt?: number;
  input?: unknown;
  output?: unknown;
  error?: string;
}

/**
 * A single run execution record
 */
export interface RunRecord {
  id: string;
  flowId: string;
  flowName: string;
  startedAt: number;
  completedAt?: number;
  duration?: number;
  status: 'running' | 'completed' | 'error' | 'aborted';
  inputs?: Record<string, unknown>;
  outputs?: Record<string, unknown>;
  nodeStatuses: NodeRunStatus[];
  logs: LogEntry[];
}

/**
 * Subflow node data
 */
export interface SubflowNodeData {
  flowId?: string;
  flowName?: string;
  inputMapping?: Record<string, string>;
}

/**
 * Input data that can be passed to a workflow
 */
export interface WorkflowInputs {
  [key: string]: string | number | boolean | object;
}

// ============================================
// Local Network Permission Types
// ============================================

/**
 * Request for local network access permission
 */
export interface LocalNetworkPermissionRequest {
  /** The URL being accessed */
  url: string;
  /** The host:port being accessed (e.g., "localhost:11434") */
  hostPort: string;
  /** The node ID making the request (for context) */
  nodeId?: string;
  /** Optional description of what the request is for */
  purpose?: string;
}

/**
 * User's response to a local network permission request
 */
export interface LocalNetworkPermissionResponse {
  /** Whether access is allowed */
  allowed: boolean;
  /** If true, add to whitelist for future requests */
  remember: boolean;
}

/**
 * Callback type for local network permission requests
 * Returns a Promise that resolves when the user makes a choice
 */
export type LocalNetworkPermissionCallback = (
  request: LocalNetworkPermissionRequest
) => Promise<LocalNetworkPermissionResponse>;
