/**
 * .oaiy Package Format Types
 *
 * Defines the schema for portable workflow packages that include
 * flows, custom nodes, services, and assets in a single distributable file.
 */

import type { Flow } from './types';
import type { ModuleManifest, NodeDefinition } from './module-types';

// =============================================================================
// Package Manifest
// =============================================================================

/**
 * Format version for the .oaiy package specification.
 * Increment when breaking changes are made to the manifest schema.
 */
export type PackageFormatVersion = '1.0';

/**
 * Permissions that a package can request.
 * These are shown to the user in the trust dialog before installation.
 *
 * Permissions are split by *capability surface* so the user can see exactly
 * what the package can do, and so the runtime can refuse high-impact actions
 * for untrusted packages.
 */
export type PackagePermission =
  | 'filesystem'        // Read/write access to user's filesystem
  | 'filesystem:read'   // Read-only filesystem access
  | 'process:execute'   // Spawn arbitrary OS processes via Filesystem.runCommand
  | 'network'           // Outbound network requests (public + local with prompt)
  | 'network:local'     // Local-network requests (e.g. localhost services)
  | 'clipboard'         // Access to system clipboard
  | 'notifications'     // Show system notifications
  | 'camera'            // Access to camera (for video capture)
  | 'microphone'        // Access to microphone (for audio recording)
  | 'terminal'          // Spawn shells, send keystrokes, screenshot terminals
  | 'browser'           // Drive headless / native browsers
  | 'service:start'     // Start subprocess services declared by the package
  | 'code:execute'      // Run JS in logic_block / condition expressions / new Function()
  | 'ui:custom'         // Render the package's own React UI components
  | 'secrets:read'      // Read user-stored secrets (API keys, etc.)
  | 'constants:read';   // Read project constants other than secrets

/**
 * System dependencies that the package requires.
 * The package loader will check for these before running.
 */
export interface SystemDependency {
  /** Name of the dependency (e.g., "python", "ffmpeg", "cuda") */
  name: string;
  /** Minimum version required (semver) */
  minVersion?: string;
  /** Whether this dependency is optional (package can run without it) */
  optional?: boolean;
  /** URL to download/install instructions */
  installUrl?: string;
}

/**
 * Custom UI component mapping for a package node
 */
export interface PackageNodeUIComponent {
  /** Node type ID (without package prefix) */
  nodeType: string;
  /** Path to the UI component file within the package (relative to module path) */
  componentPath: string;
  /** Export name of the component (default: 'default') */
  exportName?: string;
}

/**
 * Reference to a custom node module included in the package.
 */
export interface PackageNodeModule {
  /** Relative path to the module directory within the package */
  path: string;
  /** Module manifest (inline or path to module.json) */
  manifest?: ModuleManifest;
  /** Whether this module overrides a built-in module */
  overridesBuiltIn?: string;
  /** Custom prefix for node IDs (default: package ID) */
  prefix?: string;
  /**
   * Custom UI components for nodes in this module.
   * Note: Custom components require the package to be trusted.
   * If not specified, nodes use GenericNode for rendering.
   */
  uiComponents?: PackageNodeUIComponent[];
}

/**
 * Reference to a service included in the package.
 */
export interface PackageService {
  /** Unique service ID within the package */
  id: string;
  /** Relative path to the service directory */
  path: string;
  /** Display name for the service */
  name?: string;
  /** Whether to start this service automatically when the package opens */
  autoStart?: boolean;
  /** Preferred port (if not available, a free port will be assigned) */
  preferredPort?: number;
  /** Environment variables to set for this service */
  env?: Record<string, string>;
}

/**
 * Isolation settings for the package sandbox.
 */
export interface PackageIsolation {
  /** Whether to run in sandboxed mode (restricted filesystem access) */
  sandboxed: boolean;
  /** Paths that the package is allowed to access (relative or special tokens) */
  allowedPaths?: string[];
  /** Whether to allow outbound network requests */
  networkAccess?: boolean;
  /** Whether to allow inter-process communication with other packages */
  ipcAllowed?: boolean;
}

/**
 * Main manifest file for a .oaiy package.
 * This is the root configuration that defines the package contents.
 */
export interface OAIYPackageManifest {
  // -------------------------------------------------------------------------
  // Identification
  // -------------------------------------------------------------------------

  /** Format version for compatibility checking */
  formatVersion: PackageFormatVersion;

  /** Unique package identifier (reverse domain style: com.example.my-package) */
  id: string;

  /** Human-readable package name */
  name: string;

  /** Package version (semver) */
  version: string;

  /** Short description of what the package does */
  description?: string;

  /** Package author name or organization */
  author?: string;

  /** SPDX license identifier */
  license?: string;

  /** Homepage or repository URL */
  homepage?: string;

  /** Package icon (relative path to image or icon name) */
  icon?: string;

  /** Keywords for search/discovery */
  tags?: string[];

  // -------------------------------------------------------------------------
  // Entry Points
  // -------------------------------------------------------------------------

  /** Path to the main flow to open when the package runs */
  entryFlow: string;

  /** Optional startup script to run before the flow */
  startupScript?: string;

  // -------------------------------------------------------------------------
  // Contents
  // -------------------------------------------------------------------------

  /** Paths to flow JSON files included in the package */
  flows: string[];

  /** Custom node modules included in the package */
  nodes?: PackageNodeModule[];

  /** Services included in the package */
  services?: PackageService[];

  /** Static assets (images, samples, etc.) */
  assets?: string[];

  /** Paths to macro flow JSON files included in the package */
  macros?: string[];

  // -------------------------------------------------------------------------
  // Dependencies
  // -------------------------------------------------------------------------

  /** Package dependencies */
  dependencies?: {
    /** Minimum OAIY version required to run this package */
    oaiyVersion?: string;
    /** Built-in modules that must be available */
    modules?: string[];
    /** System-level dependencies (Python, ffmpeg, etc.) */
    system?: SystemDependency[];
    /** Other .oaiy packages this package depends on */
    packages?: Array<{
      id: string;
      version?: string;
    }>;
  };

  // -------------------------------------------------------------------------
  // Permissions & Security
  // -------------------------------------------------------------------------

  /** Permissions requested by the package */
  permissions?: PackagePermission[];

  /** Isolation/sandbox settings */
  isolation?: PackageIsolation;

  /**
   * SHA-256 hash of the package contents. Required for signed packages
   * and used by the installer to detect tampering before extraction.
   */
  contentHash?: string;

  /**
   * Detached Ed25519 signature over `contentHash`, base64-encoded. The
   * desktop installer verifies this against the trusted publisher keys
   * configured in `<app-data>/oaiy/publisher_keys.json`. A signed package
   * whose signature does not match a trusted key will fail to install
   * (we do not silently downgrade to "Trusted").
   */
  signature?: string;
}

// =============================================================================
// Installed Package State
// =============================================================================

/**
 * Trust level for an installed package.
 */
export type PackageTrustLevel =
  | 'untrusted'   // Not yet reviewed by user
  | 'trusted'     // User has approved this package
  | 'verified'    // Ed25519 signature verified against a trusted publisher key
  | 'blocked';    // User has explicitly blocked this package

/**
 * Status of an installed package.
 */
export type PackageStatus =
  | 'installed'   // Package is installed and ready
  | 'running'     // Package is currently active
  | 'updating'    // Package is being updated
  | 'error';      // Package has an error

/**
 * Information about an installed package.
 */
export interface InstalledPackage {
  /** Package manifest */
  manifest: OAIYPackageManifest;

  /** Path to the extracted package directory */
  installPath: string;

  /** Original .oaiy file path (if retained) */
  sourcePath?: string;

  /** Installation timestamp */
  installedAt: string;

  /** Last time the package was opened/run */
  lastRunAt?: string;

  /** Current status */
  status: PackageStatus;

  /** User's trust decision */
  trustLevel: PackageTrustLevel;

  /** Permissions the user has granted (may be subset of requested) */
  grantedPermissions: PackagePermission[];

  /** Running services for this package */
  runningServices: string[];

  /** Error message if status is 'error' */
  error?: string;
}

// =============================================================================
// Package Operations
// =============================================================================

/**
 * Options for creating a .oaiy package.
 */
export interface CreatePackageOptions {
  /** Package manifest configuration */
  manifest: Omit<OAIYPackageManifest, 'formatVersion' | 'contentHash'>;

  /** Flows to include (will be written to flows/ directory) */
  flows: Flow[];

  /** Macros to include (will be written to macros/ directory) */
  macros?: Flow[];

  /** Custom node modules to include */
  nodeModules?: Array<{
    /** Source path to the module directory */
    sourcePath: string;
    /** Module manifest */
    manifest: ModuleManifest;
    /** Node definitions */
    nodes: NodeDefinition[];
  }>;

  /** Services to include */
  services?: Array<{
    /** Source path to the service directory */
    sourcePath: string;
    /** Service configuration */
    config: PackageService;
  }>;

  /** Additional assets to include */
  assets?: Array<{
    /** Source file path */
    sourcePath: string;
    /** Target path within package */
    targetPath: string;
  }>;

  /** Output path for the .oaiy file */
  outputPath: string;

  /** Whether to compress the package (default: true) */
  compress?: boolean;
}

/**
 * Result of creating a package.
 */
export interface CreatePackageResult {
  success: boolean;
  /** Path to the created .oaiy file */
  packagePath?: string;
  /** SHA-256 hash of the package contents */
  contentHash?: string;
  /** Size in bytes */
  size?: number;
  /** Error message if failed */
  error?: string;
}

/**
 * Options for installing a package.
 */
export interface InstallPackageOptions {
  /** Path to the .oaiy file */
  packagePath: string;

  /** Target installation directory (defaults to user packages directory) */
  installDir?: string;

  /** Skip the trust dialog (for pre-approved packages) */
  skipTrustDialog?: boolean;

  /** Permissions to grant automatically */
  grantPermissions?: PackagePermission[];

  /** Whether to start services automatically after install */
  autoStartServices?: boolean;
}

/**
 * Result of installing a package.
 */
export interface InstallPackageResult {
  success: boolean;
  /** Installed package info */
  package?: InstalledPackage;
  /** Whether the user cancelled the trust dialog */
  cancelled?: boolean;
  /** Error message if failed */
  error?: string;
}

/**
 * Options for opening/running a package.
 */
export interface OpenPackageOptions {
  /** Package ID or path to .oaiy file */
  packageIdOrPath: string;

  /** Input values to pass to the entry flow */
  inputs?: Record<string, unknown>;

  /** Whether to auto-start required services */
  autoStartServices?: boolean;

  /** Run in headless mode (no UI) */
  headless?: boolean;
}

/**
 * Information about a running package context.
 */
export interface PackageContext {
  /** The installed package */
  package: InstalledPackage;

  /** Currently loaded flows */
  flows: Flow[];

  /** Loaded macros (package-scoped, available only within this package) */
  macros: Flow[];

  /** Loaded custom nodes */
  nodes: Map<string, NodeDefinition>;

  /** Service status map (service_id -> port) */
  services: Map<string, number>;

  /** Sandbox file path mappings */
  pathMappings: Map<string, string>;
}

// =============================================================================
// Package Events
// =============================================================================

/**
 * Event types emitted during package operations.
 */
export type PackageEventType =
  | 'package:installing'
  | 'package:installed'
  | 'package:uninstalling'
  | 'package:uninstalled'
  | 'package:opening'
  | 'package:opened'
  | 'package:closing'
  | 'package:closed'
  | 'package:error'
  | 'package:service:starting'
  | 'package:service:started'
  | 'package:service:stopping'
  | 'package:service:stopped'
  | 'package:service:error';

/**
 * Package event payload.
 */
export interface PackageEvent {
  type: PackageEventType;
  packageId: string;
  serviceId?: string;
  error?: string;
  data?: unknown;
  timestamp: number;
}

/**
 * Event handler for package events.
 */
export type PackageEventHandler = (event: PackageEvent) => void;

// =============================================================================
// Validation
// =============================================================================

/**
 * Validation error for package manifest.
 */
export interface PackageValidationError {
  path: string;
  message: string;
  severity: 'error' | 'warning';
}

/**
 * Result of validating a package manifest.
 */
export interface PackageValidationResult {
  valid: boolean;
  errors: PackageValidationError[];
  warnings: PackageValidationError[];
}

/**
 * Validates a package manifest for correctness and completeness.
 */
export function validatePackageManifest(manifest: unknown): PackageValidationResult {
  const errors: PackageValidationError[] = [];
  const warnings: PackageValidationError[] = [];

  if (!manifest || typeof manifest !== 'object') {
    errors.push({
      path: '',
      message: 'Manifest must be an object',
      severity: 'error',
    });
    return { valid: false, errors, warnings };
  }

  const m = manifest as Record<string, unknown>;

  // Required fields
  if (!m.formatVersion) {
    errors.push({ path: 'formatVersion', message: 'formatVersion is required', severity: 'error' });
  } else if (m.formatVersion !== '1.0') {
    errors.push({ path: 'formatVersion', message: `Unknown formatVersion: ${m.formatVersion}`, severity: 'error' });
  }

  if (!m.id || typeof m.id !== 'string') {
    errors.push({ path: 'id', message: 'id is required and must be a string', severity: 'error' });
  } else if (!/^[a-z0-9.-]+$/i.test(m.id)) {
    warnings.push({ path: 'id', message: 'id should use reverse domain notation (e.g., com.example.my-package)', severity: 'warning' });
  }

  if (!m.name || typeof m.name !== 'string') {
    errors.push({ path: 'name', message: 'name is required and must be a string', severity: 'error' });
  }

  if (!m.version || typeof m.version !== 'string') {
    errors.push({ path: 'version', message: 'version is required and must be a string', severity: 'error' });
  }

  if (!m.entryFlow || typeof m.entryFlow !== 'string') {
    errors.push({ path: 'entryFlow', message: 'entryFlow is required and must be a string', severity: 'error' });
  }

  if (!m.flows || !Array.isArray(m.flows)) {
    errors.push({ path: 'flows', message: 'flows is required and must be an array', severity: 'error' });
  } else if (m.flows.length === 0) {
    errors.push({ path: 'flows', message: 'flows must contain at least one flow', severity: 'error' });
  }

  // Optional field validation
  if (m.permissions && !Array.isArray(m.permissions)) {
    errors.push({ path: 'permissions', message: 'permissions must be an array', severity: 'error' });
  }

  if (m.services && !Array.isArray(m.services)) {
    errors.push({ path: 'services', message: 'services must be an array', severity: 'error' });
  }

  return {
    valid: errors.length === 0,
    errors,
    warnings,
  };
}

// =============================================================================
// Embedded Content (for self-contained packages)
// =============================================================================

/**
 * Embedded asset data (base64 encoded)
 */
export interface EmbeddedAsset {
  /** Target path within the package */
  path: string;
  /** Asset type */
  type: 'image' | 'video' | 'audio' | 'file' | 'model' | 'font';
  /** Base64 encoded data */
  data: string;
  /** MIME type of the asset */
  mimeType?: string;
  /** Original file name */
  originalName?: string;
  /** File size in bytes (before encoding) */
  size?: number;
}

/**
 * Embedded macro definition (inline flow)
 */
export interface EmbeddedMacro {
  /** Macro ID */
  id: string;
  /** Macro name */
  name: string;
  /** Full flow definition */
  flow: Flow;
}

/**
 * Embedded custom node definition
 */
export interface EmbeddedCustomNode {
  /** Node type ID */
  id: string;
  /** Node display name */
  name: string;
  /** Node description */
  description?: string;
  /** Node category */
  category: string;
  /** Node icon (icon name or path) */
  icon?: string;
  /** Input definitions */
  inputs: Array<{
    id: string;
    name: string;
    type: string;
    required?: boolean;
    defaultValue?: unknown;
  }>;
  /** Output definitions */
  outputs: Array<{
    id: string;
    name: string;
    type: string;
  }>;
  /** Property definitions (configurable fields in node body) */
  properties?: Array<{
    id: string;
    name: string;
    type: 'text' | 'number' | 'boolean' | 'select' | 'textarea' | 'color' | 'code' | 'json' | 'secret';
    default?: unknown;
    options?: Array<{ value: string; label: string }>;
    min?: number;
    max?: number;
    step?: number;
    advanced?: boolean;
    group?: string;
  }>;
  /** Source TypeScript code */
  source: {
    /** Compiler code (TypeScript) */
    compiler: string;
    /** Runtime code (TypeScript) */
    runtime: string;
    /** UI component code (TSX) */
    ui?: string;
  };
  /** Compiled JavaScript code (populated after installation) */
  compiled?: {
    compiler: string;
    runtime: string;
    ui?: string;
  };
}

/**
 * Node extension definition - extends an existing node with additional functionality
 */
export interface NodeExtension {
  /** Extension ID (unique within package) */
  id: string;
  /** Display name for the extension */
  name: string;
  /** Description of what this extension adds */
  description?: string;
  /** The node type this extension applies to (e.g., 'ai_llm', 'text_input') */
  extends: string;
  /** Additional inputs to add to the node */
  additionalInputs?: Array<{
    id: string;
    name: string;
    type: string;
    required?: boolean;
    defaultValue?: unknown;
    /** Position in the inputs list (default: append at end) */
    position?: number;
  }>;
  /** Additional outputs to add to the node */
  additionalOutputs?: Array<{
    id: string;
    name: string;
    type: string;
  }>;
  /** Additional properties (UI fields) to add to the node */
  additionalProperties?: Array<{
    id: string;
    name: string;
    type: 'string' | 'number' | 'boolean' | 'select' | 'textarea' | 'code';
    defaultValue?: unknown;
    options?: Array<{ label: string; value: string }>;
    /** Property group (for organization in UI) */
    group?: string;
  }>;
  /** TypeScript source code for the extension */
  source: {
    /**
     * Compiler hook code (TypeScript)
     * Should export: preCompile?, postCompile?, or compile? functions
     */
    compilerHook?: string;
    /**
     * Runtime hook code (TypeScript)
     * Should export: preExecute?, postExecute?, or execute? functions
     */
    runtimeHook?: string;
    /**
     * UI extension code (TSX)
     * Should export a React component that receives extended node data
     */
    ui?: string;
  };
  /** Compiled JavaScript code (populated after installation) */
  compiled?: {
    compilerHook?: string;
    runtimeHook?: string;
    ui?: string;
  };
}

/**
 * Embedded node extensions in a package
 */
export interface EmbeddedNodeExtension extends NodeExtension {
  /** Package ID this extension belongs to */
  packageId?: string;
}

// =============================================================================
// Quick Export Types
// =============================================================================

/**
 * Options for quick export of a flow as a package
 */
export interface QuickExportOptions {
  /** Flow to export */
  flowId: string;
  /** Package name (defaults to flow name) */
  name?: string;
  /** Package version (defaults to 1.0.0) */
  version?: string;
  /** Package description */
  description?: string;
  /** Package author */
  author?: string;
  /** Whether to include referenced macros */
  includeMacros?: boolean;
  /** Whether to embed local assets (images, files) */
  embedAssets?: boolean;
  /** Whether to include project constants used by the flow */
  includeConstants?: boolean;
  /** Custom tags for the package */
  tags?: string[];
  /** Output path (defaults to desktop/downloads) */
  outputPath?: string;
}

/**
 * Result of quick export operation
 */
export interface QuickExportResult {
  /** Whether the export succeeded */
  success: boolean;
  /** Path to the exported .oaiy file */
  packagePath?: string;
  /** Package manifest */
  manifest?: OAIYPackageManifest;
  /** Number of flows included */
  flowCount?: number;
  /** Number of macros included */
  macroCount?: number;
  /** Number of assets embedded */
  assetCount?: number;
  /** Size of the package in bytes */
  size?: number;
  /** Error message if failed */
  error?: string;
  /** Warnings during export */
  warnings?: string[];
}

/**
 * Extended manifest that supports embedded content
 * This is used for self-contained packages that don't require file paths
 */
export interface OAIYPackageManifestWithEmbedded extends OAIYPackageManifest {
  /** Embedded macros (in addition to paths in 'macros' field) */
  embeddedMacros?: EmbeddedMacro[];
  /** Embedded assets (in addition to paths in 'assets' field) */
  embeddedAssets?: EmbeddedAsset[];
  /** Embedded custom node definitions */
  embeddedCustomNodes?: EmbeddedCustomNode[];
  /** Embedded node extensions (for extending existing nodes) */
  embeddedNodeExtensions?: EmbeddedNodeExtension[];
}

// =============================================================================
// Constants
// =============================================================================

/** File extension for OAIY packages */
export const PACKAGE_EXTENSION = '.oaiy';

/** MIME type for OAIY packages */
export const PACKAGE_MIME_TYPE = 'application/x-oaiy-package';

/** Current package format version */
export const CURRENT_FORMAT_VERSION: PackageFormatVersion = '1.0';

/** Default installation directory name */
export const PACKAGES_DIR_NAME = 'packages';

/** Manifest file name within a package */
export const MANIFEST_FILE_NAME = 'manifest.json';

/** Port range for package services */
export const PACKAGE_SERVICE_PORT_RANGE = {
  start: 8900,
  end: 8999,
} as const;
