// OAIY Core - Shared engine for workflow compilation and execution
// This package is used by oaiy-cli, oaiy-ui, and oaiy-desktop

// Compiler
export { OAIYCompiler } from './compiler';
export type { CompilerConfig } from './compiler';

// Template Compiler (for package nodes with compiler.template)
export { createTemplateCompiler, createSingleNodeTemplateCompiler } from './template-compiler';

// Runtime
export { createRuntime, OAIYRuntime } from './runtime';
export type { RuntimeConfig } from './runtime';

// Error Types (classes for typed error handling)
export {
  OAIYError,
  CompilationError,
  CycleDetectedError,
  UnknownNodeTypeError,
  InvalidLoopError,
  RuntimeError,
  AbortError,
  MissingInputError,
  ExternalApiError,
  ModuleError,
  ModuleValidationError,
  ModuleDependencyError,
  ModuleLoadError as ModuleLoadException,  // Renamed to avoid conflict with ModuleLoadError interface
  ValidationError as WorkflowValidationError,  // Renamed to avoid conflict with ValidationError interface
  // Type guards
  isOAIYError,
  isCompilationError,
  isRuntimeError,
  isModuleError,
} from './errors';

// User-Friendly Error Messages
export {
  formatErrorForUser,
  getErrorSummary,
  isUserFacingError,
} from './errors/user-messages';
export type { UserFriendlyError } from './errors/user-messages';

// Job Queue System
export { JobManager } from './queue/JobManager';
export type {
  Job,
  JobStatus,
  JobConfig,
  QueueMode,
  JobStateCallback,
  JobLogCallback,
  NodeStatusCallback as QueueNodeStatusCallback,
  StreamTokenCallback,
  ImageUpdateCallback,
  ActiveJobEntry,
  PendingAIRequest,
  AIYieldCallback,
} from './queue/types';
export type { JobManagerOptions, ExtendedModuleRegistry } from './queue/JobManager';

// Types
export type {
  NodeType,
  GraphNode,
  GraphEdge,
  WorkflowGraph,
  Flow,
  HttpPreset,
  LLMEndpoint,
  ImageGenEndpoint,
  OAIYProject,
  RunRecord,
  LogEntry,
  StreamCallback,
  LogCallback,
  ImageCallback,
  NodeStatusCallback,
  WorkflowInputs,
  DatabaseCallback,
  DatabaseRequest,
  DatabaseResult,
  DatabaseOperation,
  DatabaseStorageType,
  ProjectConstant,
  ProjectSettings,
  AIProvider,
  AIProviderInfo,
  AIProviderDataFlow,
  ImageProvider,
  LogPathPolicy,
  LocalNetworkPermissionRequest,
  LocalNetworkPermissionResponse,
  LocalNetworkPermissionCallback,
  MacroPortDefinition,
} from './types';

// AI provider metadata (used for the privacy badges in settings UI)
export { AI_PROVIDER_INFO } from './types';

// Browser utilities moved to core-browser plugin
// See: /plugins/core-browser/src/browser-utils.ts

// Value extraction helpers
export {
  extractNodeOutputs,
  extractDeepValue,
} from './value-utils';

// FlowPlan DSL (AI Flow Designer)
export type {
  FlowPlan,
  FlowPlanInput,
  FlowPlanCollection,
  FlowPlanLoop,
  FlowPlanStep,
  FlowPlanStepBase,
  FileReadStep,
  FileWriteStep,
  TemplateStep,
  AILLMStep,
  AIImageStep,
  ConditionStep,
  HTTPRequestStep,
  DatabaseStoreStep,
  LogStep,
  OutputStep,
  LogicBlockStep,
  GenericStep,
  FlowPlanValidationResult,
  FlowPlanValidationError,
  FlowPlanValidationWarning,
} from './flowplan';

export {
  validateFlowPlan,
  parseTemplateReferences,
  hasTemplateReferences,
  VALID_STEP_TYPES,
  CORE_STEP_TYPES,
  VALID_INPUT_TYPES,
  VALID_COLLECTION_TYPES,
} from './flowplan';

// FlowPlan Compiler
export type { FlowPlanCompilationResult, FlowPlanCompilerOptions } from './flowplan-compiler';
export { compileFlowPlan, layoutFlowPlanGraph } from './flowplan-compiler';

// FlowPlan Decompiler
export type { FlowPlanDecompilationResult } from './flowplan-decompiler';
export { decompileFlowPlan, summarizeFlowPlan } from './flowplan-decompiler';

// AI Flow Designer
export type {
  DesignerCapabilities,
  AIGenerationOptions,
  AIGenerationResult,
} from './ai-designer';

export {
  generateSystemPrompt,
  generateErrorCorrectionPrompt,
  parseAIResponse,
  generateFlowPlan,
  DEFAULT_CAPABILITIES,
  getDesignerModules,
} from './ai-designer';

// Module System
export type {
  ModuleManifest,
  ModuleCategory,
  ModulePermission,
  ModuleSettingDefinition,
  NodeDefinition,
  HandleDefinition,
  HandleDataType,
  HandlePosition,
  PropertyDefinition,
  PropertyType,
  PropertyOption,
  PropertyCondition,
  // CompilerConfig is already exported from ./compiler
  UIConfig,
  // RuntimeConfig is already exported from ./runtime
  RuntimeModule,
  RuntimeContext,
  LoadedModule,
  ModuleLoadResult,
  ModuleLoadError,
  ModuleRegistry,
  ModuleSettings,
  ModuleEvent,
  ModuleEventType,
  ModuleEventHandler,
  ValidationResult,
  ValidationError,
  CompilerContext,
  NodeUIMapping,
  ModuleUIConfig,
  ModuleCompiler,
  ModuleCompilerContext,
} from './module-types';

export {
  ModuleLoader,
  getModuleLoader,
  resetModuleLoader,
} from './module-loader';

export {
  ModuleDiscoveryService,
  TauriFileSystem,
  MemoryFileSystem,
  getModuleDiscovery,
  resetModuleDiscovery,
  loadBundledModules,
  type FileSystemAdapter,
  type DiscoveryOptions,
  type DiscoveredModule,
} from './module-discovery';

// Bundled Modules
export {
  BUNDLED_MODULES,
  registerBundledModules,
  getBundledModulesArray,
  getBundledManifests,
  getBundledNodeDefinitions,
  getBundledNodeDefinition,
  getBundledModule,
  getBundledUIRegistrations,
  getNodeTypeComponentMap,
  getComponentNameForNodeType,
  getCoreModuleIds,
  getModuleDirectories,
  getValidStepTypes,
  getFlowPlanEnabledNodes,
  getFlowPlanStepNodes,
  isValidStepType,
  MODULE_DIRECTORIES,
  type BundledModule,
  type ModuleUIRegistration,
} from './bundled-modules';

// ComfyUI Analyzer
export {
  analyzeComfyUIWorkflow,
  applyWorkflowOverrides,
  getWorkflowSummary,
  type ComfyUINode,
  type ComfyUIWorkflow,
  type ComfyUIAnalysis,
  type DetectedPromptInput,
  type DetectedImageInput,
  type DetectedOutput,
  type DetectedSeedNode,
} from './comfyui-analyzer';

// Logger
export {
  Logger,
  createLogger,
  compilerLogger,
  runtimeLogger,
  databaseLogger,
  moduleLogger,
  type LogLevel,
  type LogEntry as LoggerEntry, // Renamed to avoid conflict with types.ts LogEntry
  type LogHandler,
} from './logger';

// Metrics
export {
  metrics,
  MetricsCollector,
  type TimingStats,
  type MetricsSummary,
} from './metrics';

// Media Server Utilities
export {
  initMediaServerPort,
  initMediaServerToken,
  getMediaServerPort,
  getMediaServerToken,
  setMediaServerPort,
  setMediaServerToken,
  pathToMediaUrl,
  isLocalPath,
} from './media-utils';

// Package System (.oaiy format)
export type {
  PackageFormatVersion,
  PackagePermission,
  SystemDependency,
  PackageNodeModule,
  PackageNodeUIComponent,
  PackageService,
  PackageIsolation,
  OAIYPackageManifest,
  PackageTrustLevel,
  PackageStatus,
  InstalledPackage,
  CreatePackageOptions,
  CreatePackageResult,
  InstallPackageOptions,
  InstallPackageResult,
  OpenPackageOptions,
  PackageContext,
  PackageEventType,
  PackageEvent,
  PackageEventHandler,
  PackageValidationError,
  PackageValidationResult,
  // Embedded content types
  EmbeddedAsset,
  EmbeddedMacro,
  EmbeddedCustomNode,
  EmbeddedNodeExtension,
  // Package-level node extension (different from node-extension-types.ts)
  NodeExtension as PackageNodeExtension,
  // Quick export types
  QuickExportOptions,
  QuickExportResult,
  OAIYPackageManifestWithEmbedded,
} from './package-types';

// Flow Analyzer (for package export)
export type {
  FlowExportAnalysis,
  FlowAnalysisOptions,
  FlowAsset,
  FlowDependency,
  FlowWarning,
  AssetType,
} from './package/flow-analyzer';

export {
  analyzeFlowForExport,
  analyzeFlowsForExport,
} from './package/flow-analyzer';

export {
  validatePackageManifest,
  PACKAGE_EXTENSION,
  PACKAGE_MIME_TYPE,
  CURRENT_FORMAT_VERSION,
  PACKAGES_DIR_NAME,
  MANIFEST_FILE_NAME,
  PACKAGE_SERVICE_PORT_RANGE,
} from './package-types';

// Macro Selection Analyzer (for macro conversion)
export type {
  SelectionAnalysis,
  ExternalInput,
  ExternalOutput,
  BoundingBox,
  MacroConversionResult,
} from './macro/selection-analyzer';

export {
  analyzeSelection,
  convertSelectionToMacro,
} from './macro/selection-analyzer';

// Custom Node Types
export type {
  CustomNodeInput,
  CustomNodeOutput,
  CustomNodeProperty,
  CustomNodeDefinition,
  CustomNodeCompilerContext,
  CustomNodeCompilerResult,
  CustomNodeRuntimeContext,
  CustomNodeCompiler,
  CustomNodeRuntime,
  CustomNodeUIProps,
  CustomNodePackageStructure,
  CustomNodeValidationResult,
} from './custom-node-types';

export {
  validateCustomNodeDefinition,
  customNodeInputsToHandles,
  customNodeOutputsToHandles,
  customNodePropertiesToDefinitions,
} from './custom-node-types';

// Custom Node Registry
export type {
  RegisteredCustomNode,
  CustomNodeRegistryOptions,
} from './custom-node-registry';

export {
  CustomNodeRegistry,
  getCustomNodeRegistry,
  resetCustomNodeRegistry,
  createCustomNodeRegistry,
} from './custom-node-registry';

// Custom Node Compiler
export type {
  NodeCompilerOptions,
  NodeCompileResult,
} from './package/node-compiler';

export {
  compileCustomNode,
  compileCustomNodes,
  validateNodeSource,
  generateCompilerTemplate,
  generateRuntimeTemplate,
  generateUITemplate,
} from './package/node-compiler';

// Node Extension Types
export type {
  ExtensionInput,
  ExtensionOutput,
  ExtensionProperty,
  CompilerHook,
  CompilerHookContext,
  RuntimeHook,
  RuntimeHookContext,
  UIExtension,
  UIExtensionPosition,
  NodeExtension,
  LoadedExtension,
  LoadedCompilerHook,
  LoadedRuntimeHook,
  ExtendedCompilationResult,
  ExtendedRuntimeResult,
  ExtensionValidationResult,
} from './node-extension-types';

export {
  validateNodeExtension,
  extensionConditionMatches,
} from './node-extension-types';

// Node Extension Registry
export type {
  ExtensionRegistryOptions,
} from './node-extension-registry';

export {
  NodeExtensionRegistry,
  getNodeExtensionRegistry,
  resetNodeExtensionRegistry,
  createNodeExtensionRegistry,
} from './node-extension-registry';
