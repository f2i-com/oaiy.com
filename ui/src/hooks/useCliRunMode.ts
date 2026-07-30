/**
 * Hook for CLI run mode - handles workflow execution from command line
 *
 * Usage: oaiy --run workflow.json --inputs '{"key": "value"}' --output results.json
 */

import { useState, useEffect, useCallback } from 'react';
import { invoke } from '@tauri-apps/api/core';
import { createLogger } from '../utils/logger';

const logger = createLogger('CLI');

export interface CliRunConfig {
  isRunMode: boolean;
  workflowPath: string | null;
  inputs: string | null;
  outputPath: string | null;
}

/**
 * Get CLI run configuration from Rust backend
 */
export async function getCliRunConfig(): Promise<CliRunConfig> {
  try {
    const config = await invoke<CliRunConfig>('get_cli_run_config');
    return config;
  } catch (err) {
    logger.error('Failed to get CLI run config', { error: err });
    return {
      isRunMode: false,
      workflowPath: null,
      inputs: null,
      outputPath: null,
    };
  }
}

/**
 * Write workflow results to output file
 */
export async function writeCliOutput(path: string, content: string): Promise<void> {
  await invoke('write_cli_output', { path, content });
}

/**
 * Exit the application with optional exit code
 */
export async function exitApp(code: number = 0): Promise<void> {
  await invoke('exit_app', { code });
}

/**
 * Hook for CLI run mode
 */
export function useCliRunMode() {
  const [config, setConfig] = useState<CliRunConfig | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    getCliRunConfig()
      .then(setConfig)
      .finally(() => setLoading(false));
  }, []);

  const writeOutput = useCallback(async (results: unknown) => {
    if (config?.outputPath) {
      const content = JSON.stringify(results, null, 2);
      await writeCliOutput(config.outputPath, content);
      logger.info('Results written to', { path: config.outputPath });
    }
  }, [config?.outputPath]);

  const exit = useCallback(async (code: number = 0) => {
    logger.info('Exiting with code', { code });
    await exitApp(code);
  }, []);

  return {
    config,
    loading,
    isRunMode: config?.isRunMode ?? false,
    workflowPath: config?.workflowPath ?? null,
    inputs: config?.inputs ?? null,
    outputPath: config?.outputPath ?? null,
    writeOutput,
    exit,
  };
}

export default useCliRunMode;
