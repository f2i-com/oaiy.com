/**
 * CLI Workflow Runner
 *
 * Handles workflow execution when app is started with --run flag.
 * Runs in hidden window mode for cron jobs and CLI automation.
 */

import { useEffect, useRef, useState } from 'react';
import { useCliRunMode } from '../hooks/useCliRunMode';
import { useJobQueue } from '../contexts/JobQueueContext';
import { createLogger } from '../utils/logger';
import { readTextFile } from '@tauri-apps/plugin-fs';
import { invoke } from '@tauri-apps/api/core';

const logger = createLogger('CLI-Runner');

interface CliWorkflowRunnerProps {
  onComplete?: () => void;
}

/**
 * Component that handles CLI workflow execution
 * Should be rendered when the app is started with --run flag
 */
export function CliWorkflowRunner({ onComplete }: CliWorkflowRunnerProps) {
  const { workflowPath, inputs, writeOutput, exit } = useCliRunMode();
  const { submitJob, jobs } = useJobQueue();
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState<'loading' | 'running' | 'completed' | 'error'>('loading');
  const [error, setError] = useState<string | null>(null);
  const hasStarted = useRef(false);
  const hasExited = useRef(false);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Watchdog: abort a run that never reaches a terminal job status (e.g. a
  // blocking node in headless --run/cron mode would otherwise hang forever).
  // Default 45 min; overridable via VITE_OAIY_CLI_TIMEOUT_MS.
  const RUN_TIMEOUT_MS = (() => {
    const raw = import.meta.env.VITE_OAIY_CLI_TIMEOUT_MS;
    const parsed = raw != null ? Number(raw) : NaN;
    return Number.isFinite(parsed) && parsed > 0 ? parsed : 45 * 60 * 1000;
  })();

  // Start workflow execution
  useEffect(() => {
    if (hasStarted.current || !workflowPath) return;
    hasStarted.current = true;

    async function runWorkflow() {
      try {
        logger.info('Loading workflow', { path: workflowPath });

        // Load workflow file
        let workflowData: { nodes: unknown[]; edges: unknown[] };

        // Check if it's a .oaiy package or .json file
        const isPackage = workflowPath!.toLowerCase().endsWith('.oaiy');

        if (isPackage) {
          // Load from package using Tauri command
          const content = await invoke<string>('read_package_flow_content', {
            packagePath: workflowPath,
          });
          workflowData = JSON.parse(content);
        } else {
          // Load JSON file directly
          const content = await readTextFile(workflowPath!);
          workflowData = JSON.parse(content);
        }

        // Parse input values if provided
        let inputValues: Record<string, unknown> = {};
        if (inputs) {
          try {
            inputValues = JSON.parse(inputs);
            logger.info('Using input values', { inputs: inputValues });
          } catch (e) {
            logger.warn('Failed to parse inputs JSON, using empty inputs', { error: e });
          }
        }

        setStatus('running');

        // Submit the workflow job
        const flowName = workflowPath!.split(/[/\\]/).pop() || 'CLI Workflow';
        const id = submitJob(
          'cli-run',
          { nodes: workflowData.nodes as never[], edges: workflowData.edges as never[] },
          inputValues,
          flowName
        );

        setJobId(id);
        logger.info('Workflow job started', { jobId: id });

        // Arm the watchdog so a job that never reaches a terminal status
        // (blocking node in headless mode) cannot hang the process forever.
        timeoutRef.current = setTimeout(() => {
          if (hasExited.current) return;
          hasExited.current = true;

          logger.error('Workflow timed out', { jobId: id, timeoutMs: RUN_TIMEOUT_MS });
          setError('CLI run timed out');
          setStatus('error');

          (async () => {
            await writeOutput({ success: false, error: 'CLI run timed out' });
            await exit(1);
          })();
        }, RUN_TIMEOUT_MS);
      } catch (err) {
        if (hasExited.current) return;
        hasExited.current = true;

        const errMsg = err instanceof Error ? err.message : String(err);
        logger.error('Failed to start workflow', { error: errMsg });
        setError(errMsg);
        setStatus('error');

        // Write error to output and exit with error code
        await writeOutput({ success: false, error: errMsg });
        await exit(1);
      }
    }

    runWorkflow();
  }, [workflowPath, inputs, submitJob, writeOutput, exit]);

  // Monitor job completion
  useEffect(() => {
    if (!jobId || hasExited.current) return;

    const job = jobs.find((j) => j.id === jobId);
    if (!job) return;

    if (job.status === 'completed') {
      if (hasExited.current) return; // Guard against multiple calls
      hasExited.current = true;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      logger.info('Workflow completed successfully');
      setStatus('completed');

      // Write results and exit
      (async () => {
        await writeOutput({
          success: true,
          jobId,
          results: job.result || {},
          logs: job.logs,
        });
        onComplete?.();
        await exit(0);
      })();
    } else if (job.status === 'failed' || job.status === 'aborted') {
      if (hasExited.current) return; // Guard against multiple calls
      hasExited.current = true;
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }

      const errMsg = job.error || 'Workflow failed';
      logger.error('Workflow failed', { error: errMsg });
      setError(errMsg);
      setStatus('error');

      // Write error and exit
      (async () => {
        await writeOutput({
          success: false,
          jobId,
          error: errMsg,
          logs: job.logs,
        });
        onComplete?.();
        await exit(1);
      })();
    }
  }, [jobId, jobs, writeOutput, exit, onComplete]);

  // Clear the watchdog on unmount so a normal completion / teardown never
  // triggers a late exit.
  useEffect(() => {
    return () => {
      if (timeoutRef.current) {
        clearTimeout(timeoutRef.current);
        timeoutRef.current = null;
      }
    };
  }, []);

  // This component doesn't render anything visible (hidden window mode)
  // But we can log status for debugging
  useEffect(() => {
    logger.info('CLI Runner status', { status, error, jobId });
  }, [status, error, jobId]);

  return null;
}

export default CliWorkflowRunner;
