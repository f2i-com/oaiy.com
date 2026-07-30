import { useState, useCallback, useEffect } from 'react';
import { createPortal } from 'react-dom';
import { ReactFlowProvider } from '@xyflow/react';
import { invoke } from '@tauri-apps/api/core';
import OAIYApp from './components/OAIYApp';
import ErrorBoundary from './components/ErrorBoundary';
import { ToastProvider } from './components/Toast';
import { ThemeProvider } from './contexts/ThemeContext';
import SplashScreen from './components/SplashScreen';
import { JobQueueProvider } from './contexts/JobQueueContext';
import CliWorkflowRunner from './components/CliWorkflowRunner';
import { loadRuntimePlugins } from './dynamicModules';
import { initMediaServerPort, initMediaServerToken } from 'oaiy-core';
import { openSharedFlowFromUrl, shareHashFromUrl } from './lib/openSharedFlow';
import PasswordPromptModal from './components/dialogs/PasswordPromptModal';
import { getCliRunConfig, type CliRunConfig } from './hooks/useCliRunMode';
import { createLogger } from './utils/logger';

const logger = createLogger('App');

function App() {
  // Check if we should skip splash (set by API restart)
  const shouldSkipSplash = localStorage.getItem('skipSplash') === 'true';
  const [showSplash, setShowSplash] = useState(!shouldSkipSplash);
  const [appReady, setAppReady] = useState(false);
  const [cliConfig, setCliConfig] = useState<CliRunConfig | null>(null);
  const [cliChecked, setCliChecked] = useState(false);
  // Dev-mode auto-skip: when OAIY_NO_AUTH=1 the API server starts with auth
  // disabled (see api_server::ApiServerConfig::default). We treat that as a
  // signal that the operator wants headless/scripted access and bypass the
  // splash so plugins + the API bridge mount automatically. Null until the
  // first status probe lands.
  const [devModeChecked, setDevModeChecked] = useState(false);
  const [devModeNoAuth, setDevModeNoAuth] = useState(false);
  // Opening an ENCRYPTED shared flow prompts for the password via a masked modal (below) instead
  // of the unmasked window.prompt fallback. Holds the previous attempt's error + the promise
  // resolver the openSharedFlowFromUrl await is parked on.
  const [pwPrompt, setPwPrompt] = useState<{
    lastError: string | null;
    resolve: (password: string | null) => void;
  } | null>(null);

  // Check CLI run mode on startup
  useEffect(() => {
    getCliRunConfig().then((config) => {
      setCliConfig(config);
      setCliChecked(true);
      if (config.isRunMode) {
        logger.info('CLI run mode detected', { config });
      }
    });
  }, []);

  // Probe API status once to learn whether OAIY_NO_AUTH was honored. The
  // env var is the only thing that can produce `allow_no_auth: true` at
  // startup (it isn't persisted to the secrets backend), so this flag is
  // a reliable proxy for "developer/scripted mode" right after boot. We
  // bypass the splash so external tooling (MCP, smoke tests) doesn't have
  // to drive the UI.
  useEffect(() => {
    invoke<{ allow_no_auth?: boolean }>('get_api_status')
      .then((status) => {
        const envFlag = status?.allow_no_auth === true;
        setDevModeNoAuth(envFlag);
        setDevModeChecked(true);
        if (envFlag) {
          logger.info('OAIY_NO_AUTH detected — bypassing splash');
        }
      })
      .catch((err) => {
        logger.debug('get_api_status probe failed', { err });
        setDevModeChecked(true);
      });
  }, []);

  const handleStart = useCallback(async (appDataPath?: string) => {
    // Initialize media server port + capability token (for dynamic port
    // support + URL minting). Both must be cached before any node that
    // renders a `<video>`/`<audio>` element via `pathToMediaUrl` mounts,
    // otherwise the URL is missing the token and the media server rejects
    // it (empty preview box). Run them in parallel — independent commands.
    try {
      await Promise.all([initMediaServerPort(), initMediaServerToken()]);
    } catch (err) {
      logger.error('Failed to initialise media server (port/token)', { error: err });
    }

    // Load runtime plugins when user clicks start
    // appDataPath is the root folder, plugins are in {appDataPath}/plugins
    try {
      await loadRuntimePlugins(appDataPath);
    } catch (err) {
      logger.error('Failed to load runtime plugins', { error: err });
    }

    // Shared-link auto-load — when the URL has `?flow=<hash>`, fetch
    // the flow from the backend and stash it in localStorage so
    // useProject picks it up on next mount. Encrypted flows prompt for
    // the password via a masked modal (see openSharedFlowFromUrl). A
    // failed/cancelled prompt logs but doesn't block the rest of boot.
    if (shareHashFromUrl()) {
      try {
        const result = await openSharedFlowFromUrl({
          // Masked modal prompt instead of the cleartext window.prompt fallback.
          promptPassword: (lastError) =>
            new Promise<string | null>((resolve) => setPwPrompt({ lastError, resolve })),
        });
        if (result) {
          logger.info('Loaded shared flow', {
            hash: result.hash, isEditable: result.isEditable, title: result.meta.title,
          });
          // Persist the loaded snapshot through the same key useProject
          // reads from at init, then strip ?flow= from the URL so a
          // page reload doesn't re-prompt.
          try {
            const PROJECT_KEY = 'oaiy_project';
            const flowJson = result.flow as { flows?: unknown[]; settings?: unknown; constants?: unknown };
            localStorage.setItem(PROJECT_KEY, JSON.stringify({
              version: '1.0.0',
              name: result.meta.title || 'Shared flow',
              description: '',
              createdAt: result.meta.created_at,
              updatedAt: result.meta.updated_at,
              flows: Array.isArray(flowJson.flows) ? flowJson.flows : [],
              settings: flowJson.settings ?? {},
              constants: Array.isArray(flowJson.constants) ? flowJson.constants : [],
              llmEndpoints: [],
              imageGenEndpoints: [],
              httpPresets: [],
              // SECURITY: a shared flow is UNTRUSTED — mark the project so the run path executes it
              // hardened (worker isolation + permission gate, no code:execute/secrets:read), blocking
              // a malicious logic_block from running arbitrary JS / stealing API keys in this origin.
              // Set HERE by the importer, NOT copied from flowJson, so the flow can't unset it.
              untrusted: true,
            }));
          } catch (e) {
            logger.warn('Failed to persist shared flow to localStorage', { e });
          }
          // Drop the query string so a reload doesn't re-fetch + re-prompt.
          const cleanUrl = window.location.origin + window.location.pathname;
          window.history.replaceState({}, '', cleanUrl);
        }
      } catch (err) {
        logger.error('Failed to load shared flow', { err });
      }
    }

    setShowSplash(false);
    setAppReady(true);
  }, []);

  // Auto-start when skipSplash flag is set (e.g., from API restart),
  // when in CLI run mode, or when OAIY_NO_AUTH is set (dev/scripted mode).
  //
  // The web build hits `devModeNoAuth = true` every load (the shim
  // returns `allow_no_auth: true` because there's no real auth gate in
  // the browser). Without a delay the splash would skip to the
  // workspace instantly. We add a ~900ms minimum-display window so
  // the ORCHESTRATE AI YOURSELF wordmark + spinner actually register as
  // a deliberate splash, then transition.
  useEffect(() => {
    if (!cliChecked || !devModeChecked) return;

    if (cliConfig?.isRunMode) {
      // CLI run mode - auto-start immediately, no splash beat.
      logger.info('Auto-starting for CLI run mode');
      handleStart();
    } else if (shouldSkipSplash) {
      // Same: explicit "skip splash" flag wants instant transition.
      localStorage.removeItem('skipSplash');
      handleStart();
    } else if (devModeNoAuth) {
      // Shared-link visitors land on the splash purely as a side-effect
      // of the auto-skip — they came for the flow, not the wordmark. Skip
      // the 900ms beat for them so the flow appears as fast as it loads.
      // Everyone else still gets the brief brand reveal.
      const hasSharedLink = typeof window !== 'undefined'
        && new URLSearchParams(window.location.search).has('flow');
      const delay = hasSharedLink ? 0 : 900;
      logger.info(`Auto-starting for OAIY_NO_AUTH dev mode (${delay}ms ${hasSharedLink ? '— shared link, skipping beat' : 'beat'})`);
      const t = window.setTimeout(() => handleStart(), delay);
      return () => window.clearTimeout(t);
    }
  }, [cliChecked, devModeChecked, cliConfig?.isRunMode, shouldSkipSplash, devModeNoAuth, handleStart]);

  // The masked password prompt for an encrypted shared flow. Portaled to <body> so it renders over
  // WHICHEVER boot state is showing while openSharedFlowFromUrl awaits the password (the splash,
  // or null during a skip-splash boot). The theme (.dark + accent vars) is global, so it stays
  // themed even outside ThemeProvider.
  const pwModal = pwPrompt
    ? createPortal(
        <PasswordPromptModal
          lastError={pwPrompt.lastError}
          onSubmit={(pw) => {
            pwPrompt.resolve(pw);
            setPwPrompt(null);
          }}
          onCancel={() => {
            pwPrompt.resolve(null);
            setPwPrompt(null);
          }}
        />,
        document.body,
      )
    : null;

  // Wait for CLI + dev-mode checks to complete
  if (!cliChecked || !devModeChecked) {
    return pwModal;
  }

  if (showSplash && !cliConfig?.isRunMode) {
    return (
      <ErrorBoundary>
        <ThemeProvider>
          <SplashScreen onStart={handleStart} />
          {pwModal}
        </ThemeProvider>
      </ErrorBoundary>
    );
  }

  if (!appReady) {
    return pwModal;
  }

  // CLI run mode - minimal UI with workflow execution
  if (cliConfig?.isRunMode) {
    return (
      <ErrorBoundary>
        <ThemeProvider>
          <ToastProvider>
            <JobQueueProvider>
              <CliWorkflowRunner />
            </JobQueueProvider>
          </ToastProvider>
        </ThemeProvider>
      </ErrorBoundary>
    );
  }

  return (
    <ErrorBoundary>
      <ThemeProvider>
        <ToastProvider>
          <ReactFlowProvider>
            <OAIYApp />
          </ReactFlowProvider>
        </ToastProvider>
      </ThemeProvider>
    </ErrorBoundary>
  );
}

export default App;
