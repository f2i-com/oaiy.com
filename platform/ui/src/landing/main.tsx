// Landing-page entry (the marketing root at `/`). Deliberately tiny: it
// pulls in ONLY the design tokens + ThemeProvider + the LandingPage — none
// of the app's heavy boot (companion detection, dynamic module discovery,
// plugin loading) runs here. The full builder lives at /app.html.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../styles/fonts';
import '../index.css';
import { ThemeProvider } from '../contexts/ThemeContext';
import LandingPage from './LandingPage';

// Shared-flow links are composed as `${origin}${pathname}?flow=<hash>` by the
// app (see useBackendIntegration.composeShareUrls). Those normally point at
// /app.html, but if one ever targets the marketing root we forward it to the
// app — which is the only entry that knows how to fetch + open a shared flow.
const params = new URLSearchParams(window.location.search);
if (params.has('flow')) {
  window.location.replace(`app.html${window.location.search}`);
} else {
  createRoot(document.getElementById('root')!).render(
    <StrictMode>
      <ThemeProvider>
        <LandingPage />
      </ThemeProvider>
    </StrictMode>,
  );
}
