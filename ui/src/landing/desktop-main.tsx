// Desktop-companion page entry (`/desktop.html`). Tiny, like the landing
// entry: design tokens + ThemeProvider + the page. No app boot runs here.
import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import '../styles/fonts';
import '../index.css';
import { ThemeProvider } from '../contexts/ThemeContext';
import DesktopPage from './DesktopPage';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ThemeProvider>
      <DesktopPage />
    </ThemeProvider>
  </StrictMode>,
);
