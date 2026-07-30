import React from 'react';
import ReactDOM from 'react-dom/client';
import App from './App';
import { ToastProvider } from './Toasts';
import { applyTheme, initialTheme } from './theme';
import './styles.css';

// Set the theme class before first paint to avoid a flash of the wrong theme.
applyTheme(initialTheme());

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <ToastProvider>
      <App />
    </ToastProvider>
  </React.StrictMode>,
);
