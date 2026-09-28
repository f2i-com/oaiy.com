/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./app.html",
    "./src/**/*.{js,ts,jsx,tsx}",
    // Vendored UI components + bundled module UIs (standalone — was the sibling repo).
    "./vendor/oaiy-ui-components/src/**/*.{js,ts,jsx,tsx}",
    "./src/bundled-modules/**/ui/**/*.{js,ts,jsx,tsx}",
  ],
  darkMode: 'class',
  theme: {
    // ONE breakpoint scale, in px, shared with the hand-written .oaiy-* rules
    // in src/index.css. Tailwind's defaults are rem, every hand-written rule is
    // px, and nothing sets a root font-size -- so a user browsing at a 20px
    // default moved Tailwind's md from 768 to 960 and landed it exactly on the
    // CSS 960 rule, reordering the whole cascade. Declared in full (not
    // `extend`) so the ORDER is explicit rather than merge-dependent.
    screens: {
      xs: '480px',   // phone portrait
      sm: '640px',
      md: '760px',   // matches the shell's own 760 rule
      lg: '960px',   // matches the shell's 960 rule
      xl: '1240px',  // matches the shell's rail-collapse rule
      '2xl': '1536px',
    },
    extend: {
      colors: {
        accent: {
          DEFAULT: 'rgb(var(--accent-primary) / <alpha-value>)',
          hover: 'rgb(var(--accent-hover) / <alpha-value>)',
        },
        surface: {
          primary: 'rgb(var(--color-bg-primary) / <alpha-value>)',
          secondary: 'rgb(var(--color-bg-secondary) / <alpha-value>)',
          tertiary: 'rgb(var(--color-bg-tertiary) / <alpha-value>)',
          elevated: 'rgb(var(--color-bg-elevated) / <alpha-value>)',
          canvas: 'rgb(var(--color-bg-canvas) / <alpha-value>)',
        },
        content: {
          primary: 'rgb(var(--color-text-primary) / <alpha-value>)',
          secondary: 'rgb(var(--color-text-secondary) / <alpha-value>)',
          tertiary: 'rgb(var(--color-text-tertiary) / <alpha-value>)',
          muted: 'rgb(var(--color-text-muted) / <alpha-value>)',
        },
        edge: {
          primary: 'rgb(var(--color-border-primary) / <alpha-value>)',
          secondary: 'rgb(var(--color-border-secondary) / <alpha-value>)',
        },
      },
      animation: {
        'pulse-cursor': 'pulse 1s cubic-bezier(0.4, 0, 0.6, 1) infinite',
        'toast-enter': 'toast-enter 0.3s ease-out forwards',
        'toast-exit': 'toast-exit 0.3s ease-in forwards',
      },
      keyframes: {
        'toast-enter': {
          '0%': { opacity: '0', transform: 'translateX(100%)' },
          '100%': { opacity: '1', transform: 'translateX(0)' },
        },
        'toast-exit': {
          '0%': { opacity: '1', transform: 'translateX(0)' },
          '100%': { opacity: '0', transform: 'translateX(100%)' },
        },
      },
    },
  },
  plugins: [
    function ({ addVariant }) {
      addVariant('short', '@media (max-height: 780px)');
    },
  ],
}
