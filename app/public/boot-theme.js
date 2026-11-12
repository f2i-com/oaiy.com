// The theme, said before the first paint. index.html loads this first, as a file of the page's own: its content
// policy allows no inline script, and without it the page drew its dark starting screen for a moment in a light
// window (the stylesheet is dark until told otherwise, and the app's own modules run after the first paint).
// The same order as src/ui/theme.ts, which takes over when the app starts: in OAIY's window what the desktop said
// while the page loaded, then what it said last in this window, then what it gave the page; on its own, the system's.
(function () {
  var theme = 'dark';
  try {
    var desktop = window.__OAIY_DESKTOP__;
    if (desktop) {
      var kept = null;
      try {
        kept = sessionStorage.getItem('oaiy-desktop-theme');
      } catch (e) {
        // storage can be unavailable
      }
      var said = [window.__OAIY_THEME__, kept, desktop.theme];
      for (var i = 0; i < said.length; i++) {
        if (said[i] === 'light' || said[i] === 'dark') {
          theme = said[i];
          break;
        }
      }
    } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: light)').matches) {
      theme = 'light';
    }
  } catch (e) {
    // dark, as the stylesheet starts
  }
  document.documentElement.setAttribute('data-theme', theme);
})();
