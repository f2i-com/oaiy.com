# Build a SoftN web app in OAIY

Choose the framework the user asks for. Vanilla JavaScript is the smallest starting point. React and Vue use the same HTML, CSS and JavaScript project layout. All three starters include everything their preview needs offline.

## Start a project

Call `softn_examples` with `name` set to `vanilla-js`, `react` or `vue` and `install_to` set to a project folder. The tool returns the new app's path. Read its `app.js`, `index.html` and `manifest.json` before editing. For example, to build a React tracker in the workspace:

```json
{"name":"react","install_to":"apps"}
```

Each starter has `"engine": 2` and `"main": "index.html"` in its manifest. Keep these fields when changing the app's name or adding files. Load code with `<script type="module" src="./app.js"></script>` and place a `<div id="app"></div>` before it. Relative imports resolve to files inside the project.

## Vanilla JavaScript

Edit the DOM with browser APIs. For example, this entry creates a button and counts clicks:

```js
const button = document.createElement('button');
let count = 0;
button.textContent = 'Count 0';
button.addEventListener('click', () => {
  button.textContent = `Count ${++count}`;
});
document.getElementById('app').appendChild(button);
```

## React

The React starter includes React and its DOM renderer in `vendor/react.js`. Edit components and hooks directly in `app.js`. This form runs without a JSX transform:

```js
import { React, createRoot } from './vendor/react.js';

function Counter() {
  const [count, setCount] = React.useState(0);
  return React.createElement('button', {
    onClick: () => setCount(value => value + 1)
  }, `Count ${count}`);
}

createRoot(document.getElementById('app')).render(React.createElement(Counter));
```

## Vue

The Vue starter includes its runtime in `vendor/vue.js`. Edit its component or add more JavaScript modules using `ref` and `h`:

```js
import { createApp, h, ref } from './vendor/vue.js';

createApp({
  setup() {
    const count = ref(0);
    return () => h('button', {
      onClick: () => { count.value += 1; }
    }, `Count ${count.value}`);
  }
}).mount('#app');
```

## Preview and share

Keep the framework module and its notices with the project. Use `softn_check` with the app folder to check its files and start it. Click its controls directly in **Preview** to test interactions. The published runtime draws web apps in an isolated shadow root. Agent screenshots, text inspection and scripted clicks are available for classic apps. For web apps, check the visible preview and report which interactions you actually tested. Export the folder as a `.softn` archive to reopen it in SoftN Runtime, Studio or Builder.

SoftN Studio and Builder also support JSX and Vue single-file components. Their exports carry both the original files and prepared JavaScript. Keep both when importing one into OAIY. After editing a `.jsx` or `.vue` file, reopen the source in Studio or Builder and export again to refresh the prepared modules. The OAIY starters above use plain `app.js`, so edits preview immediately.

For an existing classic app, keep its manifest and component syntax. Its JavaScript or Python writing guide is available through `softn_docs` with the app's folder.
