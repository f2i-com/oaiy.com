# OAIY Agent

A coding agent that runs entirely in your browser.

> In OAIY this app is the **Agent**: OAIY Desktop shows it as the Agent page, where it
> also answers the phone (the Front desk) and sets OAIY up. See the
> [repository's README](../README.md), [the AI Receptionist](../docs/RECEPTIONIST.md) and
> [the Agent's control of OAIY](../docs/AGENT_CONTROL.md). This page covers the app itself.

- **Your projects stay in the browser.** Files live in the browser's private file system (OPFS). Open a folder from disk, import a `.zip`, export one back, or start from scratch. Everything works offline once the page has loaded (it installs as an app).
- **AI-written code runs on the [Zipp](https://github.com/f2i-com/zipp.org) VM.** JavaScript and Python run in Zipp's WebAssembly engine inside a Web Worker. The code can reach the project and nothing else, except what the network gate lets through.
- **A shell, emulated.** The terminal and the agent's `sandbox_shell` are a bash-like shell written in JavaScript on the same sandbox, with `git`, `jq`, `tar`/`zip`, `node` and `python` built in (see [The shell](#the-shell)). There are no real processes, so `npm install` and compilers don't exist here.
- **Any model.** A server on your own machine (Ollama, LM Studio, OAIY, llama.cpp — anything OpenAI-compatible), Anthropic's API, or any OpenAI-compatible API.
- **Images, video and audio.** With OAIY, or any service with OpenAI's media APIs, the agent can make pictures, short videos (talking ones too), speech, music, sound effects and 3D models straight into the project, and remove a picture's background or upscale it (see [Images, video and audio](#images-video-and-audio)).
- **Web pages with a live preview.** The agent builds HTML, CSS and JavaScript pages and sees them as you do: their JavaScript runs on the Zipp VM against the preview's real DOM, and it can screenshot the page at phone, tablet or desktop sizes (see [Web pages](#web-pages)).
- **A network gate.** `/internet on | off | allowlist | allow <host> | deny <host> | status` (or `/net`) decides every request made on the model's behalf. Requests to your AI provider are not gated.

## Context and long work

- **The model's context window** is found out from its server before the first request:
  - Ollama: the size the model is loaded with;
  - LM Studio: the loaded context length;
  - llama.cpp: `/props`;
  - vLLM, OpenRouter and other OpenAI-compatible servers: the model list;
  - cloud models: a table of known windows.

  You can type your own number under **Context** in Settings, and **Detect** asks again. If a server rejects a prompt as too long and states its limit, that limit is remembered.
- **Compaction:** before a prompt would pass 75% of the window (adjustable in Settings), the model summarizes the older turns: goal, decisions, files changed, plan, open errors. The recent turns stay word for word.
  - If the old part is too big for the model to read at once, it's summarized in chunks.
  - If no summary can be made, the old turns are reduced to their requests and file changes instead.
  - The chat keeps the whole conversation; only the model reads from the summary on.
- **Sub-agents:** for a big task, the agent can hand independent parts to sub-agents with `delegate`.
  - Each task runs in a fresh agent with its own, smaller context (32k tokens by default, set in Settings) and the same tools except `delegate` and `update_plan`.
  - Each reports back what it did and which files it changed. The main conversation keeps only those reports, not the sub-agents' work.
  - Tasks wait in a queue per model: one at a time for a local server (which usually answers one request at a time), three for an API. Change it under **Agents** in Settings. Stop cancels the waiting tasks as well as the running ones.
  - A task can name the plan step it completes, so the plan updates as tasks finish. The chat shows each task live: queued, working (with what it's doing right now), then done, with its report.
  - A video of two or more scenes is made a scene per sub-agent, once its script, characters and voices are ready, so the main conversation (and each of its steps) stays short.
  - A message you send while sub-agents work reaches them too: each one working then reads it at its next step (a task that starts later reads it first), and the main agent gets it after them. A picture's reviewer hears it the same way.
- **A run that placed a call is not pushed on by its plan while the call is live:** the automatic plan nudge (and the status checks it brought) of the conversation that placed it, the Front desk's runner or a project's agent, stays quiet, so a local model server, which answers one request at a time, is not kept busy while a caller waits for a reply. The call's result comes to the conversation once, when it ends; a reply with no words is still asked, at most twice, to say what it started. Nothing is held or reordered: a request already on the server finishes first. See [docs/CALLS.md](../docs/CALLS.md#a-run-that-placed-a-call-waits-for-its-result).
- **The meter** in the Agent pane's title shows how full the context is. The provider's own token counts calibrate the estimate as it goes.
- **The chat** reads newest first: what the agent is doing now is at the top. Older messages are drawn as you scroll down to them, and **↑ Current** goes back to the top. Scrolled down to read, the view stays put while new messages arrive above.
  - Marks in the log show where the plan moved (a step started, a step done, the plan changed), so the history reads step by step.
  - The window never scrolls as a page: the file tree, the editor, the terminal and the chat each scroll on their own. The chat box stays at the bottom of the Agent pane, beside even a very large open file, at any window size.
  - The model's thinking is shown as it thinks, and stays shown (it is saved with the chat, but never sent back to the model). Each step has a folded **prompt** entry: open it to read exactly what the model was sent (the system prompt, the tools and the conversation). A picture, clip or sound shows the prompt it was made from on its card.
- **Prompt reuse:** old pictures leave the prompt a few at a time rather than one per step, so the start of the prompt stays the same and a server that reuses it (OAIY, llama.cpp) reads only what is new. In incognito, OAIY keeps the project's prompt state in memory only, for its next step, and wipes it when the project is cleared or incognito is turned off.
- **A short system prompt:** the model starts with a few lines and your request, so nothing crowds it out. What a kind of work needs is in its guide (apps, videos, pictures and sound, long documents), read with the `guide` tool when the work calls for it, or with your request when it plainly asks for that work ("make a video" reads the video guide). A guide stays in the instructions once read. The first picture, clip or sound made without its guide waits for it, and the app tools come only once the app guide is read or the project has an app.
- **Scripts follow the request:** while the agent makes a video, your request stays in its instructions word for word, as a reference, and it is never trimmed from the conversation. Before the first picture, clip or voice is made, the agent checks the script against the request point by point, and fixes the script where it strays. A write that only repeats what a file already holds is not made, and a file written in full a third time in one request tells the agent to mark the step done and move on (or change just a part).
- **A picture sent back is remade where it is:** until it is made again at the same path, no other picture of that video is made (a fix under another name would leave the old one in place, the one the script, the review and the clips use). Asking for a review of the sent-back picture before it is remade says what to fix rather than judging the same picture again; the remake is reviewed as soon as it is made, from its new bytes.

## Images, video and audio

The agent gets more tools when a media service is set up:
- `generate_image` saves PNGs: from a prompt, or from reference pictures to edit or combine.
- `generate_video` saves an MP4 from a prompt. It can also:
  - animate a start image, optionally moving to an end image;
  - make a character talk, with lip movement: it takes `say` (words to speak in a voice) or `soundtrack` (a speech or audio file to follow). Without a length, the clip is as long as the speech, up to the model's few seconds.
- `generate_speech` saves spoken audio (mp3, wav, opus, aac, flac) in a saved voice, an OpenAI voice name, or a voice described in words.
- `create_voice` designs a voice from a description and saves it on OAIY, so a character keeps the same voice. It uses the speech model chosen in Settings (OAIY's Qwen3-TTS or Breeze TTS 2). A voice Breeze TTS 2 made is always spoken by a Breeze model.
- `generate_music` saves a song from a style and lyrics (with [Verse] and [Chorus] sections), or an instrumental.
- `generate_sound_effect` saves a sound effect (wav or mp3, up to 30 seconds) from a description of what makes it, where, and how it sounds (OAIY's MOSS-SoundEffect).
- `generate_3d_model` saves a 3D model (a GLB mesh) of one object from a picture of it, with OAIY's Pixal3D, and beside it the object as the service cut it out, with a transparent background (`NAME.cutout.png`, 2048 pixels a side when OAIY has Real-ESRGAN). When OAIY has BiRefNet it removes any background itself, and the tool tells the agent so; Real-ESRGAN enlarges a small picture first. The agent makes the picture first with `generate_image`: the object alone, whole and centred, from a three-quarter view. The model has baked PBR textures, Y up, its front facing +Z, and fits a unit cube; it takes about a minute and a half. The agent looks at it with `preview_screenshot` (see [3D models](#3d-models)) and puts it in a SoftN app's `assets/`, where a `Scene3D` shows it.
- `remove_background` saves a picture with its background removed (a PNG with a transparent background, for sprites, icons and cut-outs), and `upscale_image` makes a picture two or four times larger with its detail restored. OAIY offers them when it has BiRefNet and Real-ESRGAN, which its Get models page downloads.

The results appear in the chat (with a player for video and audio) and in the project, where an app can use them.

### Editing video and sound

These tools work on files already in the project, with or without a media service:
- `media_info` reports what a video, audio or image file holds: duration, size, frame rate and exact frame count, codecs, sample rate and channels.
- `video_frames` saves frames as PNGs: at times, by frame number, one every N seconds, or the first and last. The last frame of a clip is the start image for the next one, so a long video is made clip by clip.
- `video_split` cuts a video into parts at times or frame numbers.
- `media_compose` builds a timeline and saves it as .mp4 or .webm, or as .wav or .m4a for sound only:
  - **clips** play one after another: videos, trimmed with start and end, or still pictures shown for a duration. Each can fade from or to black and set its own sound's volume.
  - **audio** lays music, speech or effects over the whole timeline. Each track is placed at a time, trimmed, looped, set louder or quieter, and faded.
  - A track can **duck** everything else while it plays, for a voice over music.
  - The whole mix has its own volume and fades, and is turned down automatically if it would clip.

They run in the page with the browser's own video and audio codecs (WebCodecs: H.264 and AAC where the system has them, otherwise VP9 and Opus). [Mediabunny](https://mediabunny.dev) reads and writes the files. Everything is re-encoded, so cuts are exact to the frame. Nothing leaves the computer, and it works offline and in the desktop app. A model with a context window under 16k tokens doesn't get these four tools, so that its window still has room to work.

- **OAIY is found on its own, in OAIY's own windows and the desktop app.** When the page opens there, it asks `http://127.0.0.1:8080/v1/discovery`. In a tab in a browser the page asks nothing of your computer when it opens (a look at your own network is a permission prompt in current Chrome and Edge, and a public page has no business with it): press **Settings → Images, video and audio → Find OAIY** once, and from then on the page asks OAIY at the address it found when it opens.
  - If OAIY answers, its image, video, speech, music, sound effects and 3D models, their limits (sizes, edits, seconds), its saved voices and its defaults fill **Settings → Images, video and audio**.
  - OAIY is also added as a chat provider if none points at it yet. It becomes the active one only if nothing else is.
  - Found again later, its model lists are refreshed. Your chosen models and key stay.
- **OAIY allows the Agent.** Without an API key, OAIY answers only the origins in `gateway.cors_origins` in its config. Its defaults include `https://bot.computer` (where the Agent was first served), `http://localhost:5317` and the desktop app. Serving the Agent from anywhere else means adding that address there (the chat says which one), or setting an API key in OAIY and typing it in Settings. Then press **Find OAIY**.
- **Other services.** Type the address of any OpenAI-spec service (for example `https://api.openai.com/v1`) and its key, then **List models** and choose. Images use `/images/generations` and `/images/edits`. Video uses `/videos` and follows the job until it's done. Speech uses `/audio/speech`.
- **While it works.** The chat shows the progress of a video or a song. Stopping the agent cancels the job. The models' limits are in the tools' descriptions, so the model asks for sizes and lengths the service can make.
- **Privacy.** These requests go straight from the page to the service you set up, like requests to your AI provider. They are not behind the network gate. Keys are stored encrypted with the provider keys.

## Files, images and big files

- **Viewer:** opening an image in the file tree shows it (fit or actual size; an SVG can switch to its source). Audio and video files open in a player.
- **In the chat:** an attached image appears as a thumbnail, and audio or video gets a player. Clicking one opens the file (or, for a `.softn`, previews its app). The agent can hand you a file with `present_file`, such as a sound or image it made, and it shows up the same way.
- **Attach** files to a message with 📎, by dropping them on the chat, or by pasting an image.
  - Files are saved in the project under `uploads/`, so the agent can open them with its tools.
  - Images also go to the model with the message, if it can see images. If it can't, the run carries on in text.
  - A `.softn` file is unpacked into an app folder of its own, and the original stays in `uploads/`. The agent is told what the app holds. If you ask for changes, it edits that folder. If you ask it to recreate or rebuild the app, or make something like it, it writes a new app in another folder and leaves the original as it is. The `softn_import` tool unpacks any `.softn` in the project again, for a fresh copy.
- **`view_image`:** shows the model an image scaled to its vision budget. To zoom, it passes a region in the image's original pixels, and that region is shown at up to 2048 px. A small region therefore shows real detail rather than an enlarged thumbnail. `grid: true` overlays labelled coordinates to aim the next zoom.
- **Large files:**
  - `file_info` gives the size, line count and longest line.
  - `search_file` finds regex matches with line, column, character offset and context.
  - `read_file` pages by lines, cuts very long lines with a pointer to the rest, and reads raw character ranges with `char_start` (for minified or single-line files).

## Web pages

Any `.html` file of the project is a web page, with its CSS, images and JavaScript beside it. The **Preview** tab shows it; its picker lists the project's pages, SoftN apps and 3D models, and clicking an `.html` file in the tree shows that page.

- **JavaScript on Zipp.** A page's scripts don't run in the browser's own engine. The preview frame (`public/webpage/`) builds the page from its HTML and CSS, takes out its `<script>` elements and `on*` attributes, and runs them on the Zipp VM instead, against a DOM facade (`src/preview/zippDom.guest.js`). Every node, event, style or canvas the page touches is a proxy that reads, writes and calls the real object in the frame over Zipp's synchronous host channel. Listeners, timers and observers are the page's functions, called back when the real event fires. So the browser renders, lays out and dispatches, and only the page's code runs on Zipp. This follows the live DOM bridge described in zipp-browser's design notes.
  - Works: classic scripts (several share globals and run in order), the DOM, `addEventListener` and `on*` attributes (including ones added through `innerHTML`), timers and `requestAnimationFrame`, canvas, `localStorage` (kept per page while OAIY is open), `fetch` and `XMLHttpRequest` for the project's own files, and links between the project's pages.
  - Doesn't: ES module `import`/`export` between files, JSX or anything that needs a build step, and the internet: the preview is offline, so CDN scripts, web fonts and remote images don't load. The page is told what didn't load.
  - A script that never finishes (an endless loop) is stopped after about two seconds, and the page's scripts stop with it.
  - The frame is sandboxed with an opaque origin and a strict CSP, so nothing of the page runs natively, even HTML it inserts later.
- **Screen sizes.** The preview's size picker shows the page at a phone (390×844), tablet (820×1180), laptop (1366×768) or desktop (1920×1080) size, or any custom size. The page lays out at that size (media queries, `innerWidth`), and the frame is scaled down to fit the pane.
- **The agent's tools:** `page_check` renders a page and reports script errors and files that didn't load. `page_inspect` and `page_interact` describe and use the page the way `softn_inspect` and `softn_interact` do for apps. `preview_viewport` sets the screen size. `preview_screenshot` shows the agent how the page (or a SoftN app) looks at that size, one screen or the whole page, and can save the PNG in the project.
- **Screenshots** are drawn inside the frame with [modern-screenshot](https://github.com/qq15725/modern-screenshot) (bundled, offline), so they work in the browser and in the desktop app alike. WebGL canvases (a SoftN `Scene3D`, a page's canvas) keep their drawing buffer, so they come out in the screenshot too.

## 3D models

Any `.glb` or `.gltf` file of the project is a 3D model the **Preview** tab can show: pick it in the preview's picker, or click it in the tree.

- **The viewer** is OAIY's own, with [three.js](https://threejs.org) (MIT), bundled into one script at build time (`scripts/modelview/`, built into `public/modelview/` by the dev server and the build). It runs in a sandboxed opaque-origin frame that gets the model's bytes over `postMessage` and loads nothing else. The model is fitted to the view, in soft studio light (a room environment, a sky-and-ground fill and a key light) over a floor grid, and you turn, move and zoom it with the mouse.
- **For the agent:** `preview_screenshot` with a model's path returns one image with four labelled views (front, right, back and top) at the preview's size, drawn by the viewer itself. `yaw` and `pitch` show it from any other angle. The text gives its triangles, vertices, size and position (and whether it fits the unit cube), vertex colours, textures and materials.
- **In SoftN apps:** a model in an app's `assets/` folder shows in a `Scene3D` as an object of type `"model"` with `modelUrl: asset("assets/…")`. The runtime's `asset()` only covers images, sounds and fonts, so `scripts/softn-bridge/install.mjs` adds glTF models to it when it installs the bridge, as `blob:` URLs, which `Scene3D` accepts. The bridge answers `fetch` for those and for `data:` URLs from the bytes the frame already holds, since the runtime's CSP lets `fetch` reach only its own folder.
- A web page can't show 3D: three.js comes as ES modules or from a CDN, and neither runs on Zipp.

## SoftN apps

OAIY can build [SoftN](https://github.com/f2i-com/softn.com) apps and show them running while they're being built.

- **Start one:** use **New SoftN app** (or `/softn new`), or ask the agent for an app. The dialog starts from a small working task list, a blank page, or one of the example apps, in a new project or in a folder of this one. It reads `softn_docs` (SoftN Studio's own writing guide, regenerated with `npm run softn:guide`), writes `manifest.json`, `ui/*.ui` and `logic/*.logic`, and runs `softn_check`.
- **Reference for the agent:** SoftN is in its tools, so it doesn't have to guess.
  - `softn_docs` with no arguments gives a map of the writing guide, the published guides, every component and the example apps. It reads any of them by topic (`"guide#mistakes"`, `"xdb-data#operations"`). With `search` it finds a term across all of them, including the example apps' source, and says how to open each hit.
  - `softn_components` gives a component's exact props, events and an example.
  - `softn_examples` lists, reads or installs complete apps from softn.com's catalogue: notes, a 2048 game, a component showcase, 3D, WebGPU and device permissions.
  - All of this is bundled for offline use. Regenerate it with `npm run softn:knowledge`.
- **Live preview:** the **Preview** tab renders the app with SoftN's hosted runtime, in a sandboxed opaque-origin iframe with its own strict CSP. It re-renders about 0.7 s after edits settle.
- **Errors the agent sees and fixes:**
  - After every agent step that changes an app, OAIY checks it: the files (manifest, listed files, `.logic` syntax compiled on Zipp), then a real render. The outcome goes into that step's result, so the agent fixes what's broken before going on. You see each check as a card in the chat.
  - If the same errors come back three times in a row, the run stops rather than loop.
  - A small bridge in the preview frame (`scripts/softn-bridge/`, installed into the runtime by `npm run fetch:softn` and by the dev server and build) reports errors the running app raises, like a handler throwing when you click.
  - If the app raises an error while you're using it, a banner appears over the preview. **Fix with agent** sends the errors to the agent.
- **Plan first, then run until done:**
  - Given a task, the agent first sets a checklist with `update_plan`: the goal and 3 to 8 steps, each a piece of the result that can be finished and checked on its own (a page, a section, a feature). If it starts changing several files (or any app) without one, it's told to make one.
  - The agent works through the steps one at a time. There is always a step in progress: when one is done, the next one starts.
  - **A review after each step:** when the agent marks a step done, it's told which files it changed during that step and asked to check them before going on: read them back, then check the page or app (`page_check` and a screenshot, `softn_check`) or run the code. If the step falls short, the agent reopens it and fixes it.
  - **The plan can change:** the agent marks one step (`{"step": 2, "status": "done"}`), rewords one, or sends the whole list again to add, drop, reorder or reopen steps. It does so when you ask for something different (a message sent while it works says to), or when it finds a mistake or a missing step.
  - **Keeping direction:** after 12 steps with no change to the plan, the agent is reminded of the step it's on. It's asked to mark the step done, split it, or change its approach if it's going round in circles.
  - The checklist is pinned above the chat, with a progress bar. Finished steps fold into one line, and the step in progress shows what the agent is doing right now.
  - The agent carries the plan out without stopping to ask for permission.
  - If it stops while steps are still open, or while an app it changed still fails its check, it's asked to carry on with the step in progress. That repeats (up to 30 times in a run, within its 200 steps) as long as the agent works in between (a tool run, a file changed, a step done). It stops after two nudges in a row with nothing done. An agent that says what it will do next and then stops is told, briefly, to do it now.
  - If a run ends with steps still open, the next message you send is given that plan. The agent carries on with it if the message is about that work ("continue"), and otherwise leaves it.
- **Testing like a person:** `softn_inspect` describes what the page shows as text (headings, text, buttons, inputs and their values). `softn_interact` clicks, fills, selects and presses keys in the running app, then describes the result, so the agent can check that the app actually works. `preview_screenshot` shows the agent how it looks, and `preview_viewport` sets the screen size, as for web pages.
- **Export:** **Export .softn** (or `/softn export [folder]`) downloads an app as a `.softn` file. That's a flat zip with `manifest.json` at its root, and the manifest's `main`, `version` and `files` are filled in from the files actually present.
- **Several apps per project:** any folder whose `manifest.json` has a `.ui` `main` is an app. The preview has a picker, and `/softn new <folder>`, `/softn check <folder>`, `/softn export <folder>` and `/softn apps` take a folder. **Import .softn…** (or attaching a `.softn` in the chat) unpacks an app into a folder, so the agent can read an existing app and recreate or change it in another folder. `softn_check` takes the app's folder and switches the preview to it.

The preview runtime is optional (about 23 MB) and comes from a checksummed softn.com release: `npm run fetch:softn`. Without it, apps can still be written, checked and exported.

## Safe with your work

- **Saving:**
  - The conversation is saved as a run goes, and everything is written out when the tab is hidden or closed.
  - Closing the tab while the agent works asks first.
  - Switching projects mid-run asks, then waits for the run to stop.
  - Browser storage is requested as persistent, so the browser keeps it rather than clearing it under pressure.
- **Editing alongside the agent:** if the agent (or the terminal) changes a file while you have unsaved typing in it, the editor asks whether to keep yours or take the new version, instead of overwriting either.
- **Two tabs:** opening the same project in two tabs shows a warning in both, since each can overwrite the other.
- **Terminal:** Ctrl+C stops a running command.
- **Long chats:** a long chat opens at its latest turns, with **Show earlier** for the rest.
- **Security:**
  - The page has a Content Security Policy (scripts only from OAIY itself), which also stops script in an SVG opened in a new tab.
  - The network gate refuses redirects to this machine or its local network, however the address is written.
  - Archives are checked before they're unpacked.
  - Errors reported by a running app are passed to the agent as data, not as your words.

## Any screen

On a phone or a narrow window, the panes (Files, Editor, Preview, Terminal and Agent) become full-screen views switched from a bottom tab bar. Project actions fold into the ☰ menu.

## Run it

The Agent runs three ways. All three use the same build of the same web app.

| | How | Where your projects live |
|---|---|---|
| **On the web** | Open it from any HTTPS host (the `dist/` folder is a static site). It installs as an app from the browser, works offline and updates when you say ([Installing and updating](#installing-and-updating-the-web-app)). | That browser's storage for the site |
| **On this computer** | `npm start` builds it and serves it at http://localhost:5317 | That browser's storage for `localhost:5317` |
| **Desktop app** | The installer from `npm run desktop:build` (Windows, macOS, Linux) | The app's own webview storage |
| **Portable app** | One file from `npm run desktop:portable`: nothing to install, run it from anywhere (a USB stick) | `bot.computer-data/` beside the exe |

```sh
npm install          # also fetches and verifies the Zipp engine (public/zipp/, not committed)
npm run fetch:softn  # optional: SoftN's app preview runtime (public/softn/, not committed)
npm run dev          # develop: http://localhost:5317
npm start            # build and serve locally: http://localhost:5317
npm run build        # dist/: a static site; serve it from anywhere
npm run desktop      # the desktop app, running from the dev server
npm run desktop:build  # the desktop app and its installer (src-tauri/target/release/bundle/)
npm run desktop:portable  # one portable exe (src-tauri/target/release/bundle/portable/)
node tests/e2e/desktop.mjs  # Windows: checks the built app in WebView2 (isolation, sandbox, tray, saving on quit)
```

The Agent always uses port 5317 (`strictPort`), so a local server can allow it by origin. OAIY allows it by default.

### Installing and updating the web app

The web app is a progressive web app called **OAIY**: it has a manifest (`public/manifest.webmanifest`), OAIY's icons and a service worker (`public/sw.js`) that keeps the app and the Zipp engine for offline use.

- **Install.** Where the browser says the app can be installed (Chrome, Edge and the other Chromium browsers), the ☰ menu has **Install app**, above the project's own actions; choosing it opens the browser's own question. The item is not there once the app is installed or runs in a window of its own, in a browser that offers nothing, or in OAIY's own window and the desktop app (they carry the page themselves). On iOS Safari, which never sends that offer, the item shows one sentence instead: tap the Share button, then Add to Home Screen. Any other browser that offers nothing shows nothing, rather than instructions that may be wrong.
- **Icons.** PNG 192 and 512, a maskable 512 (the mark inside the safe zone, on the tile's gradient), a 180 px touch icon for iOS and a 32 px favicon beside the SVG one. `python scripts/make-icons.py` (Pillow) makes them from OAIY's icon, `platform/desktop/src-tauri/icons/icon.png`, and gives the same files every time.
- **Update.** Every build has its own cache, named `oaiy-agent-` and a hash of the build. A new version installs beside the running one and waits (if a file it precaches cannot be fetched, its install fails instead: the running version stays in charge, nothing is announced, and the browser tries again at the next visit): a tab that is open keeps running the version it started with, and the running version keeps its own cache. The page shows **A new version is ready** with a **Reload** button. Reload first asks the page whether it may be left: if it would object (the agent is at work, or changes are still being saved) the person is asked in the app's own dialog, and nothing is sent to the worker until they agree. Then Reload tells the waiting worker to take over and reloads the page once; nothing reloads it by itself, and if the browser's own leave prompt stops the reload, the button works again and the next click reloads. If another tab has already switched, the message says so too. When the page is shown again after more than an hour, it asks the browser for a new version (the browser also asks on every visit).
- **Old caches.** When a version takes over it keeps the cache of the version that ran before it and deletes every other cache of the app: the ones older than that, the `bot.computer-*` ones from before the app was called OAIY (unless one is the previous version's), and the ones of builds that were replaced while they waited. Nothing else on the same origin is touched. The kept cache is what lets a tab that has not reloaded carry on after another tab took the update: the worker looks in it for a file the new version does not have, so a hashed file the host no longer serves still loads. It goes when the version after next takes over, so a tab that stays open through a second update should be reloaded.
- **First visit.** The first worker has nothing to wait for and starts at once; on a host without the isolation headers the page then reloads itself once (at most once a minute) to get them.
- **From before.** The manifest's id is `/`, the id an installed copy had when the manifest had none (the origin), so a copy installed when the app was called bot.computer is the same app to the browser and takes the new name and icons when it updates, and a browser that still runs the old worker shows **A new version is ready** on its next visit. Change that id and every installed copy becomes a different app.
- **Not in OAIY's window.** There the service worker is skipped: the desktop serves the page with the isolation headers and ships its files.

`node tests/e2e/pwa.mjs` checks all of this in Chrome against the production build: the manifest and every icon it names, Chrome's installability errors (none), the old caches, "Install app" in each case above, and an update from the message to the reload.

### The desktop app

The desktop app is a [Tauri](https://tauri.app) v2 shell around the same `dist/`. It doesn't call any native commands. It runs in the system's webview: WebView2 on Windows, WebKit on macOS and Linux.

- **Its own origin.** It serves the app from its own scheme: `http://botcomputer.localhost` on Windows, `botcomputer://localhost` elsewhere.
- **Headers.** That scheme sends the headers the sandbox and the SoftN preview need (COOP, COEP, CORP, and CORS on `/softn/`), so it needs no service worker.
- **Offline.** The installer includes the Zipp engine and the SoftN runtime, so it works offline from the first start.
- **In the system tray.** Minimizing the window hides it in the tray, and so does closing it. Either way the agent keeps working in the background.
  - Click the tray icon, or start the app again, to bring the window back. Only one copy runs at a time.
  - The tray menu has **Open**, **Keep running when the window is closed** (on by default; switch it off to make closing quit), and **Quit**.
  - Quit stops the agent and saves your files and chat before exiting.
  - A hidden window isn't throttled, so long runs, video jobs and timers carry on at full speed.
- **Links.** Links that leave the app open in your browser.
- **Portable.** The executable is self-contained, with the web app, the Zipp engine and the SoftN runtime built into it (about 8 MB). `npm run desktop:portable` saves it as `bot.computer_<version>_x64-portable.exe`.
  - A copy whose name contains `portable` keeps everything in a `bot.computer-data` folder beside it: projects, chats, settings and the tray preference. Nothing is written to the user profile, so the exe and that folder can move together, for example on a USB stick.
  - It lives in the tray just like the installed app.
  - Each portable folder is its own single instance: it runs alongside the installed app, and starting the same copy again brings its window back.
  - It needs the system's WebView2 runtime. That is part of Windows 11 and current Windows 10; on an older Windows 10, install it once from Microsoft.
- **Your data.** Projects, chats and settings live in the app's webview storage, separate from any browser.
- **Building it** needs Rust and [Tauri's prerequisites](https://tauri.app/start/prerequisites/). On Windows the result is an NSIS installer (`bot.computer_<version>_x64-setup.exe`) that installs for the current user. It doesn't update itself: install a newer version over it.

Serve the web app over `localhost` or HTTPS. The sandbox needs a cross-origin isolated page, because a Worker blocks on `SharedArrayBuffer` while the page answers its file and network calls. The dev and preview servers send the headers. On a static host, the service worker adds them to every response, and the page reloads itself once on the first visit.

### Connecting a local model

Open **⚙ Settings → Add a provider**, pick the server, then **List models** and **Test**. The browser calls the server directly, so the server must allow the page's origin:

- **Ollama:** start it with `OLLAMA_ORIGINS=*` (or the exact origin you serve the Agent from).
- **LM Studio:** turn on *Enable CORS* in the server settings.
- **Others:** send `Access-Control-Allow-Origin` for the page's origin.

If you serve the Agent from a public HTTPS address and the model runs on `localhost`, Chrome's Private Network Access rules apply. Serving it locally avoids that.

### API keys

Keys are stored in IndexedDB, encrypted with a non-extractable AES-GCM key that was generated in this browser. That protects a key at rest, for example in a copied profile or a synced backup. It does not hide the key from code running on this page. Keys are sent only to their own provider, straight from the browser, and sandboxed code never sees them.

## How the sandbox works

```
page (trusted)                                   Worker (one per program)
 ├─ agent loop, UI, project files (Vfs)           └─ Zipp engine (wasm)
 ├─ SandboxHost: answers host calls  ◀── post ──     guest JS / Python / shell
 │    fs.* → the project, net.fetch → the gate        __coderHostCall(kind, …)
 └─ ReplyWriter ──▶ SharedArrayBuffer ──▶ Atomics.wait … reply
```

- **Each program gets its own Worker and a fresh engine,** with an instruction budget. The page terminates the Worker at its deadline.
- **A guest's only way out is a host call.** It rides Zipp's synchronous `localStorage.getItem` bridge with a reserved key prefix (ordinary keys are inert). The Worker blocks on shared memory until the page answers, so guest code keeps Node's synchronous `fs.readFileSync` semantics.
- **The page decides every call.** Files come from the project's virtual filesystem, which has no outside: `..` stops at the root. Network requests pass the gate, and local-network hosts need an explicit allow.
- **The guest scripts are shared with coder-cli.** `src/sandbox/guest/prelude.js` (Node-style `fs`/`path`/`fetch`/`process`) and `shell.js` (the emulated shell) come from coder-cli, where the same host calls are answered by a native process.

A dedicated host bridge in Zipp would replace the `localStorage` tunnel; see [docs/zipp-app-bridge.md](docs/zipp-app-bridge.md).

### The shell

The shell works on the project's files through the same host calls, so it can't get out of the project either.

- **Syntax.** It covers most of what bash scripts use:
  - pipes, `&&`/`||`, redirects, heredocs and here-strings;
  - `$(...)`, `$((...))`, `<(...)` and brace expansion;
  - indexed and associative arrays, functions and `local`;
  - `if`/`case`/`for`/`while`, C-style `for ((...))`, `getopts` and `trap EXIT`.
- **Files and text:**
  - `ls`, `find`, `grep -r`, `sed -i`, `awk`, `sort`, `cut`, `tr`, `column`, `paste`, `comm`, `split`;
  - `diff -u` and `patch`;
  - `xxd`, `file`, `md5sum`/`sha256sum`, `bc`, `mktemp`.
- **`jq`** covers paths, pipes, `map`/`select`/`sort_by`/`group_by`, `|=` and `del`, `reduce`, `def`, `@csv`/`@tsv`/`@base64`, and `--arg`/`-r`/`-c`/`-s`/`-n`/`-e`.
- **Archives.** `tar` (with `-z`), `zip`/`unzip`, `gzip`/`gunzip`/`zcat`. Unpacking checks entry paths and size limits.
- **`git`** keeps a real local repository in `.git/`. Blobs get git's SHA-1 ids.
  - Commands: `init`, `status`, `add`, `rm`, `mv`, `commit`, `log`, `diff`, `show`, `branch`, `switch`/`checkout`, `restore`, `reset`, `merge` (three-way, with conflict markers), `cherry-pick`, `revert`, `stash`, `tag`, `blame`, `grep`, `clean` and `describe`.
  - There are no remotes, so `push`, `pull` and `clone` say so.
  - Exporting a `.zip` leaves `.git/` out: it's in the sandbox's own format, which a real git can't read.
- **Programs.**
  - `node file.js` supports `require`, ES modules, `node_modules` and Node's core modules.
  - `python script.py` imports sibling modules, uses the working folder, and reads `input()`/stdin. It also supports `python -m`, `-c`, exit codes, and a stdlib subset (`csv`, `datetime`, `glob`, `shutil`, `urllib.parse`, `logging`, …).
  - `sh script.sh`, and scripts that start with a `#!` line, also run.

## Tests

```sh
npm test           # unit: gate, virtual filesystem, agent loop over both wire formats
npm run test:e2e   # headless Chrome: the sandbox, shell, git and tools (tests/e2e/run.mjs), the whole app with a scripted model (tests/e2e/app.mjs), web pages on Zipp with screenshots and screen sizes, 3D models in the viewer and in a SoftN Scene3D (tests/e2e/webpage.mjs), media with a mock oaiy (tests/e2e/oaiy.mjs), the production build on a static server with its service worker: isolation and offline use (tests/e2e/static.mjs), installing and updating (tests/e2e/pwa.mjs)
```

The end-to-end tests use a local Chrome or Edge (`CHROME=<path>` to choose one).

## License

Licensed under the Apache License, Version 2.0.
