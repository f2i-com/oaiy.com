# Example flows

Sample flow files — each is a minimal OAIY **project** (a top-level `flows` array
wrapping one flow graph) so the same file both runs headlessly with the CLI
(`cli/`) **and** imports cleanly into the web app. They drive **managed AI
services** installed via the companion / `oaiy-server` (see `desktop/README.md`,
`cli/README.md`).

| File | What it does | Services needed |
|---|---|---|
| `krea2-text-to-image.json` | Renders an image from a prompt with **Krea-2 Turbo**. | `krea2` (port 17910) |
| `llm-enhance-to-krea2.json` | An **LLM (llama.cpp)** expands a terse idea into a detailed prompt, which feeds **Krea-2** to render the image. | `llama-cpp` (8080, a GGUF selected) + `krea2` (17910) |

The `service_call` nodes carry their HTTP call **inline** (`endpoint` /
`bodyTemplate` / `responsePath`), so a file is self-contained — it runs the same
whether or not the companion's service list is populated.

## Open in the web app

In the **Flows** sidebar click **Load from File** and pick a file — it's added
as a new flow (your current project is untouched), then hit **Run**. (The header
**Import project** button also works, but it *replaces* the whole project.)

Make sure the services are running first (start them from the desktop
companion's **Services** tab, or the CLI below).

## Run with the CLI

Install + start the services, then run a flow:

```bash
# one-time: install + start the services (via the CLI or the desktop app)
oaiy service install krea2 && oaiy service start krea2
oaiy service install llama-cpp && oaiy service start llama-cpp   # needs a model — see the note below

# run a flow (the generated image's URL is printed; it serves from the krea2 service)
oaiy run examples/krea2-text-to-image.json
oaiy run examples/llm-enhance-to-krea2.json
```

> The `llm-enhance-to-krea2` flow calls llama.cpp's OpenAI-compatible
> `/v1/chat/completions` on port 8080. llama.cpp loads **one** model and refuses
> to start until you pick it. In the **desktop app**, choose a GGUF in the
> **Model selector** on the llama.cpp card. **Headless** (CLI against an
> `oaiy-server`) there's no Model selector — start the server with
> `OAIY_LLAMACPP_MODEL=/path/to/model.gguf` (e.g. one of yours in `E:\models`),
> the headless equivalent. Either way, start the service before running.

The image flows return an `imageUrl` pointing at the krea2 service's `/file`
endpoint, loadable directly in a browser (or fetched by the engine). Edit the
`template` node's text — or wire an `input_text` node — to change the prompt.
The `llm-enhance-to-krea2` flow also outputs the LLM's `enhancedPrompt` so you
can see what it sent to the image model.
