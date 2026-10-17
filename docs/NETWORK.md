# OAIY Network: other computers' GPUs, through this one

OAIY Network lets one OAIY (the **primary**) use the models of other OAIY installations on the same LAN (its
**workers**): their GPUs, their engines, their models, each where it is. The primary keeps everything a person or an
app talks to (the dashboard, the Agent, Flows, the FormLogic link); a worker only makes what the primary asks of it.
It is off until switched on, and a computer with it off behaves as before.

This is the design record; [Setting it up](#setting-it-up) and [What is not done](#not-done) say what it does now.

## Decisions

| Question | Decision | Why |
| --- | --- | --- |
| Who orchestrates | A computer can use other computers (a **primary** to them), lend itself (a **worker** to them), or both at once; the computer a person works on talks to FormLogic and its apps, and uses the others through its own routes | One place for identity, permissions, audit and the cloud link per computer used; a computer with a GPU, a model or OAIY Voice to spare lends it while still using another's |
| Where inference runs | On the worker that has the model, by its own engines (the Studio), as it would locally | No model weights move; each machine's engine knows its own GPU (the Mac's eGPU path included) |
| How requests travel | Through the primary: a client calls the primary's existing routes, the primary calls the worker | Flows, the Agent and apps never see a LAN address or a worker credential |
| Discovery | mDNS/DNS-SD (`_oaiy-worker._tcp.local.`) advertising a name, a node id and a port, and a manual address | Standard and LAN-local; multicast is often blocked, so an address always works too. Discovery grants nothing |
| Trust | Pairing: each node's own key and self-signed certificate; a 6-digit code from both certificates and a fresh random value from each side, the worker's committed to before it learns the primary's, shown on both screens; approval on the worker, then on the primary (the person says the worker showed the same code); each pins the other's certificate | A numeric comparison with committed random values defeats a man in the middle on the first contact, as Bluetooth's does, with no shared secret to type or leak: a computer in the middle learns each side's value only after its own is fixed, so it cannot make the two screens agree (one time in a million). Asking both sides means a computer answering for the worker cannot pair itself by approving its own request |
| Transport | Mutual TLS (rustls) with the pinned certificates, both directions, after pairing | Encrypted and authenticated at the connection, so no bearer token crosses the LAN in the clear |
| Model identity | A remote model is `engine:<kind>:node.<nodeId>/<model>`; the worker's own name kept beside it | Never mistaken for a local model of the same name; a flow that names one shows it offline rather than switching |
| Placement | Manual: the person chooses a model on a worker, and may pin it there; nothing moves, downloads or evicts by itself | Predictable; a GPU's memory is the person's to spend |
| Flows and services | A worker runs the flows its person shares (*Let apps run it*) for its primary, and lends OAIY Voice's speech; nothing else of it | A worker's person decides what leaves it: the same mark that lets an app run a flow, and a known service's known routes, never a person's own service |
| Files | A worker's picture is streamed through the primary at a primary address (`/api/network/nodes/<id>/gateway/files/…`), and the worker's answers name their files there (`url`, `oaiy.file`), not at its own loopback address | The by-reference picture flow (docs/IMAGE_FLOW.md) keeps working; no worker URL reaches a client |
| Pins | A pin is the worker's Studio's own setting (`media.pinned`, `llm.pinned`), set by its paired primary | It outlives a restart of either computer, and the worker's engine enforces it, not a primary that may be away |

## The pieces

```
FormLogic, apps, the Agent, Flows
        │ the primary's existing routes (127.0.0.1:17972, its credentials)
        ▼
PRIMARY  network registry ─ pinned peers ─ catalogue of remote models ─ proxy
        │ mutual TLS, pinned certificates (the worker's LAN port)
        ▼
WORKER   worker listener (/worker/v1/*) ─ its own Studio (gateway + resources) ─ its GPUs
```

- **Identity** (`network/identity.rs`): a node id, a P-256 key pair and a self-signed certificate, made when Network is
  first switched on and kept in the data folder (`network/identity.json`, owner-only; one that cannot be read is an
  error, never replaced). Its fingerprint is the SHA-256 of the certificate. Pairing: the worker answers the
  primary's request with a commitment (SHA-256) to a random value of its own; the primary then tells its random
  value (`POST /worker/v1/pair/:id`, once), the worker tells its own back, and the primary checks it against the
  commitment. The code is six digits of the SHA-256 of both fingerprints and both values, so both screens show the
  same code only when each computer sees the other's real certificate. Protocol 2 (OAIY Network's first release
  paired with protocol 1: update both computers).
- **Worker connections**: 64 at once; of those, 16 at most in their TLS handshake from one address, and 16 from
  computers not paired with it (four from one address), each carrying one request (`health`, pairing) with 15 s for
  its head. A LAN computer holding connections open cannot keep the paired primary out.
- **Peers** (`network/peers.json`): each paired node's id, name, role, certificate fingerprint, last address and when
  it was paired; revoking removes it, and a revoked node's connections are refused at once.
- **Worker listener**: on a worker, `/worker/v1/*` over TLS on the LAN, bound only while it is enabled. Before pairing
  it answers only `health` and the pairing requests; after, only a paired primary's certificate is accepted, and only
  for inference, the worker's resources and its models (no files beyond the pictures its engine made, no plugins, no
  settings, no Agent).
- **Primary**: `/api/network/*` on the desktop's own API (discovery, pairing, the nodes and their resources), the
  remote models added to `GET /api/ai/engine/services`, and the routes that carry their requests and files.

## Setting it up

Both computers run OAIY Desktop (or the headless `oaiy-server`) of the same version, on the same network.

Each computer has two switches on Setup → Networking, either or both on: **Use other computers** (it is a primary
to the computers it pairs with) and **Lend this computer** (it is a worker to them). A computer can lend its GPU and
models while it uses another's OAIY Voice, say.

1. **On the computer that lends** (its GPUs, models, shared flows or OAIY Voice): switch on *Lend this computer*. It
   listens on port 17975 (change it there), only on LAN addresses, and shows the addresses another computer can type.
   Allow OAIY through the computer's firewall for that port if it asks.
2. **On the computer that uses it**: switch on *Use other computers*. Computers lending themselves show under *Add a
   computer to use* (found by mDNS); where computers cannot find each other (Wi-Fi client isolation, VLANs, a
   firewall), type the address the other computer's page shows.
3. Press **Pair**. This computer shows a six-digit code; the other's Networking page (or a prompt on whichever page
   it is on) shows the same code under *Asking to use this computer*. Approve it there only if the codes match. This
   computer then asks whether the other showed the same code: *Yes, pair* pairs them; *No* pairs nothing and the
   other forgets this computer. *Stop waiting* withdraws the request. The request expires after five minutes; one
   approved there that this computer never answered is undone there then too. Where this computer lends itself too,
   *Let it use this computer too* (ticked) pairs them both ways at once: the other may then use this computer as well
   (once *Use other computers* is on there), and is told where this computer listens.
4. Each paired computer is one card under *Paired computers*, saying which way it is paired (*This computer uses
   it*, *Uses this computer*, or both). One this computer uses shows its GPUs (memory in use as a meter and how busy
   each is), the job it is running, its models with their state and where each is loaded, the flows it shares and
   the services it lends. One that does not answer is shown offline with why, and what it last reported dimmed
   (*Last report … ago*).

Unpair from either side: both ways are forgotten, the other side refuses it from then on, and a computer that used it
shows it offline with why.
A worker unpaired at the primary while it is off is told once it answers again (while the primary runs; after a
restart of the primary, unpair it on the worker too). Switch off on either side to stop at once; switching on again
keeps the pairing.

For computers that should not use multicast at all, start OAIY with `OAIY_NETWORK_DISCOVERY=off` and pair by
address. `OAIY_NETWORK_NAME` sets the name other computers see (the computer's own name otherwise; the Networking
page sets it too).

## Using a worker's models

Once paired, a worker's models are in the primary's engine catalogue (`GET /api/ai/engine/services`), each as
`engine:<kind>:node.<nodeId>/<model>`, named `… · <worker's name>`, never a default, and marked `nodeId`,
`nodeName`, `locality: "network"`, `ready` (the worker answers now) and `pinned`.

- **Flows**: the model menus of the typed nodes (Image Gen, the AI node, video, speech…) list them beside this
  computer's; a worker that is offline is shown *(offline)*. A flow that names one runs that step on the worker,
  in the editor, from the bridge (FormLogic's oaiy-flows connector), the CLI and the Agent alike; the rest of the
  flow runs here. A picture it makes is passed on by its primary address, so the next step (a review on this
  computer, or an edit on another worker) reads it through the primary.
- **The Agent**: Settings → Agent → *On a paired computer* runs the Agent itself on a worker's language model (its
  requests and tool calls go through the primary). Its tools also see the network: `network_status` (read only)
  lists the paired computers, their GPUs and their models' ids, and `agent_model_set` can choose a worker's
  language model when the person asks. Pairing and pins stay the person's: no tool changes them.
- **Engines → Other computers** lists each paired worker's GPUs and models by kind, and makes one of them this
  computer's **default** of its kind (*Make default*; *Use this computer's* gives it back). A worker's model as the
  default is marked `default` in the catalogue in place of this computer's own, so Flows' *Default … model* settings
  use it; a worker's language model as the default answers the engine's chats
  (`/api/ai/providers/oaiy-engine/v1/chat/completions`, and chats that name no provider when none is switched on),
  and the Agent on *the model chosen in Engines* runs on it (`GET /api/agent/preferences` answers it as
  `engineDefault`). A worker's language model as the default is whichever it runs now. Choosing a default in the
  Engines tab (or with the Agent's `model_set_default`) makes this computer's own the default again: the default it
  had is kept beside the worker's, and one chosen since takes its place. Unpairing the worker gives its kinds back. A
  default on a worker that is offline says so when it is used; one not heard from since this computer started
  leaves this computer's own in place until it is. Set with `PUT /api/network/settings`
  `{"defaults": {"image": "engine:image:node.<id>/<model>"}}` (`null` gives it back), listed in `GET
  /api/network/status` as `defaults`; the Agent's `model_set_default` takes a worker's model's id too.
- **Any client of this desktop** reaches a worker's media routes at `/api/network/nodes/<id>/gateway/v1/…` and its
  chat at `/api/network/nodes/<id>/v1/chat/completions` (streamed as the worker streams it), with the desktop's own
  credentials (scope `ai.use`).

A model that is offline, or a worker that refuses, answers `503 node_offline` naming the worker and why
(*connection refused*, *the worker no longer accepts this computer*): nothing falls back to another model. A chat
naming a language model the worker no longer runs (its person chose another) is `409 model_unavailable`, naming
the one it runs, rather than answered by that one. A media answer names its files at the primary's route for the
worker, as the caller reached the primary (`http://<host>/api/network/nodes/<id>/gateway/files/…`).

## A worker's flows

A worker lends its primary the flows it **shares**: the ones its person marked *Let apps run it* in its Flows editor
(the same mark that lets FormLogic run a flow), and no other. They are listed with the worker (its card on the
Networking page, `GET /api/network/nodes`, the Agent's `network_status`) and run there, by the worker's own runner,
models and GPUs:

- **Flows → On other computers**: each worker's shared flows with their inputs; *Run on <worker>* starts one, the run
  is followed as it goes (and can be stopped), and its answer shows when it is done: its pictures, read through the
  primary, and its words.
- **The Agent**: `flows_list` with `node` lists a worker's shared flows, and `flow_run` with `node` runs one there
  (waiting for it by default, its pictures handed to the Agent as images).
- **Routes**: `GET /api/network/nodes/:id/flows`, `POST /api/network/nodes/:id/flows/:flow/runs`
  (`{input?, timeoutMs?, wait?, key?}`: answered at once with the run, or with `wait` once it has finished),
  `GET /api/network/nodes/:id/runs/:run`, `POST /api/network/nodes/:id/runs/:run/cancel`.

A run is kept in the worker's own run ledger (its Run history), under the primary that asked: a primary reads and
stops its own runs alone. Pictures in a run's input that are this primary's are sent inline (the worker cannot read
the primary's addresses); the pictures a run makes are named at the primary's route for the worker.

## A worker's services: OAIY Voice

A worker lends its primary **OAIY Voice** when it is set up there: **Services → On other computers** lists it, and
*Use it here* makes it this computer's speech in place of its own (*Use this computer's* gives it back). The
computer connected to the phone stays the primary: its calls, the transcription route (`/api/voice/transcribe`, the
flow editor's speech-to-text and the Agent's listening) then send their speech-to-text and speech to the worker,
which starts its OAIY Voice when asked. A call's voice is copied to the worker the first time it is used there (kept
there as `from-<primary>-<name>`, beside the worker's own voices). A plugin's setup that asks for OAIY Voice counts
the worker's. Set with `PUT /api/network/settings` `{"services": {"oaiy-voice": "<nodeId>"}}` (`null` gives it back),
listed in `GET /api/network/status` as `services`.

The worker lends that service's speech routes alone (`/worker/v1/services/oaiy-voice/…`: its models and voices,
`/v1/audio/transcriptions`, `/v1/audio/speech`, streamed as it speaks), and no other service: one a person added
could be anything.

## Pinning

A worker's model can be **pinned** from the primary's Networking page (*Pin* beside the model): it is loaded on the
worker and kept loaded there.

- A picture model (Qwen Image, MOSS-Image, FLUX 3 Image Klein on WebGPU) is pinned in the worker's Studio
  (`media.pinned`, crates/oaiy-studio/src/media/pins.rs): its picture worker is kept whatever the keep time or a
  release asks; it is loaded when the Studio starts, and again whenever its worker ends (a failure, a cancelled
  picture, a crash while it waited), by a small picture made when nothing else is queued; one that cannot load (or
  cannot be pinned any more, its settings changed) is shown *could not load* with why, kept, and tried again every
  five minutes. Unpinned while it loads, its loading job is cancelled.
- A job that would let a pinned model go is refused, saying which pin holds the GPU: another family's picture that
  the worker cannot keep beside it, a video, speech or music job on the same GPU, and a chat for a language model
  on the eGPU's card while a pinned picture model holds the card. A job refused for a pin on the card leaves the card
  lent to the pinned worker (the card's language model is not brought back beside it). MOSS-Image's and Klein's are
  kept side by side, so both can be pinned together; a second model of a family, or Qwen Image beside them, is
  refused at pinning. A pin on this computer's own GPU lets a language model there come back beside it.
- The language model chosen on the worker can be pinned too (`llm.pinned`: never stopped for media or for being
  idle). Another of its language models is chosen in the worker's own Engines first.

Unpin to let it go as any other model. The Studio's own routes are `POST /api/resources/pins` and
`POST /api/resources/unpin` (`{model, kind}`), and `GET /api/resources` lists `pins` with each one's state.

## Routes

The primary (`/api/network/*`, on the desktop's own API; in legacy access mode changes are privileged and reads
restricted, as `/api/engines/` are):

| Route | Scope | What |
| --- | --- | --- |
| `GET /api/network/status` | auth.read | On or off, the role, this computer's name, fingerprint and addresses |
| `PUT /api/network/settings` | auth.manage | `{enabled, uses, lends, name, port, bind, defaults, services}` (`role`: one of `primary`, `worker`, `both`, as before) |
| `POST /api/network/discover` | auth.manage | The workers found on the LAN |
| `POST /api/network/pairings/start` | auth.manage | `{address}` or `{nodeId}`: the code to compare |
| `POST /api/network/pairings/confirm` | auth.manage | `{pairingId, accept?}`: `pending`, `approved_there` (the worker approved), then with `accept: true` paired here (`false`: not, and the worker forgets it, or withdraws the request it has not answered) |
| `GET /api/network/requests` | auth.read | On a worker: primaries asking to pair, each with its code |
| `POST /api/network/requests/:id/approve`, `…/deny` | auth.manage, auth.revoke | The worker's person decides |
| `GET /api/network/nodes` | auth.read | The paired computers, online or not, their GPUs and models |
| `DELETE /api/network/nodes/:id` | auth.revoke | Unpair |
| `GET /api/network/nodes/:id/resources` | auth.read | A worker's GPUs, models and queue, now |
| `PUT /api/network/nodes/:id/pins`, `DELETE …/pins/:model` | auth.manage | Pin a worker's model (`{id}` or `{model, kind}`), unpin it |
| `POST /api/network/nodes/:id/v1/chat/completions` | ai.use | Its language model, streamed |
| `ANY /api/network/nodes/:id/gateway/*path` | ai.use | Its media routes, jobs and files |
| `GET /api/network/nodes/:id/flows` | flows.read | The flows it shares |
| `POST /api/network/nodes/:id/flows/:flow/runs` | runs.write | One run there (`{input?, timeoutMs?, wait?, key?}`) |
| `GET /api/network/nodes/:id/runs/:run`, `POST …/cancel` | runs.read, runs.write | A run there, as it is now; stopped |

The worker (`/worker/v1/*` on its LAN port, TLS with client certificates): `health` (anyone: a name, a node id, the
protocol), `pair` and `pair/:id` (anyone with a certificate, to ask), `unpair` (the caller's own pairing, both ways,
or request withdrawn), and, for its paired primary only, `lends` (it lends itself to this one too: paired both ways), `capabilities`, `resources`, `pins`, `chat`, `gateway/*path`, `flows`,
`flows/:id/runs`, `runs/:id` (and `…/cancel`), `services/oaiy-voice/*path` and `voices/:name` (a voice copied there). The gateway takes the engine's media routes and
files only (not its settings, logs, downloads or playground), and what it answers about itself leaves out its local
paths and loopback addresses.

## Testing it on one computer

Two `oaiy-server`s on one computer can pair by address over loopback, both beside the same engines:

```
OAIY_DATA_DIR=/tmp/oaiy-primary OAIY_SERVER_PORT=17973 OAIY_ENGINES_UI=http://127.0.0.1:7860 \
  OAIY_NETWORK_DISCOVERY=off OAIY_NETWORK_NAME="Test primary" OAIY_SERVER_TOKEN=… oaiy-server
OAIY_DATA_DIR=/tmp/oaiy-worker OAIY_SERVER_PORT=17984 OAIY_ENGINES_UI=http://127.0.0.1:7860 \
  OAIY_NETWORK_DISCOVERY=off OAIY_NETWORK_NAME="Test worker" OAIY_SERVER_TOKEN=… oaiy-server
```

Then `PUT /api/network/settings` `{"enabled": true, "role": "worker", "port": 17986}` on the worker,
`{"enabled": true, "role": "primary"}` on the primary, `POST /api/network/pairings/start` `{"address":
"127.0.0.1:17986"}` on the primary, approve the request on the worker, and confirm on the primary
(`POST /api/network/pairings/confirm` `{"pairingId": …, "accept": true}` once it answers `approved_there`).

Measured that way on the Mac with its RTX 4090 eGPU (2026-10-18; both servers on the Mac, the worker's engines the
Mac's own):

| What | Result |
| --- | --- |
| Pairing | The same code on both sides; nothing paired before the worker approved |
| Catalogue | 24 of the worker's models listed beside the primary's 25, none default, no local path or loopback address in them |
| A picture (MOSS-Image Q8 8-step, 1024²) through the primary | 10.6 s; its file read through the primary, byte for byte the worker's |
| Text to image (the flow) with its picture step on the worker | 27 s, passed first time, the review reading the picture by its primary address |
| A chat streamed from the worker's language model | streamed as it came |
| The Agent's kind of request (a tool call) through the worker | `tool_calls` answered in 1.9 s |
| A chat stream closed by the client | The worker's engine stopped within about a second (its CPU 64% → 4%), not at the answer's end |
| The worker stopped | `503 node_offline`, *connection refused*; the flow failed saying so, nothing fell back |
| Both restarted | The worker listened again by itself; the primary saw it online within 2 s |
| Pinning MOSS-Image Q8 8-step from the primary | Loaded on the card by itself in 33 s; kept past a release and a picture asking for release (10.8 s, nothing reloaded); a Qwen Image picture refused at once, naming the pin; unpinned again |
| Pairing, both screens | `approved_there` after the worker's yes, paired only on the primary's yes |

The desktop's tests cover the same with a fake engine (src/network/tests.rs: pairing and use, the committed pairing
code, a pairing given up or run out, a stranger and an impostor, unpairing from each side and while a worker is
off, pins, paths that would leave the gateway (dot segments, tabs and newlines, encoded slashes), a worker saved
again on its port, off and public addresses refused), and the Studio's the pins with a fake picture worker
(`a_pinned_picture_model_is_loaded_kept_and_never_let_go_by_a_job`) and a fake eGPU card
(`a_pin_on_the_card_keeps_it_lent_through_a_job_refused_for_it`).

## Not done

- **The Engines tab itself (the engines' own page) lists this computer's models alone**: a worker's are on the
  *Other computers* tab beside it. Plugins' text completions through the engine keep this computer's language model,
  whatever the default.
- **No Load now / Release when idle**: a pin loads a model and keeps it; there is no loading without pinning, and
  no per-pin device, priority, memory budget or concurrency limit (a pin goes where the worker's pictures go).
- **Pins are picture models and the chosen language model**: video, speech, music, sound and 3D models run in a
  worker of their own per job, so they cannot be kept loaded.
- **No routing trace or debug panel**, no per-worker queue limits beyond the worker's own Studio queue, and no
  request ids carried from the primary to the worker's jobs.
- **Credential rotation**: a computer's identity is made once; to replace it, delete `network/` in its data folder
  and pair again.
- **Not measured on a second physical computer** yet: the tests above ran both sides on one Mac (loopback), and
  discovery by mDNS was not exercised between two computers.
- `network/` is left out of backups: a restored computer is a new one to its peers, and pairs again.
