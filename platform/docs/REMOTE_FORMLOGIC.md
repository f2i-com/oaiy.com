# FormLogic through your linked OAIY computer

OAIY can run on a different computer from the browser using FormLogic. The
desktop makes outbound HTTPS requests to FormLogic's account-scoped queues;
FormLogic relays requests and replies. Do not expose port 17972 to the internet.

## Setup

1. Install the OAIY build with encrypted flow support and configure a provider
   or local model. Keep the desktop running and the computer awake.
2. Open **Connections → Linked account**, choose FormLogic and enter the address
   of your deployment, such as `https://formlogic.com`.
3. Complete the account approval in the browser. A browser pairing is separate
   and is only needed for direct access on the same computer.
4. On any device, sign into that FormLogic account. In **Connect your AI**, choose
   OAIY and **Through my FormLogic account**, select a provider and check setup.
   The setup check reads the provider catalogue; try a chat to verify inference.
5. For an automation, save the flow, choose **Desktop relay**, open its test
   panel and select **Linked computer**. The saved server graph is executed.

The default target uses the account's assignment, or its single online computer.
When multiple computers are available, choose one explicitly or configure
**Settings → AI & devices → Linked desktops**. An offline selected computer must
not silently redirect work to another machine.

## What is encrypted

| Traffic | Protection and visibility |
| --- | --- |
| AI chat, provider/model catalogue | Browser-to-OAIY authenticated encryption, carried over HTTPS. FormLogic routes ciphertext. |
| Desktop relay flow inputs and terminal output/errors | Same end-to-end encryption. OAIY checks that the encrypted flow ID matches the queued flow ID. |
| Flow definitions, routing metadata and status | Authenticated HTTPS; visible to FormLogic. The desktop fetches saved graphs with its linked account credential. |
| Service/plugin commands, records and event-triggered queued flows | Authenticated HTTPS; these existing lanes are not end-to-end encrypted. |

The relay reuses the existing X25519 / XSalsa20-Poly1305 identity and per-request
ephemeral keys. Browsers pin the desktop's key on first use and reject an
unexpected change. Initial key delivery relies on the authenticated FormLogic
site. This is an application relay, not a general network tunnel.

A flow can intentionally submit records or make API calls; data explicitly sent
to FormLogic by those steps is visible to its backend. Linking a computer grants
scoped access to the account, not unrestricted access to anonymous site visitors.
Existing app roles, connector permissions and device approvals still apply.

## Reliability and limits

- OAIY claims each request before running it. A lost claim never executes.
- Terminal replies retry using identical ciphertext; failed delivery never
  reruns a flow that may already have changed records or contacted someone.
- Execution has a five-minute budget. Terminal JSON is capped at 192 KiB.
- This worker reports queue/running/terminal state; it does not stream individual
  CLI node progress yet. The browser must not show an invented node count.
- **Connections** shows whether encrypted-flow polling is active or failing.
  The AI and flow workers share one persistent identity. Relinking republishes
  its public key for the new account connection.
- If a completion cannot be confirmed, inspect the application's records/logs
  before retrying manually. A disconnected browser does not undo side effects.

## Developer verification

Debug desktop builds simulate plugin hardware by default. To test Aokie with an
actual paired phone, launch the debug executable with `OAIY_PLUGIN_DEV_MODE=0`
in its process environment. Preserve that setting on every restart; otherwise
the AI relay can work while the phone plugin is only simulated. Release builds
use real hardware by default. Verify the phone connection and radio diagnostics
in **AI Receptionist** before a live call; a running plugin alone is insufficient.

From `desktop/src-tauri`, run `cargo test --no-default-features --lib`.
The sealed-flow tests use an isolated HTTP relay and real envelope encryption to
check claims, encrypted results/errors, unlinking and completion retries.

After building `cli/`, set `OAIY_CLI` to its absolute `dist/oaiy.mjs` path and run:

```text
cargo test --no-default-features --lib link::sealed_flows::tests::encrypted_relay_executes_the_real_zipp_cli -- --ignored
```

That test executes a real flow through the CLI and decrypts the final result.
It does not require an installed model, a live FormLogic account or browser
access to OAIY's loopback API. Live deployment and a physical second-device
test are separate checks.
