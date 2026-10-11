"""The eGPU plugin's process for OAIY Desktop: what an NVIDIA card in a Thunderbolt enclosure needs on a Mac, said to
OAIY (docs/TINYGPU.md, "The eGPU plugin").

The plugin's folder has it all: a Python (python/), tinygrad (lib/), the card's server and its launcher (card/), the
card's kernels compiled ahead of time (card/kernels/), and what tinygrad would download as it first opens a card
(card/firmware/), so a Mac needs neither Docker, nor a compiler, nor the network. This process holds nothing: it
answers OAIY (JSON-RPC, a message a line, on standard input and output), and its `card.runtime` command tells OAIY's
engines where those are. The engines start the launcher themselves, which holds the card: a plugin is stopped within
seconds of being asked (then killed), and a card's holder killed has left its link down.

    connector egpu: card.runtime  -> {python, pythonpath, scripts, kernels, firmware, pack, tinygrad, driver, version}
                    card.status   -> the same, and whether the card can be used ({ready, why, state, detail})
"""
import json, os, pathlib, subprocess, sys

CARD = pathlib.Path(__file__).resolve().parent
ROOT = CARD.parent
VERSION = "0.1.0"


def driver() -> dict:
    """tinygrad's TinyGPU driver: a system extension, installed with TinyGPU.app and allowed in System Settings."""
    try:
        listed = subprocess.run(["/usr/bin/systemextensionsctl", "list"], capture_output=True, text=True, timeout=10).stdout
    except (OSError, subprocess.SubprocessError):
        listed = ""
    line = next((l for l in listed.splitlines() if "org.tinygrad.tinygpu" in l), "")
    return {"installed": bool(line), "enabled": "[activated enabled]" in line, "app": os.path.isdir("/Applications/TinyGPU.app")}


def runtime() -> dict:
    index = CARD / "kernels" / "index.json"
    tinygrad = ROOT / "lib" / "tinygrad-commit.txt"
    return {
        "python": str(ROOT / "python" / "bin" / "python3.12"),
        "pythonpath": str(ROOT / "lib"),
        "scripts": str(CARD),
        "kernels": str(CARD / "kernels"),
        "firmware": str(CARD / "firmware"),
        "pack": json.loads(index.read_text()) if index.exists() else None,
        "tinygrad": tinygrad.read_text().strip() if tinygrad.exists() else None,
        "driver": driver(),
        "version": VERSION,
    }


def status() -> dict:
    r = runtime()
    why = None
    if not pathlib.Path(r["python"]).is_file():
        why = "the plugin's Python is missing: install the plugin again"
    elif not r["pack"]:
        why = "the plugin has no kernel pack: install the plugin again"
    elif not r["driver"]["installed"]:
        why = "tinygrad's TinyGPU driver is not installed: open TinyGPU.app, then allow its extension in System Settings"
    elif not r["driver"]["enabled"]:
        why = "tinygrad's TinyGPU driver is installed but not enabled: allow it in System Settings, General, Login Items & Extensions"
    pack = r["pack"] or {}
    have = f"tinygrad {(r['tinygrad'] or '?')[:9]}, {pack.get('kernels', 0):,} kernels for {', '.join(pack.get('archs', [])) or 'no card'}"
    return {**r, "ready": why is None, "why": why, "state": "Ready" if why is None else "Not ready", "detail": why or have}


def reply(message_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        out["error"] = {"code": -32601, "message": error}
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def main():
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            message = json.loads(line)
        except ValueError:
            continue
        method, message_id, params = message.get("method"), message.get("id"), message.get("params") or {}
        if method == "plugin.shutdown":
            return
        if message_id is None:
            continue
        if method == "plugin.init":
            reply(message_id, {"name": "eGPU", "version": VERSION})
        elif method == "plugin.health":
            # (the card is OAIY's engines' to use: unready is said by `status`, not as the plugin's ill health)
            reply(message_id, {"status": "ok"})
        elif method == "connector.request":
            command = params.get("command")
            if command == "card.runtime":
                reply(message_id, {"ok": True, "data": runtime()})
            elif command == "card.status":
                reply(message_id, {"ok": True, "data": status()})
            else:
                reply(message_id, {"ok": False, "error": {"code": "invalid_request", "message": f"no command {command!r}"}})
        else:
            reply(message_id, error=f"no method {method!r}")


if __name__ == "__main__":
    main()
