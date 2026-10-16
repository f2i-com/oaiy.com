"""The eGPU plugin's process for OAIY Desktop: what an NVIDIA card in a Thunderbolt enclosure needs on a Mac, said to
OAIY, and its setup (docs/TINYGPU.md, "The eGPU plugin").

The plugin's folder has it all: a Python (python/), tinygrad (lib/), the card's server and its launcher (card/), the
card's kernels compiled ahead of time (card/kernels/), and what tinygrad would download as it first opens a card
(card/firmware/), so a Mac needs neither Docker, nor a compiler, nor the network. This process holds nothing: it
answers OAIY (JSON-RPC, a message a line, on standard input and output), and its `card.runtime` command tells OAIY's
engines where those are. The engines start the launcher themselves, which holds the card: a plugin is stopped within
seconds of being asked (then killed), and a card's holder killed has left its link down.

Its setup (OAIY's wizard, and the plugin's eGPU page) puts in what the folder cannot: tinygrad's TinyGPU driver (its
app in /Applications, and the driver extension it asks macOS for, which the person allows in System Settings), and, for
kernels the pack lacks (a model newer than the plugin), NVIDIA's compiler in Docker, as tinygrad's
extra/setup_nvcc_osx.sh sets it up (its image, and its shim in ~/.local/bin, where the card's server looks for it).
What takes long runs on a thread of its own, one at a time, and `setup.status` says how it goes.

    connector egpu: card.runtime        -> {python, pythonpath, scripts, kernels, firmware, pack, tinygrad, driver, version}
                    card.status         -> the same, and whether the card can be used ({ready, why, state, detail})
                    setup.status        -> {driver, card, docker, compiler, pack, missing, job, steps}
                    driver.install      -> TinyGPU.app put in /Applications and its driver asked of macOS (a job)
                    driver.openSettings -> System Settings at Login Items & Extensions, where the driver is allowed
                    docker.download     -> Docker Desktop's download page, in the browser
                    docker.start        -> Docker Desktop started
                    compiler.install    -> NVIDIA's compiler in Docker: tinygrad's image built, its shim put in (a job)
"""
from __future__ import annotations

import collections, json, os, pathlib, re, shutil, subprocess, sys, threading, time, urllib.request

CARD = pathlib.Path(__file__).resolve().parent
ROOT = CARD.parent
VERSION = "0.2.0"
HOME = pathlib.Path.home()
# (where OAIY keeps this plugin's own files: what it downloads, never its folder, which is trusted by its digest)
DATA = pathlib.Path(os.environ.get("OAIY_PLUGIN_DATA_DIR") or HOME / "Library" / "Application Support" / "com.oaiy.app" / "plugin-data" / "egpu")
TINYGPU_APP = pathlib.Path("/Applications/TinyGPU.app")
TINYGPU_PROGRAM = TINYGPU_APP / "Contents" / "MacOS" / "TinyGPU"
EXTENSIONS_SETTINGS = "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
DOCKER_PAGE = "https://www.docker.com/products/docker-desktop/"
DOCKER_BINS = ("/Applications/Docker.app/Contents/Resources/bin", str(HOME / ".docker" / "bin"), "/usr/local/bin", "/opt/homebrew/bin")
# the card's server's kernel cache, and the kernels a pack lacked kept there (tools/tinygpu/webgpu_server.py)
MISSING = pathlib.Path(os.environ.get("TINYGPU_CACHE") or HOME / ".cache" / "tinygpu-webgpu") / "missing"
SHIMS = HOME / ".local" / "bin"
# NVIDIA's compiler for Linux on arm64, as tinygrad's extra/setup_nvcc_osx.sh builds it
IMAGE = "cuda-nvcc:12.8"
DOCKERFILE = """FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends wget ca-certificates && \\
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/sbsa/cuda-keyring_1.1-1_all.deb && \\
  dpkg -i cuda-keyring_1.1-1_all.deb && \\
  apt-get update && apt-get install -y --no-install-recommends cuda-nvcc-12-8 cuda-nvdisasm-12-8 cuda-cuobjdump-12-8 && rm -rf /var/lib/apt/lists/*
ENV PATH=/usr/local/cuda/bin:$PATH
"""
# tinygrad's shim, with Docker's own folders on its PATH (OAIY starts the card's server with a short one)
SHIM = """#!/bin/sh
set -eu
PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin:$HOME/.docker/bin:/usr/local/bin:/opt/homebrew/bin"
cname="cuda-nvcc-persistent"
if ! docker inspect --format='{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; then
  docker rm -f "$cname" >/dev/null 2>&1 || true
  docker run -d --platform=linux/arm64 --name "$cname" \\
    -v /var/folders:/var/folders -v "$HOME":"$HOME" \\
    cuda-nvcc:12.8 sleep 300 >/dev/null
fi
exec docker exec "$cname" "$(basename "$0")" "$@"
"""
# Cards by their PCI device id: some names, and every family's architecture (a range of ids a family).
NAMES = {0x2684: "RTX 4090", 0x2702: "RTX 4080 SUPER", 0x2704: "RTX 4080", 0x2782: "RTX 4070 Ti", 0x2786: "RTX 4070",
         0x2204: "RTX 3090", 0x2206: "RTX 3080", 0x2B85: "RTX 5090", 0x2C02: "RTX 5080"}
FAMILIES = [(0x2300, 0x233F, "sm_90", "Hopper"), (0x2200, 0x25FF, "sm_86", "RTX 30 series (Ampere)"),
            (0x2600, 0x28FF, "sm_89", "RTX 40 series (Ada)"), (0x2900, 0x29FF, "sm_100", "Blackwell"),
            (0x2B00, 0x2FFF, "sm_120", "RTX 50 series (Blackwell)")]


def quiet(argv: list, timeout: float, env: dict | None = None) -> subprocess.CompletedProcess:
    """`argv` run with its output kept; a program missing, or past `timeout`, as a failed run."""
    try:
        return subprocess.run(argv, capture_output=True, text=True, timeout=timeout, env=env)
    except (OSError, subprocess.SubprocessError) as e:
        return subprocess.CompletedProcess(argv, 127, "", str(e))


def driver() -> dict:
    """tinygrad's TinyGPU driver: a system extension, installed with TinyGPU.app and allowed in System Settings."""
    lines = [l for l in quiet(["/usr/bin/systemextensionsctl", "list"], 10).stdout.splitlines() if "org.tinygrad.tinygpu" in l]
    live = [l for l in lines if "terminated" not in l and "uninstall" not in l]
    enabled = any("[activated enabled]" in l for l in live)
    installed = bool(live)
    state = "allowed" if enabled else "waiting" if installed else "absent"
    said = next((l.rsplit("\t", 1)[-1].strip() for l in live), "")
    return {"installed": installed, "enabled": enabled, "app": TINYGPU_APP.is_dir(), "state": state, "said": said}


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
        why = "tinygrad's TinyGPU driver is not installed: the eGPU's setup installs it (Plugins, eGPU, Set up)"
    elif not r["driver"]["enabled"]:
        why = "tinygrad's TinyGPU driver is installed but not allowed: allow it in System Settings, General, Login Items & Extensions"
    pack = r["pack"] or {}
    have = f"tinygrad {(r['tinygrad'] or '?')[:9]}, {pack.get('kernels', 0):,} kernels for {', '.join(pack.get('archs', [])) or 'no card'}"
    return {**r, "ready": why is None, "why": why, "state": "Ready" if why is None else "Not ready", "detail": why or have}


_seen_card = [0.0, None]


def card() -> dict:
    """The NVIDIA card macOS sees on the PCI bus (system_profiler's, kept 8 s: it takes a second or two), its link,
    and its architecture by its device id."""
    if _seen_card[1] is not None and time.monotonic() - _seen_card[0] < 8:
        return _seen_card[1]
    items = []
    def walk(x):
        if isinstance(x, dict):
            if "sppci_vendor-id" in x:
                items.append(x)
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    try:
        walk(json.loads(quiet(["/usr/sbin/system_profiler", "SPPCIDataType", "-json"], 30).stdout or "{}"))
    except ValueError:
        pass
    gpus = [i for i in items if str(i.get("sppci_vendor-id", "")).lower() == "0x10de" and i.get("sppci_device_type") == "sppci_vgacontroller"]
    found = {"found": False}
    if gpus:
        g = gpus[0]
        try:
            device = int(str(g.get("sppci_device-id", "0")), 16)
        except ValueError:
            device = 0
        family = next(((arch, name) for lo, hi, arch, name in FAMILIES if lo <= device <= hi), (None, "NVIDIA"))
        link = g.get("sppci_link-status", "")
        found = {
            "found": True,
            "device": f"0x{device:04x}",
            "name": f"NVIDIA {NAMES[device]}" if device in NAMES else f"NVIDIA {family[1]} card (device 0x{device:04x})",
            "arch": family[0],
            "family": family[1],
            "link": link,
            "up": link.lower() == "link up",
            "speed": g.get("sppci_link-speed", ""),
            "width": g.get("sppci_link-width", ""),
            "slot": g.get("sppci_slot_name", ""),
        }
    _seen_card[:] = [time.monotonic(), found]
    return found


def docker_cli() -> str | None:
    """Docker's own program: Docker Desktop's, or one on the PATH."""
    for d in DOCKER_BINS:
        if os.access(p := os.path.join(d, "docker"), os.X_OK):
            return p
    return shutil.which("docker")


def docker_env() -> dict:
    # (its credential helper, docker-credential-desktop, beside it: a pull from Docker Hub asks it)
    return dict(os.environ, PATH=os.pathsep.join([os.environ.get("PATH", "/usr/bin:/bin"), *(d for d in DOCKER_BINS if os.path.isdir(d))]))


def docker() -> dict:
    cli = docker_cli()
    running = image = False
    if cli:
        running = bool(quiet([cli, "info", "--format", "{{.ServerVersion}}"], 8, docker_env()).stdout.strip())
        image = running and quiet([cli, "image", "inspect", "--format", "{{.Id}}", IMAGE], 8, docker_env()).returncode == 0
    return {"app": os.path.isdir("/Applications/Docker.app"), "cli": cli, "running": running, "image": image}


CHECKED = DATA / "compiler.json"


def compiler(d: dict) -> dict:
    """NVIDIA's compiler as the card's server finds it: on the PATH, else the shim in ~/.local/bin, ready where its
    image is in Docker (seen now, or when Docker last ran: `CHECKED`, so a Docker not running is not a step undone)."""
    shim = SHIMS / "nvcc"
    present = os.access(shim, os.X_OK)
    on_path = shutil.which("nvcc")
    native = bool(on_path) and pathlib.Path(on_path).resolve() != (SHIMS / "nvccshim").resolve()
    if d["image"] and not CHECKED.is_file():
        try:
            CHECKED.parent.mkdir(parents=True, exist_ok=True)
            CHECKED.write_text(json.dumps({"image": IMAGE, "seen": time.time()}))
        except OSError:
            pass
    built = d["image"] or (not d["running"] and CHECKED.is_file())
    return {"shim": present, "native": native, "at": on_path if native else str(shim) if present else None,
            "built": built, "ready": native or (present and built), "image": IMAGE}


def missing() -> int:
    """The kernels a model asked for that the pack lacked, with no compiler here (kept for the next pack)."""
    try:
        return sum(1 for _ in MISSING.glob("*.cu"))
    except OSError:
        return 0


class Job:
    """One setup job on a thread of its own: its step, the last of what it said, and how it ended."""

    def __init__(self, name: str):
        self.name, self.state, self.step, self.error = name, "running", "", None
        self.log = collections.deque(maxlen=60)
        self.started, self.finished = time.time(), None

    def say(self, text: str):
        for line in str(text).splitlines():
            if line.strip():
                self.log.append(line.rstrip()[:300])

    def view(self) -> dict:
        return {"name": self.name, "state": self.state, "step": self.step, "error": self.error, "log": list(self.log),
                "seconds": round((self.finished or time.time()) - self.started)}


_job = [None]
_job_lock = threading.Lock()


def start(name: str, work) -> dict:
    """`work(job)` on a thread, unless a job runs already."""
    with _job_lock:
        running = _job[0]
        if running is not None and running.state == "running":
            return {"started": False, "busy": running.name, "job": running.view()}
        job = _job[0] = Job(name)

    def run():
        try:
            work(job)
            job.state = "done"
        except Exception as e:  # (said on the page: the job's end, never the plugin's)
            job.state, job.error = "failed", str(e)
            job.say(f"failed: {e}")
        finally:
            job.finished = time.time()

    threading.Thread(target=run, name=f"egpu-{name}", daemon=True).start()
    return {"started": True, "job": job.view()}


def tinygpu_zip(job: Job) -> pathlib.Path:
    """TinyGPU.app's zip: the plugin's own (card/firmware), else the one tinygrad's commit names, downloaded from
    tinygrad's releases (as tinygrad itself would) into the plugin's data folder."""
    carried = sorted((CARD / "firmware").glob("TinyGPU_*.zip"))
    if carried:
        job.say(f"TinyGPU.app from the plugin ({carried[-1].name})")
        return carried[-1]
    system = (ROOT / "lib" / "tinygrad" / "runtime" / "support" / "system.py").read_text()
    commit = re.search(r'def ensure_app[\s\S]*?commit = "([0-9a-f]{40})"', system)
    if not commit:
        raise RuntimeError("this tinygrad does not name its TinyGPU app")
    kept = DATA / "downloads" / f"TinyGPU_{commit.group(1)}.zip"
    if kept.is_file():
        return kept
    url = f"https://github.com/tinygrad/tinygpu_releases/raw/{commit.group(1)}/TinyGPU.zip"
    job.step = "Downloading TinyGPU.app"
    job.say(f"downloading {url}")
    kept.parent.mkdir(parents=True, exist_ok=True)
    part = kept.with_suffix(".part")
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "oaiy-egpu-plugin"}), timeout=60) as r, open(part, "wb") as f:
        shutil.copyfileobj(r, f)
    part.rename(kept)
    job.say(f"downloaded {kept.stat().st_size / 2**20:.1f} MB")
    return kept


def install_driver(job: Job):
    """TinyGPU.app into /Applications where it is not there (its signature checked), then its `install`, which asks
    macOS for its driver extension: macOS then asks the person to allow it, in System Settings, which is opened."""
    if driver()["enabled"]:
        job.step = "The driver is allowed already"
        return
    if not TINYGPU_PROGRAM.is_file():
        zipped = tinygpu_zip(job)
        job.step = "Putting TinyGPU.app in Applications"
        r = quiet(["/usr/bin/ditto", "-xk", str(zipped), "/Applications"], 120)
        if r.returncode != 0:
            raise RuntimeError(f"TinyGPU.app could not be put in Applications: {(r.stderr or r.stdout).strip()}")
    signed = quiet(["/usr/bin/codesign", "--verify", "--deep", "--strict", str(TINYGPU_APP)], 60)
    if signed.returncode != 0:
        raise RuntimeError(f"TinyGPU.app's signature does not check out: {signed.stderr.strip()}")
    team = re.search(r"TeamIdentifier=(\S+)", quiet(["/usr/bin/codesign", "-dv", str(TINYGPU_APP)], 30).stderr)
    job.say(f"TinyGPU.app is signed{f' (team {team.group(1)})' if team else ''}")
    job.step = "Asking macOS for the driver"
    asking = subprocess.Popen([str(TINYGPU_PROGRAM), "install"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    opened = False
    began = time.monotonic()
    # (its install may wait for the person's answer: ten minutes for it, and done as soon as the driver is allowed)
    while time.monotonic() < began + 600:
        if asking.poll() is not None:
            job.say(asking.stdout.read())
            break
        if not opened and time.monotonic() > began + 3:
            # (where the person allows it: macOS's own notice says the same)
            quiet(["/usr/bin/open", EXTENSIONS_SETTINGS], 10)
            opened = True
            job.step = "Allow TinyGPU in System Settings"
        if driver()["enabled"]:
            break
        time.sleep(1)
    if asking.poll() is None:
        asking.terminate()
    if asking.returncode not in (0, None, -15):
        raise RuntimeError(f"TinyGPU.app's install ended with {asking.returncode}: see its words above")
    if not opened:
        quiet(["/usr/bin/open", EXTENSIONS_SETTINGS], 10)
    d = driver()
    job.step = "The driver is allowed" if d["enabled"] else "Allow TinyGPU in System Settings: General, Login Items & Extensions, Driver Extensions"


def install_compiler(job: Job):
    """NVIDIA's compiler in Docker, as tinygrad's extra/setup_nvcc_osx.sh sets it up: Docker Desktop started where it
    is not running, tinygrad's image built (about 2 GB from Ubuntu's and NVIDIA's servers), and its shim put in
    ~/.local/bin as nvcc and nvdisasm (one of the person's own there left as it is); then nvcc asked its version."""
    cli = docker_cli()
    if not cli:
        raise RuntimeError("Docker Desktop is not installed: get it from docker.com, open it once, then set the compiler up")
    env = docker_env()
    if not docker()["running"]:
        job.step = "Starting Docker Desktop"
        quiet(["/usr/bin/open", "-a", "Docker"], 20)
        deadline = time.monotonic() + 240
        while not docker()["running"]:
            if time.monotonic() > deadline:
                raise RuntimeError("Docker Desktop did not start in 4 minutes: start it, then set the compiler up again")
            time.sleep(3)
    job.step = "Building NVIDIA's compiler image (about 2 GB to download)"
    build = subprocess.Popen([cli, "build", "--progress=plain", "--platform=linux/arm64", "-t", IMAGE, "-"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    build.stdin.write(DOCKERFILE)
    build.stdin.close()
    for line in build.stdout:
        job.say(line)
    if build.wait() != 0:
        raise RuntimeError("Docker could not build the compiler's image: its words are above")
    job.step = "Putting nvcc in ~/.local/bin"
    SHIMS.mkdir(parents=True, exist_ok=True)
    (SHIMS / "nvccshim").write_text(SHIM)
    (SHIMS / "nvccshim").chmod(0o755)
    for name in ("nvcc", "nvdisasm"):
        at = SHIMS / name
        if at.is_symlink() or not at.exists():
            at.unlink(missing_ok=True)
            at.symlink_to("nvccshim")
        else:
            job.say(f"~/.local/bin/{name} is a program of yours: left as it is")
    job.step = "Checking nvcc"
    asked = quiet([str(SHIMS / "nvcc"), "--version"], 300, env)
    said = (asked.stdout + asked.stderr).strip()
    job.say(said)
    if "Cuda compilation tools" not in said:
        raise RuntimeError("nvcc did not answer from Docker: its words are above")
    CHECKED.parent.mkdir(parents=True, exist_ok=True)
    CHECKED.write_text(json.dumps({"image": IMAGE, "seen": time.time()}))
    job.step = "NVIDIA's compiler is ready"


def setup_status() -> dict:
    r = runtime()
    c = card()
    pack = r["pack"] or {}
    if c.get("found"):
        c["covered"] = c.get("arch") in pack.get("archs", [])
    d = docker()
    comp = compiler(d)
    job = _job[0].view() if _job[0] is not None else None
    return {
        "driver": r["driver"],
        "card": c,
        "docker": d,
        "compiler": comp,
        "pack": {"kernels": pack.get("kernels", 0), "archs": pack.get("archs", [])},
        "tinygrad": r["tinygrad"],
        "missing": missing(),
        "job": job,
        "version": VERSION,
        # (the wizard's steps: each done when this says so)
        "steps": {"driver": r["driver"]["enabled"], "card": bool(c.get("found") and c.get("up")), "compiler": comp["ready"]},
    }


def reply(message_id, result=None, error=None):
    out = {"jsonrpc": "2.0", "id": message_id}
    if error is not None:
        out["error"] = {"code": -32601, "message": error}
    else:
        out["result"] = result
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()


def opened(target: list) -> dict:
    r = quiet(["/usr/bin/open", *target], 20)
    return {"opened": r.returncode == 0, "error": (r.stderr or "").strip() or None}


COMMANDS = {
    "card.runtime": runtime,
    "card.status": status,
    "setup.status": setup_status,
    "driver.install": lambda: start("driver", install_driver),
    "driver.openSettings": lambda: opened([EXTENSIONS_SETTINGS]),
    "docker.download": lambda: opened([DOCKER_PAGE]),
    "docker.start": lambda: opened(["-a", "Docker"]),
    "compiler.install": lambda: start("compiler", install_compiler),
}


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
            if command in COMMANDS:
                try:
                    reply(message_id, {"ok": True, "data": COMMANDS[command]()})
                except Exception as e:
                    reply(message_id, {"ok": False, "error": {"code": "failed", "message": f"{command}: {e}"}})
            else:
                reply(message_id, {"ok": False, "error": {"code": "invalid_request", "message": f"no command {command!r}"}})
        else:
            reply(message_id, error=f"no method {method!r}")


if __name__ == "__main__":
    main()
