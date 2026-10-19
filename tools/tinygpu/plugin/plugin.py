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
                    setup.status        -> {driver, card, docker, compiler, pack, missing, job, steps, checked}
                                           ({refresh: true}: the card and Docker looked at again now, not as kept)
                    driver.install      -> TinyGPU.app put in /Applications and its driver asked of macOS (a job)
                    driver.openSettings -> System Settings at Login Items & Extensions, where the driver is allowed
                    docker.download     -> Docker Desktop's download page, in the browser
                    docker.start        -> Docker Desktop started
                    compiler.install    -> NVIDIA's compiler in Docker: tinygrad's image built, its shim put in (a job)
                    kernels.make        -> the card's kernels made from the pack's sources, where the pack has none for
                                           its architecture (the compiler set up first where it is not: a job)

A card whose architecture the pack has no kernels for (it carries the RTX 30, 40 and 50 series': sm_86, sm_89, sm_120)
gets them made from the pack's sources (card/kernels/sources.zip) into the card's server's cache, by the compiler: by
itself, once the compiler is set up and the card is there (a watcher looks every minute), or from the setup.

What the card and Docker were found to be is kept (`Kept`), and kept on disk for the plugin's next start: the page is
answered at once with the last look, while a newer one runs on a thread of its own. Docker's engine is asked on its own
socket with a short wait: Docker Desktop open with its engine not answering held each `docker info` for its whole
8 s, every look.
"""
from __future__ import annotations

import collections, hashlib, json, os, pathlib, re, shutil, socket, subprocess, sys, tempfile, threading, time, urllib.request, zipfile
from concurrent.futures import ThreadPoolExecutor

CARD = pathlib.Path(__file__).resolve().parent
ROOT = CARD.parent
VERSION = "0.2.1"
HOME = pathlib.Path.home()
# (where OAIY keeps this plugin's own files: what it downloads, never its folder, which is trusted by its digest)
DATA = pathlib.Path(os.environ.get("OAIY_PLUGIN_DATA_DIR") or HOME / "Library" / "Application Support" / "com.oaiy.app" / "plugin-data" / "egpu")
TINYGPU_APP = pathlib.Path("/Applications/TinyGPU.app")
TINYGPU_PROGRAM = TINYGPU_APP / "Contents" / "MacOS" / "TinyGPU"
EXTENSIONS_SETTINGS = "x-apple.systempreferences:com.apple.LoginItems-Settings.extension"
DOCKER_PAGE = "https://www.docker.com/products/docker-desktop/"
DOCKER_BINS = ("/Applications/Docker.app/Contents/Resources/bin", str(HOME / ".docker" / "bin"), "/usr/local/bin", "/opt/homebrew/bin")
# the card's server's kernel cache, and the kernels a pack lacked kept there (tools/tinygpu/webgpu_server.py)
KERNEL_CACHE = pathlib.Path(os.environ.get("TINYGPU_CACHE") or HOME / ".cache" / "tinygpu-webgpu")
MISSING = KERNEL_CACHE / "missing"
WORK = HOME / ".cache" / "oaiy-egpu-kernels"   # (where kernels are compiled: the Docker shim's container sees it)
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
# tinygrad's shim, with Docker's own folders on its PATH (OAIY starts the card's server with a short one), its container
# started by one call at a time (calls in parallel each started one, and all but one failed on its name), and living
# half an hour from its start (tinygrad's 5 minutes ended a pack's compiles halfway, every call at once)
SHIM = """#!/bin/sh
set -eu
PATH="$PATH:/Applications/Docker.app/Contents/Resources/bin:$HOME/.docker/bin:/usr/local/bin:/opt/homebrew/bin"
cname="cuda-nvcc-persistent"
running() { docker inspect --format='{{.State.Running}}' "$cname" 2>/dev/null | grep -q true; }
if ! running; then
  lock="${TMPDIR:-/tmp}/cuda-nvcc-shim.lock"
  i=0
  until mkdir "$lock" 2>/dev/null; do
    i=$((i + 1)); [ "$i" -gt 600 ] && break; sleep 0.1
  done
  if ! running; then
    docker rm -f "$cname" >/dev/null 2>&1 || true
    docker run -d --platform=linux/arm64 --name "$cname" \\
      -v /var/folders:/var/folders -v "$HOME":"$HOME" \\
      cuda-nvcc:12.8 sleep 1800 >/dev/null || true
  fi
  rmdir "$lock" 2>/dev/null || true
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


KEPT_FILE = DATA / "checks.json"


class Kept:
    """What a slow look found (the card's, system_profiler's; Docker's), kept. A look newer than `fresh` seconds is
    answered as it is; an older one is answered too, while a new look runs on a thread of its own, so the page never
    waits on one. With none yet, or asked to look `now`, the look is waited for. `seen` is when it was looked at (the
    page says how long ago); a look kept from the plugin's last run is answered once, and looked at again."""

    def __init__(self, name: str, fresh: float, look):
        self.name, self.fresh, self.look = name, fresh, look
        # (`at` on the monotonic clock, whose start is any: an old look's is -inf, not 0)
        self.value, self.at, self.seen = None, float("-inf"), None
        self.lock = threading.Lock()
        self.looking = None

    def busy(self) -> bool:
        return self.looking is not None and self.looking.is_alive()

    def stale(self):
        """Looked at again at its next asking (as that asking is answered): after what may have changed it."""
        with self.lock:
            self.at = float("-inf")

    def seed(self, value, seen: float):
        with self.lock:
            if self.value is None:
                self.value, self.at, self.seen = value, float("-inf"), seen

    def _look(self):
        value = self.look()
        with self.lock:
            self.value, self.at, self.seen = value, time.monotonic(), time.time()
        keep_looks()
        return value

    def _look_quietly(self):
        try:
            self._look()
        except Exception:  # (the kept look stays; the next asking looks again)
            pass

    def get(self, now: bool = False):
        with self.lock:
            value = self.value
            if value is not None and not now:
                if time.monotonic() - self.at >= self.fresh and not self.busy():
                    self.looking = threading.Thread(target=self._look_quietly, name=f"egpu-look-{self.name}", daemon=True)
                    self.looking.start()
                return value
        return self._look()


def keep_looks():
    """The card's and Docker's last looks, for the plugin's next start (written whole, then put in place)."""
    try:
        looks = {k.name: {"value": k.value, "seen": k.seen} for k in (_card, _docker) if k.value is not None}
        KEPT_FILE.parent.mkdir(parents=True, exist_ok=True)
        part = KEPT_FILE.with_name(f"{KEPT_FILE.name}.{os.getpid()}.{threading.get_ident()}.part")
        part.write_text(json.dumps(looks))
        part.replace(KEPT_FILE)
    except (OSError, TypeError, ValueError):
        pass


def kept_looks():
    """The looks kept by the plugin's last run, each answered once (as old) and looked at again."""
    try:
        looks = json.loads(KEPT_FILE.read_text())
    except (OSError, ValueError):
        return
    for k in (_card, _docker):
        look = looks.get(k.name) if isinstance(looks, dict) else None
        if isinstance(look, dict) and isinstance(look.get("value"), dict) and isinstance(look.get("seen"), (int, float)):
            k.seed(look["value"], float(look["seen"]))


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
    if why is None and (c := card()).get("found"):
        k = kernels(c, pack)
        job = _job[0]
        if job is not None and job.name == "kernels" and job.state == "running":
            why = f"making this card's kernels ({k['arch']}): {job.step}"
        elif not k["ready"]:
            why = (f"this card's kernels ({k['arch'] or 'its architecture unknown'}) are not in the plugin: the eGPU's setup makes "
                   f"them, with NVIDIA's compiler in Docker")
    have = f"tinygrad {(r['tinygrad'] or '?')[:9]}, {per_card(pack):,} kernels for {', '.join(archs_of(pack)) or 'no card'}"
    return {**r, "ready": why is None, "why": why, "state": "Ready" if why is None else "Not ready", "detail": why or have}


def archs_of(pack: dict) -> list:
    """A pack's architectures, in their numbers' order (sm_86, sm_89, sm_120)."""
    return sorted(pack.get("archs", []), key=lambda a: int(re.sub(r"\D", "", a) or 0))


def per_card(pack: dict) -> int:
    """A pack's kernels for one card (its cubins are every source's for each of its architectures)."""
    return pack.get("sources") or pack.get("kernels", 0)


def card(now: bool = False) -> dict:
    """The NVIDIA card macOS sees on the PCI bus, its link, and its architecture by its device id: as kept (`Kept`,
    10 s fresh: system_profiler can take a second or two), or looked at `now`."""
    return _card.get(now)


def look_card() -> dict:
    """system_profiler's NVIDIA card, its link, and its architecture by its device id."""
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


def docker_socket() -> str | None:
    """The Docker engine's socket, as Docker's own program finds it: DOCKER_HOST, else the context in use (Docker
    Desktop's desktop-linux: ~/.docker/run/docker.sock), else Docker's own place. None for an engine elsewhere (TCP,
    SSH), which only Docker's program asks."""
    host = os.environ.get("DOCKER_HOST")
    if not host:
        name = os.environ.get("DOCKER_CONTEXT")
        if not name:
            try:
                name = json.loads((HOME / ".docker" / "config.json").read_text()).get("currentContext")
            except (OSError, ValueError, AttributeError):
                name = None
        if name and name != "default":
            meta = HOME / ".docker" / "contexts" / "meta" / hashlib.sha256(name.encode()).hexdigest() / "meta.json"
            try:
                host = json.loads(meta.read_text())["Endpoints"]["docker"]["Host"]
            except (OSError, ValueError, KeyError, TypeError):
                host = None
    if host:
        return host[len("unix://"):] if host.startswith("unix://") else None
    desktop = HOME / ".docker" / "run" / "docker.sock"
    return str(desktop) if desktop.exists() else "/var/run/docker.sock"


def engine(sock: str, path: str) -> int | str | None:
    """The HTTP status Docker's engine answers `GET path` with, on its socket: "silent" where it took the call and said
    nothing within a second and a half (Docker Desktop open, its engine not answering), None where nothing listens."""
    s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        s.settimeout(0.3)
        s.connect(sock)
    except OSError:
        s.close()
        return None
    try:
        s.settimeout(1.5)
        s.sendall(f"GET {path} HTTP/1.0\r\nHost: docker\r\n\r\n".encode())
        said = b""
        while b"\r\n" not in said and len(said) < 4096:
            chunk = s.recv(4096)
            if not chunk:
                break
            said += chunk
    except socket.timeout:
        return "silent"
    except OSError:
        return None
    finally:
        s.close()
    first = said.split(b"\r\n", 1)[0].split()
    return int(first[1]) if len(first) > 1 and first[1].isdigit() else None


def docker(now: bool = False) -> dict:
    """Docker's engine, and NVIDIA's compiler's image in it: as kept (`Kept`, 15 s fresh), or looked at `now`."""
    return _docker.get(now)


def look_docker() -> dict:
    cli = docker_cli()
    running = image = stalled = False
    sock = docker_socket()
    if sock is not None:
        ping = engine(sock, "/_ping")
        running, stalled = ping == 200, ping == "silent"
        image = running and engine(sock, f"/images/{IMAGE}/json") == 200
    elif cli:
        # (an engine elsewhere: Docker's own program asks it)
        running = bool(quiet([cli, "info", "--format", "{{.ServerVersion}}"], 8, docker_env()).stdout.strip())
        image = running and quiet([cli, "image", "inspect", "--format", "{{.Id}}", IMAGE], 8, docker_env()).returncode == 0
    return {"app": os.path.isdir("/Applications/Docker.app"), "cli": cli, "running": running, "image": image, "stalled": stalled}


_card = Kept("card", 10, look_card)
_docker = Kept("docker", 15, look_docker)


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


def recorded_arch() -> str | None:
    """The architecture tinygrad named when the card's server last opened a card (its CACHE/card.json)."""
    try:
        return json.loads((KERNEL_CACHE / "card.json").read_text()).get("arch") or None
    except (OSError, ValueError):
        return None


def kernel_name(arch: str, src: str) -> str:
    """As the card's server names a kernel's cubin (webgpu_server.kernel_name): its architecture and source's hash."""
    return hashlib.sha256((arch + "\0" + src).encode()).hexdigest()[:24]


def made_record(arch: str) -> pathlib.Path:
    return DATA / f"kernels-{arch}.json"


def kernels(c: dict, pack: dict) -> dict:
    """The card's kernels: in the pack (its architecture one of the pack's), or made for it from the pack's sources
    (`made_record`, for this pack's sources), or not yet."""
    arch = c.get("arch") or recorded_arch()
    in_pack = (arch in pack.get("archs", [])) if arch else None
    zipped = CARD / "kernels" / "sources.zip"
    made = None
    if arch and not in_pack:
        try:
            made = json.loads(made_record(arch).read_text())
        except (OSError, ValueError):
            made = None
        if made and made.get("pack") != pack.get("built"):
            made = None   # (made for another pack's sources)
    ready = bool(in_pack) or bool(made and made.get("made", 0) >= made.get("sources", 1))
    return {"arch": arch, "inPack": in_pack, "made": made, "ready": ready, "sources": per_card(pack), "canMake": zipped.is_file()}


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
            # (what it did is looked at again as the page next asks)
            _card.stale()
            _docker.stale()

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
    docker_running(job)
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


def docker_running(job: Job):
    """Docker Desktop started where it is not running, and waited for (four minutes at most)."""
    if docker(now=True)["running"]:
        return
    job.step = "Starting Docker Desktop"
    quiet(["/usr/bin/open", "-a", "Docker"], 20)
    deadline = time.monotonic() + 240
    while not docker(now=True)["running"]:
        if time.monotonic() > deadline:
            raise RuntimeError("Docker Desktop did not start in 4 minutes: start it, then try again")
        time.sleep(3)


def find_nvcc() -> tuple[str, dict | None] | None:
    """nvcc as the card's server finds it (webgpu_server.nvcc): the PATH's, else the Docker shim in ~/.local/bin."""
    if found := shutil.which("nvcc"):
        return found, None
    if os.access(SHIMS / "nvcc", os.X_OK):
        return str(SHIMS / "nvcc"), docker_env()
    return None


def make_kernels(job: Job):
    """The card's kernels made from the pack's sources (card/kernels/sources.zip) for its architecture, into the card's
    server's cache (each cubin named as the server names it, its source beside it), where the pack has none for it:
    NVIDIA's compiler set up first where it is not, Docker started where the compiler is its shim. Six at a time (each a
    second or so in Docker); what the cache has already is kept. Recorded for this pack (`made_record`) when all are."""
    c = card(now=True)
    pack = runtime()["pack"] or {}
    k = kernels(c, pack)
    arch = k["arch"]
    if not arch:
        raise RuntimeError("the card's architecture is not known: connect the card, or load a model on the eGPU once")
    if k["inPack"]:
        job.step = f"The plugin has this card's kernels already ({arch})"
        return
    if not k["canMake"]:
        raise RuntimeError("this plugin carries no kernel sources to make a card's kernels from: a newer plugin does")
    if find_nvcc() is None:
        install_compiler(job)
    program, env = find_nvcc()
    if env is not None:
        docker_running(job)
    with zipfile.ZipFile(CARD / "kernels" / "sources.zip") as z:
        sources = [z.read(n).decode() for n in z.namelist() if n.endswith(".cu")]
    KERNEL_CACHE.mkdir(parents=True, exist_ok=True)
    WORK.mkdir(parents=True, exist_ok=True)
    todo = [src for src in sources if not (KERNEL_CACHE / f"{kernel_name(arch, src)}.cubin").exists()]
    job.say(f"{len(sources):,} kernels for {arch}: {len(sources) - len(todo):,} made already, {len(todo):,} to make")
    done, failed = [len(sources) - len(todo)], []

    def make(src: str):
        name = kernel_name(arch, src)
        # (in the home folder's .cache: the shim's container sees the home folder, and no other but /var/folders)
        with tempfile.TemporaryDirectory(dir=WORK) as tmp:
            cu, cubin = pathlib.Path(tmp) / f"{name}.cu", pathlib.Path(tmp) / f"{name}.cubin"
            cu.write_text(src)
            argv = [program, f"-arch={arch}", "-cubin", "-o", str(cubin), str(cu)]
            run = subprocess.run(argv, capture_output=True, text=True, env=env)
            if run.returncode != 0:
                # (once more: the shim's container may have ended as it was asked, and is started again)
                run = subprocess.run(argv, capture_output=True, text=True, env=env)
            if run.returncode != 0:
                raise RuntimeError(f"{name}: {(run.stderr or run.stdout).strip()[-600:]}")
            shutil.copyfile(cu, KERNEL_CACHE / f"{name}.cu")
            shutil.copyfile(cubin, KERNEL_CACHE / f"{name}.{os.getpid()}.cubin")
            (KERNEL_CACHE / f"{name}.{os.getpid()}.cubin").rename(KERNEL_CACHE / f"{name}.cubin")

    with ThreadPoolExecutor(max_workers=6) as pool:
        for src, result in zip(todo, pool.map(lambda s: _tried(make, s), todo)):
            if result is not None:
                failed.append(result)
                job.say(f"failed: {result}")
            else:
                done[0] += 1
            job.step = f"Making this card's kernels ({arch}): {done[0]:,} of {len(sources):,}"
    if failed:
        raise RuntimeError(f"{len(failed)} of {len(sources)} kernels could not be made for {arch}: the first's error is above")
    DATA.mkdir(parents=True, exist_ok=True)
    made_record(arch).write_text(json.dumps({"arch": arch, "sources": len(sources), "made": done[0], "pack": pack.get("built"), "at": time.time()}))
    job.step = f"This card's {len(sources):,} kernels are made ({arch})"


def _tried(fn, *args) -> str | None:
    try:
        fn(*args)
        return None
    except Exception as e:  # (said with its kernel; the others go on)
        return str(e)


def watch_once(tried: set) -> bool:
    """One look of `watch`'s: whether it started the card's kernels' job."""
    c = card()
    if not (c.get("found") and c.get("up")) or (_job[0] is not None and _job[0].state == "running"):
        return False
    pack = runtime()["pack"] or {}
    k = kernels(c, pack)
    key = (k["arch"], pack.get("built"))
    if not (k["arch"] and k["inPack"] is False and not k["ready"] and k["canMake"]) or key in tried:
        return False
    # (Docker asked only for a card that needs the compiler: one the pack carries kernels for never does)
    if find_nvcc() is None or not (shutil.which("nvcc") or compiler(docker())["built"]):
        return False
    tried.add(key)
    return start("kernels", make_kernels)["started"]


def watch():
    """Every minute: a card the pack has no kernels for, with the compiler set up, gets its kernels made by itself (a
    job, as `kernels.make`'s), once a pack; a failed attempt is not tried again until the plugin starts again."""
    tried: set = set()
    time.sleep(20)
    while True:
        try:
            watch_once(tried)
        except Exception:
            pass
        time.sleep(60)


_looks = ThreadPoolExecutor(max_workers=2, thread_name_prefix="egpu-status")


def setup_status(payload: dict | None = None) -> dict:
    """The setup's parts as kept (the card and Docker looked at again on threads of their own where their looks are
    old), or, with `refresh`, all looked at now. `checked` says when the oldest look was, and whether newer ones run."""
    refresh = bool(isinstance(payload, dict) and payload.get("refresh"))
    # (the two looks side by side: when both wait, the page waits for the slower, not the two)
    looked_card, looked_docker = _looks.submit(card, refresh), _looks.submit(docker, refresh)
    r = runtime()
    c, d = dict(looked_card.result()), looked_docker.result()
    pack = r["pack"] or {}
    k = kernels(c, pack) if c.get("found") else {"arch": recorded_arch(), "inPack": None, "made": None, "ready": False, "sources": per_card(pack), "canMake": False}
    if c.get("found"):
        c["covered"] = k["ready"]
    comp = compiler(d)
    job = _job[0].view() if _job[0] is not None else None
    seen = [k.seen for k in (_card, _docker) if k.seen is not None]
    return {
        "driver": r["driver"],
        "card": c,
        "docker": d,
        "compiler": comp,
        "pack": {"kernels": per_card(pack), "archs": archs_of(pack)},
        "kernels": k,
        "tinygrad": r["tinygrad"],
        "missing": missing(),
        "job": job,
        "version": VERSION,
        # (the wizard's steps: each done when this says so)
        "steps": {"driver": r["driver"]["enabled"], "card": bool(c.get("found") and c.get("up")), "kernels": k["ready"], "compiler": comp["ready"]},
        # (the oldest look's time, in ms since 1970, and whether newer looks run now)
        "checked": {"at": round(min(seen) * 1000) if seen else None, "looking": _card.busy() or _docker.busy()},
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


def docker_started() -> dict:
    said = opened(["-a", "Docker"])
    _docker.stale()
    return said


# (each given the command's payload, an object or None)
COMMANDS = {
    "card.runtime": lambda p: runtime(),
    "card.status": lambda p: status(),
    "setup.status": setup_status,
    "driver.install": lambda p: start("driver", install_driver),
    "driver.openSettings": lambda p: opened([EXTENSIONS_SETTINGS]),
    "docker.download": lambda p: opened([DOCKER_PAGE]),
    "docker.start": lambda p: docker_started(),
    "compiler.install": lambda p: start("compiler", install_compiler),
    "kernels.make": lambda p: start("kernels", make_kernels),
}


def main():
    kept_looks()
    threading.Thread(target=watch, name="egpu-watch", daemon=True).start()
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
                payload = params.get("payload")
                try:
                    reply(message_id, {"ok": True, "data": COMMANDS[command](payload if isinstance(payload, dict) else None)})
                except Exception as e:
                    reply(message_id, {"ok": False, "error": {"code": "failed", "message": f"{command}: {e}"}})
            else:
                reply(message_id, {"ok": False, "error": {"code": "invalid_request", "message": f"no command {command!r}"}})
        else:
            reply(message_id, error=f"no method {method!r}")


if __name__ == "__main__":
    main()
