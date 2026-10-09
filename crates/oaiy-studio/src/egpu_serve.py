"""OAIY's way into tinygrad's LLM server (docs/MAC.md).

    python egpu_serve.py [--watch-stdin] --model FILE --serve PORT --max_context N ...   tinygrad.llm's own arguments
    python egpu_serve.py --check                                                          what this Python has, as JSON

tinygrad's server listens on every network interface and asks for no key. Started through this file it listens on
this computer only (127.0.0.1) and answers only requests that carry OAIY_EGPU_KEY as their bearer token. With
--watch-stdin it stops when its standard input closes, so it never outlives the OAIY that started it while holding the
graphics card. Nothing of tinygrad's is changed on disk, and nothing here names tinygrad's own classes: Python's socket
server, which tinygrad's server is built on, is told where to listen and whom to answer.
"""
import hmac, importlib.util, json, os, runpy, socketserver, sys, threading

# Where tinygrad has kept its LLM server: a package since April 2026, a single module before.
SERVERS = ("tinygrad.llm", "tinygrad.apps.llm")
KEY = os.environ.pop("OAIY_EGPU_KEY", "")


def local_only():
    """Every server made from here on listens on this computer only, and answers only OAIY."""
    init = socketserver.TCPServer.__init__

    def guarded(handler):
        class Guarded(handler):
            def _oaiy(self):
                given = self.headers.get("Authorization", "")
                if not KEY or hmac.compare_digest(given.encode(), ("Bearer " + KEY).encode()):
                    return True
                self.send_error(401, "this server answers OAIY only")
                return False

            def do_GET(self):
                if self._oaiy():
                    super().do_GET()

            def do_POST(self):
                if self._oaiy():
                    super().do_POST()

        return Guarded

    def listen(self, server_address, RequestHandlerClass, *args, **kwargs):
        if hasattr(RequestHandlerClass, "do_POST"):
            RequestHandlerClass = guarded(RequestHandlerClass)
        init(self, ("127.0.0.1", server_address[1]), RequestHandlerClass, *args, **kwargs)

    socketserver.TCPServer.__init__ = listen


def lifeline():
    """Stop with whoever started this: their end of standard input closes when they go, however they go.

    Read from the descriptor itself, not through sys.stdin: a thread waiting inside Python's buffered reader holds
    its lock, and an interpreter that exits meanwhile (tinygrad failing to load a model) dies on that lock with a
    crash in place of its own error message.
    """
    try:
        while os.read(0, 4096):
            pass
    except OSError:
        pass
    os._exit(0)


def as_file(args):
    """Have tinygrad read --model as the file it is.

    tinygrad takes the value as a file only when it starts with "/" or "." and otherwise as a model's name or an
    address to download. A whole path does on a Mac; one that does not (a relative one, a Windows drive's) is
    rewritten so that it does.
    """
    for flag in ("--model", "-m"):
        if flag in args[:-1]:
            at = args.index(flag) + 1
            if os.path.isfile(args[at]) and not args[at].startswith(("/", ".")):
                whole = os.path.abspath(args[at])
                args[at] = "//?/" + whole.replace("\\", "/") if os.name == "nt" else whole


def server():
    """The name of tinygrad's LLM server in this Python, or why there is none."""
    try:
        import tinygrad  # noqa: F401
    except Exception as e:
        return None, f"tinygrad is not installed for this Python ({e})"
    for name in SERVERS:
        try:
            if importlib.util.find_spec(name) is not None:
                return name, None
        except Exception:
            pass
    return None, "this tinygrad has no LLM server (tinygrad.llm): it is older than 2025, update it"


def commit(folder):
    """The commit a checkout of tinygrad is at, when it is one."""
    try:
        git = os.path.join(os.path.dirname(folder), ".git")
        head = open(os.path.join(git, "HEAD")).read().strip()
        if head.startswith("ref: "):
            head = open(os.path.join(git, *head[5:].split("/"))).read().strip()
        return head[:12]
    except Exception:
        return None


def check():
    name, why = server()
    found = {"python": sys.executable, "version": sys.version.split()[0], "server": name, "error": why, "tinygrad": None, "commit": None}
    if "tinygrad" in sys.modules:
        folder = os.path.dirname(os.path.abspath(sys.modules["tinygrad"].__file__))
        found["tinygrad"], found["commit"] = folder, commit(folder)
    try:
        import jinja2  # noqa: F401
        found["jinja2"] = True
    except Exception:
        found["jinja2"] = False  # without it tinygrad uses a plain chat format, and no tool calls
    print(json.dumps(found))


def main():
    args = sys.argv[1:]
    if "--check" in args:
        return check()
    name, why = server()
    if name is None:
        sys.exit(f"oaiy-egpu: {why}")
    if "--watch-stdin" in args:
        args.remove("--watch-stdin")
        threading.Thread(target=lifeline, daemon=True).start()
    local_only()
    as_file(args)
    sys.argv = [name] + args
    runpy.run_module(name, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
