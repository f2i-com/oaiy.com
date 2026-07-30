#!/usr/bin/env python3
"""fetch_zip.py <zip-url> <dest-dir>

Download a .zip (e.g. a GitHub `archive/refs/heads/<branch>.zip`) and
extract it into <dest-dir>, using only the Python standard library so the
companion's one-click installers need no `git` on PATH.

On success the LAST line printed to stdout is:

    EXTRACTED=<absolute path to the single top-level extracted folder>

so the calling .bat can capture it (GitHub archives extract to a single
`<repo>-<branch>/` directory). Progress is printed line-by-line so it
streams into the companion's LogsViewer.

Idempotent-ish: if <dest-dir>/<repo>-<branch> already exists it is removed
first so a re-run gets a clean tree.
"""
import os
import shutil
import ssl
import sys
import tarfile
import tempfile
import urllib.request
import zipfile


def log(msg: str) -> None:
    print(msg, flush=True)


def download(url: str, out_path: str) -> None:
    log(f"[fetch_zip] downloading {url}")
    # Some corporate/AV setups have a thin cert store; fall back to an
    # unverified context only if the default handshake fails, and say so.
    try:
        ctx = ssl.create_default_context()
        _stream(url, out_path, ctx)
    except ssl.SSLError as e:
        log(f"[fetch_zip] TLS verify failed ({e}); retrying without verification")
        ctx = ssl._create_unverified_context()
        _stream(url, out_path, ctx)


def _stream(url: str, out_path: str, ctx: ssl.SSLContext) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "oaiy-companion"})
    with urllib.request.urlopen(req, context=ctx) as resp:
        total = int(resp.headers.get("Content-Length") or 0)
        got = 0
        last_mb = 0
        with open(out_path, "wb") as f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                mb = got // (8 * 1024 * 1024)
                if mb != last_mb:
                    last_mb = mb
                    if total:
                        log(f"[fetch_zip] {got/1e6:.0f} / {total/1e6:.0f} MB")
                    else:
                        log(f"[fetch_zip] {got/1e6:.0f} MB")
    log(f"[fetch_zip] downloaded {got/1e6:.1f} MB")


def main() -> int:
    if len(sys.argv) != 3:
        log("usage: fetch_zip.py <zip-url> <dest-dir>")
        return 2
    url, dest = sys.argv[1], sys.argv[2]
    os.makedirs(dest, exist_ok=True)

    is_tar = url.endswith(".tar.gz") or url.endswith(".tgz")
    tmp = os.path.join(tempfile.gettempdir(), "oaiy_fetch" + (".tar.gz" if is_tar else ".zip"))
    try:
        download(url, tmp)
    except Exception as e:  # noqa: BLE001 - surface any network error verbatim
        log(f"[fetch_zip] ERROR downloading: {e}")
        return 1

    log(f"[fetch_zip] extracting into {dest}")
    top = ""
    try:
        if is_tar:
            with tarfile.open(tmp, "r:gz") as t:
                names = t.getnames()
                top = names[0].split("/")[0] if names else ""
                target = os.path.join(dest, top)
                if top and os.path.isdir(target):
                    log(f"[fetch_zip] removing existing {target}")
                    shutil.rmtree(target, ignore_errors=True)
                # filter="data" (py3.12) blocks path-traversal/special members;
                # fall back for older interpreters that lack the kwarg.
                try:
                    t.extractall(dest, filter="data")
                except TypeError:
                    t.extractall(dest)
        else:
            with zipfile.ZipFile(tmp) as z:
                names = z.namelist()
                top = names[0].split("/")[0] if names else ""
                target = os.path.join(dest, top)
                if top and os.path.isdir(target):
                    log(f"[fetch_zip] removing existing {target}")
                    shutil.rmtree(target, ignore_errors=True)
                z.extractall(dest)
    except Exception as e:  # noqa: BLE001
        log(f"[fetch_zip] ERROR extracting: {e}")
        return 1
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    extracted = os.path.join(dest, top) if top else dest
    log(f"[fetch_zip] done: {extracted}")
    log(f"EXTRACTED={extracted}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
