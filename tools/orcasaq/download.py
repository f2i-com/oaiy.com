"""Download/verify the pinned original OrcaSAQ2 release using only Python stdlib."""
import argparse
import hashlib
import json
import pathlib
import urllib.request


def verify(path, record):
    if not path.is_file() or path.stat().st_size != record["size"]:
        return False
    digest = hashlib.sha256() if record["sha256"] else hashlib.sha1()
    if not record["sha256"]:
        digest.update(f'blob {record["size"]}\0'.encode())
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest() == (record["sha256"] or record["git_blob"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", type=pathlib.Path,
                        default=pathlib.Path(__file__).resolve().parents[2] / "models/OrcaSAQ-2-27B")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    manifest = json.loads(pathlib.Path(__file__).with_name("manifest.json").read_text(encoding="utf-8-sig"))
    root = args.directory.resolve()
    root.mkdir(parents=True, exist_ok=True)
    for record in manifest["files"]:
        path = (root / record["name"]).resolve()
        if path.parent != root:
            raise ValueError("Manifest filename must stay inside the model directory")
        if verify(path, record):
            print(f'Verified {path.name}', flush=True)
            continue
        if args.verify_only:
            raise RuntimeError(f'Missing or invalid file: {path}')
        temporary = path.with_suffix(path.suffix + ".partial")
        offset = temporary.stat().st_size if temporary.exists() else 0
        url = f'https://huggingface.co/{manifest["repo"]}/resolve/{manifest["revision"]}/{record["name"]}'
        request = urllib.request.Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
        print(f'Downloading {path.name} ({record["size"]:,} bytes)', flush=True)
        with urllib.request.urlopen(request, timeout=120) as response:
            resume = offset > 0 and response.status == 206
            if resume and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
                raise RuntimeError("Server returned an incorrect resume offset")
            with temporary.open("ab" if resume else "wb") as stream:
                for block in iter(lambda: response.read(8 << 20), b""):
                    stream.write(block)
        if not verify(temporary, record):
            raise RuntimeError(f'Hash mismatch: {temporary}; model file was not replaced')
        temporary.replace(path)
        print(f'Verified {path.name}', flush=True)
    print(f'Ready: {root} (revision {manifest["revision"]})')


if __name__ == "__main__":
    main()
