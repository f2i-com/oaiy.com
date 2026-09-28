"""Send ./clips/*.wav (see make_clips.py) to one or more voice servers the
way OAIY Desktop does (JSON with a base64 WAV, a fresh connection each time),
then one multipart request to the first; print each answer and its time.

Usage: python compare_servers.py http://127.0.0.1:18791 http://127.0.0.1:8781
"""
import base64, json, sys, time, urllib.request, glob, os, uuid
def post_json(base, wav):
    body = json.dumps({"audio": base64.b64encode(wav).decode(), "response_format": "json"}).encode()
    req = urllib.request.Request(base + "/v1/audio/transcriptions", data=body, headers={"Content-Type": "application/json", "Connection": "close"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())
    return out.get("text", ""), time.perf_counter() - t
def post_multipart(base, wav):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"model\"\r\n\r\nwhisper-1\r\n--{b}\r\nContent-Disposition: form-data; name=\"response_format\"\r\n\r\ntext\r\n--{b}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"a.wav\"\r\nContent-Type: audio/wav\r\n\r\n").encode() + wav + f"\r\n--{b}--\r\n".encode()
    req = urllib.request.Request(base + "/v1/audio/transcriptions", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read().decode()
bases = sys.argv[1:]
ref = json.load(open("clips/reference.json"))
for base in bases:
    with urllib.request.urlopen(base + "/v1/models", timeout=10) as r:
        print(base, "models:", r.read().decode())
rows = []
for clip in sorted(glob.glob("clips/*.wav")):
    wav = open(clip, "rb").read()
    name = os.path.basename(clip)
    row = [name]
    for base in bases:
        text, dt = post_json(base, wav)
        row.append((text, dt))
    rows.append(row)
    print(name, "|", ref.get(name, "")[:60])
    for base, (text, dt) in zip(bases, row[1:]):
        print(f"   {base[-5:]} {dt*1000:7.1f} ms  {text}")
print("multipart:", post_multipart(bases[0], open("clips/tts0.wav", "rb").read()))
