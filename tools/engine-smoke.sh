#!/bin/sh
# tools/engine-smoke.sh SERVER MODEL.gguf [PORT]: one question to the language-model server, on this computer's GPU.
#
#     sh tools/engine-smoke.sh target/release/oaiy-llm-server-webgpu ~/models/Qwen3.5-4B-Q4_K_M.gguf
#
# It starts SERVER (oaiy-llm-server, or the portable oaiy-llm-server-webgpu an installer carries) on MODEL, waits
# for it to load, asks it for a few words, and says what came back, where the weights were put (the GPU, or the
# CPU for what had no room) and how long it took. The server's whole text is kept in a file it names.
#
# This is the check that the kernels a real model uses are ones this system's GPU compiler takes: each is compiled
# the first time it runs, so a kernel the compiler refuses ends the server here, with the compiler's own words, and
# not on someone's first message. It needs curl and nothing else. It changes nothing: the server it starts is
# stopped when it ends.
set -u
[ $# -ge 2 ] || { echo "usage: sh tools/engine-smoke.sh SERVER MODEL.gguf [PORT]"; exit 2; }
server=$1
model=$2
port=${3:-18431}
[ -x "$server" ] || { echo "$server is not a program that can be run (build it: docs/MAC.md, or platform/desktop/README.md)"; exit 2; }
[ -e "$model" ] || { echo "$model is not there"; exit 2; }
command -v curl > /dev/null 2>&1 || { echo "curl is missing"; exit 2; }

log="${TMPDIR:-/tmp}/oaiy-engine-smoke.txt"
log=$(printf '%s' "$log" | sed 's|//|/|g')
started=$(date +%s)
"$server" --model "$model" --name smoke --host 127.0.0.1 --port "$port" > "$log" 2>&1 &
pid=$!
trap 'kill "$pid" 2> /dev/null' EXIT

said() { echo; echo "== the server's own words (the end of $log)"; tail -"${1:-30}" "$log" | cut -c 1-600; }

# The model is loaded before the server answers: minutes for a large file on a slow drive.
waited=0
until curl -fsS -m 2 "http://127.0.0.1:$port/health" > /dev/null 2>&1; do
  if ! kill -0 "$pid" 2> /dev/null; then
    echo "The server ended before it answered (after $waited s)."
    said 40
    exit 1
  fi
  if [ "$waited" -ge 900 ]; then
    echo "The server did not answer in 15 minutes."
    said 20
    exit 1
  fi
  sleep 1
  waited=$((waited + 1))
done
echo "The server answers after $waited s."
grep -E "weights are all on the GPU|of its weights are on the GPU|WgpuBackend|adapter|on the CPU" "$log" | head -6 | cut -c 1-400
# ENGINE_SMOKE_NEEDS_GPU=1: a run that is there to try the GPU's compiler is not passed by the CPU answering instead.
if [ "${ENGINE_SMOKE_NEEDS_GPU:-0}" = 1 ] && grep -q "running on the CPU" "$log"; then
  echo "The server found no GPU it can use and runs on the CPU: this run was to try the GPU."
  exit 1
fi

body='{"model":"smoke","messages":[{"role":"user","content":"Reply with the single word: ready"}],"max_tokens":24,"temperature":0,"stream":false}'
asked=$(date +%s)
reply=$(curl -sS -m 900 -H "Content-Type: application/json" -d "$body" "http://127.0.0.1:$port/v1/chat/completions" 2>&1)
took=$(( $(date +%s) - asked ))
if ! kill -0 "$pid" 2> /dev/null; then
  echo "The server ended while it answered:"
  said 40
  exit 1
fi
text=$(printf '%s' "$reply" | grep -o '"content":"[^"]*"' | head -1 | sed 's/^"content":"//; s/"$//')
if [ -z "$text" ]; then
  echo "No reply came back in $took s. What the server sent:"
  printf '%s\n' "$reply" | cut -c 1-600
  said 20
  exit 1
fi
echo "It replied in $took s: $text"

# ENGINE_SMOKE_LONG=1: a prompt of a few thousand tokens as well, which is what an agent sends (its instructions and
# tools) and takes other kernels than a short one: a prompt is read in blocks of rows, a reply a token at a time. A
# word is named at the start, a page of plain sentences follows, and the model is asked for the word.
if [ "${ENGINE_SMOKE_LONG:-0}" = 1 ]; then
  filler=""
  i=0
  while [ "$i" -lt "${ENGINE_SMOKE_LONG_LINES:-260}" ]; do
    filler="$filler Line $i of the notes says that the river was calm and the boats stayed in the harbour that day."
    i=$((i + 1))
  done
  body="{\"model\":\"smoke\",\"messages\":[{\"role\":\"user\",\"content\":\"The secret word is marigold. Remember it.$filler Now answer with one word only: what is the secret word?\"}],\"max_tokens\":48,\"temperature\":0,\"stream\":false}"
  asked=$(date +%s)
  reply=$(printf '%s' "$body" | curl -sS -m 3000 -H "Content-Type: application/json" --data-binary @- "http://127.0.0.1:$port/v1/chat/completions" 2>&1)
  took=$(( $(date +%s) - asked ))
  if ! kill -0 "$pid" 2> /dev/null; then
    echo "The server ended while it read the long prompt:"
    said 40
    exit 1
  fi
  text=$(printf '%s' "$reply" | grep -o '"content":"[^"]*"' | head -1 | sed 's/^"content":"//; s/"$//')
  tokens=$(printf '%s' "$reply" | grep -o '"prompt_tokens":[0-9]*' | head -1 | sed 's/.*://')
  echo "The long prompt (${tokens:-?} tokens) was answered in $took s: ${text:-<nothing>}"
  case "$text" in
    *[Mm]arigold*) ;;
    *)
      echo "FAIL: the answer does not have the word the prompt named. What the server sent:"
      printf '%s\n' "$reply" | cut -c 1-600
      said 20
      exit 1 ;;
  esac
fi
echo "PASS: the model loaded and answered ($(( $(date +%s) - started )) s in all). The server's whole text is in $log"
