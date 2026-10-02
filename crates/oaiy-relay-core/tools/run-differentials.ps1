<#
.SYNOPSIS
  Runs the differential tests of oaiy-relay-core: the crate against the implementations that the rest of the system uses (libsodium in PHP, OpenSSL in Node and Python, PHP's
  json_decode and the relay's own readers, the Python ports of the shipped decoders), on generated corpora.

.DESCRIPTION
  The reviewer's drivers are tests in tests/rv_*.rs marked #[ignore]: each reads a corpus written by a generator (Python, PHP or Node) and writes the crate's verdicts for a
  comparison script. Run by themselves they fail loudly, because there is nothing to read. This script is what runs them: it copies the generators into a scratch directory
  (they write their files next to themselves, and the repository must stay clean), generates the corpora, runs the Rust driver with the right variables, and compares.

  What is asserted (the script exits 1 when any of it fails):
    ed25519        the crate accepts exactly what libsodium accepts, on the edge cases (RV_ED_SODIUM turns the driver into a check)
    x25519         the crate refuses what libsodium refuses and agrees on every shared secret (the driver asserts)
    pairing math   the arithmetic of the pairing equals an independent recomputation (the driver asserts)
    sealed         libsodium's boxes open here, the crate's boxes open in libsodium (the driver asserts, and seal_check.php exits 1 when one does not)
    poll           the poll decision core agrees with the reviewer's independent implementation of the README on every generated case (zero disagreements)
    admission      the readers agree with the Python port of the shipped decoders except for the known, documented differences (the count is pinned at 1005 for the default corpus; see -ExpectedAdmissionDisagreements)
    encoding       base64url, JSON and the text rules against PHP, Node, Python and serde_json: the comparison scripts print their counts; the lines that must be zero are checked

  A tool that is not installed makes its pipeline SKIPPED, loudly, and the exit status 3 (unless nothing else failed and -AllowSkips is given); it is never a silent pass.

.PARAMETER PollCases       Cases for the poll differential (default 20000).
.PARAMETER AdmissionCases  Damaged copies per recorded admission (default 2000; the ten recorded cases give 20,000).
.PARAMETER JsonCases       Size of the JSON corpus (default 200000).
.PARAMETER Only            Run only these pipelines: ed25519, x25519, math, sealed, poll, admission, b64, json, text.
.PARAMETER ExpectedUrlDisagreements  The pinned count of relay URLs the crate reads differently from the pattern of the schemas (16, all on purpose; see the comment where it is checked).
.PARAMETER Keep            Keep the scratch directory (its path is printed).
.PARAMETER AllowSkips      Exit 0 when pipelines were skipped for want of a tool (they are still printed as SKIPPED).
#>
param(
    [int]$PollCases = 20000,
    [int]$AdmissionCases = 2000,
    [int]$JsonCases = 200000,
    [int]$ExpectedAdmissionDisagreements = -1,
    [int]$ExpectedUrlDisagreements = 16,
    [string[]]$Only = @(),
    [switch]$Keep,
    [switch]$AllowSkips
)

$ErrorActionPreference = 'Stop'
$Only = @($Only | ForEach-Object { $_ -split ',' } | Where-Object { $_ })   # `-File` hands an array over as one comma-separated string
$known = @('ed25519', 'x25519', 'math', 'sealed', 'poll', 'admission', 'b64', 'json', 'text')
foreach ($o in $Only) { if ($known -notcontains $o) { Write-Host "FAILED: unknown pipeline '$o' (known: $($known -join ', '))"; exit 1 } }
$crate = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$root = (Resolve-Path (Join-Path $crate '..\..')).Path
$tests = Join-Path $crate 'tests'

# ---- the tools
function Find-Tool([string[]]$names, [string]$fromEnv) {
    if ($fromEnv -and (Test-Path -LiteralPath $fromEnv)) { return $fromEnv }
    foreach ($n in $names) {
        $c = Get-Command $n -ErrorAction SilentlyContinue | Where-Object { $_.CommandType -eq 'Application' } | Select-Object -First 1
        if ($c) { return $c.Source }
    }
    return $null
}
$php = Find-Tool @('php') $env:OAIY_PHP
if (-not $php) { $wamp = Get-ChildItem 'C:\wamp64\bin\php' -Directory -ErrorAction SilentlyContinue | Sort-Object Name -Descending | Select-Object -First 1; if ($wamp) { $php = Join-Path $wamp.FullName 'php.exe' } }
$py = Find-Tool @('python', 'python3') $null
$node = Find-Tool @('node') $null
$cargo = Find-Tool @('cargo') $null
$phpOk = $false
if ($php) {
    $mods = (& $php -m) -join ' '
    $phpOk = $mods -match 'sodium'
}
$pyCrypto = $false
if ($py) { & $py -c 'import cryptography' 2>$null; $pyCrypto = ($LASTEXITCODE -eq 0) }
$env:OAIY_PHP = $php

Write-Host "tools: php=$php (sodium: $phpOk)  python=$py (cryptography: $pyCrypto)  node=$node  cargo=$cargo"
if (-not $cargo) { Write-Host 'FAILED: cargo is not on the path'; exit 1 }

$work = Join-Path ([System.IO.Path]::GetTempPath()) ('oaiy-relay-core-differentials-' + [guid]::NewGuid().ToString('N').Substring(0, 8))
New-Item -ItemType Directory -Path $work | Out-Null
# The generators find the relay's PHP sources and the recorded fixtures by their place in the repository, so the scratch directory repeats that part of the layout.
$wt = Join-Path $work 'crates\oaiy-relay-core\tests'
New-Item -ItemType Directory -Path $wt | Out-Null
Copy-Item -Recurse -LiteralPath (Join-Path $tests 'rv') -Destination (Join-Path $wt 'rv')
Copy-Item -Recurse -LiteralPath (Join-Path $tests 'rv_enc') -Destination (Join-Path $wt 'rv_enc')
New-Item -ItemType Directory -Path (Join-Path $work 'platform\relay') -Force | Out-Null
Copy-Item -Recurse -LiteralPath (Join-Path $root 'platform\relay\src') -Destination (Join-Path $work 'platform\relay\src')
New-Item -ItemType Directory -Path (Join-Path $work 'platform\protocol\relay\v1\fixtures') -Force | Out-Null
Copy-Item -Recurse -LiteralPath (Join-Path $root 'platform\protocol\relay\v1\fixtures\aokie') -Destination (Join-Path $work 'platform\protocol\relay\v1\fixtures\aokie')
Write-Host "scratch: $work"

$results = [ordered]@{}
$failed = 0

function Fail([string]$why) { throw $why }

function Run([string]$exe, [string[]]$arguments, [string]$dir = $work) {
    $ErrorActionPreference = 'Continue'   # a native program's stderr is not an error of the script
    Push-Location $dir
    try {
        $out = & $exe @arguments 2>&1 | ForEach-Object { "$_" }
        $code = $LASTEXITCODE
    } finally { Pop-Location }
    $out | ForEach-Object { Write-Host "    $_" }
    if ($code -ne 0) { Fail "$(Split-Path $exe -Leaf) $($arguments -join ' ') exited with $code" }
    return $out
}

# Runs one #[ignore]d driver of the crate's tests with its environment.
function Driver([string]$testFile, [string[]]$names, [hashtable]$vars) {
    $saved = @{}
    foreach ($k in $vars.Keys) { $saved[$k] = [Environment]::GetEnvironmentVariable($k); [Environment]::SetEnvironmentVariable($k, [string]$vars[$k]) }
    try {
        $a = @('test', '-p', 'oaiy-relay-core', '--test', $testFile, '--', '--ignored', '--nocapture') + $names
        $null = Run $cargo $a $root
    } finally { foreach ($k in $vars.Keys) { [Environment]::SetEnvironmentVariable($k, $saved[$k]) } }
}

function Pipeline([string]$name, [string[]]$needs, [scriptblock]$body) {
    if ($Only.Count -gt 0 -and $Only -notcontains $name) { return }
    $missing = @()
    foreach ($n in $needs) {
        switch ($n) {
            'php'    { if (-not $phpOk) { $missing += 'php with the sodium extension' } }
            'python' { if (-not $py) { $missing += 'python' } }
            'pycrypto' { if (-not $pyCrypto) { $missing += "python's cryptography package" } }
            'node'   { if (-not $node) { $missing += 'node' } }
        }
    }
    Write-Host ''
    Write-Host "==== $name"
    if ($missing.Count -gt 0) {
        Write-Host "SKIPPED: $name needs $($missing -join ', '), which this machine does not have. Nothing was compared."
        $results[$name] = 'SKIPPED'
        return
    }
    try {
        & $body
        $results[$name] = 'ok'
    } catch {
        Write-Host "FAILED: $name : $($_.Exception.Message)"
        $results[$name] = 'FAILED'
        $script:failed++
    }
}

function Json-Count([string]$path) { (Get-Content -LiteralPath $path | Measure-Object -Line).Lines }

# ---- Ed25519: libsodium, OpenSSL (Node and Python) and the crate on hand-built edge cases
Pipeline 'ed25519' @('php', 'python', 'node', 'pycrypto') {
    $d = Join-Path $wt 'rv\ed25519'
    $null = Run $py @('gen_cases.py') $d
    $null = Run $php @('verdict_php.php', 'cases.json', 'out_php.json') $d
    $null = Run $node @('verdict_node.mjs', 'cases.json', 'out_node.json') $d
    $null = Run $py @('verdict_py.py', 'cases.json', 'out_py.json') $d
    Driver 'rv_ed25519_diff' @('the_crates_verdicts_on_the_edge_cases') @{
        RV_ED_CASES = (Join-Path $d 'cases.json'); RV_ED_OUT = (Join-Path $d 'out_rust.json'); RV_ED_SODIUM = (Join-Path $d 'out_php.json') }
    $report = Run $py @('compare.py') $d
    $line = $report | Where-Object { $_ -match '^libsodium vs crate: .* (\d+)$' } | Select-Object -First 1
    if (-not $line -or $Matches[1] -ne '0') { Fail "libsodium and the crate differ: $line" }
    $line = $report | Where-Object { $_ -match '^relay key validity .* differ on (\d+)$' } | Select-Object -First 1
    if (-not $line) { Fail 'the key-validity line is missing from compare.py' }
    Write-Host "    key validity: $line"
}

# ---- X25519
Pipeline 'x25519' @('php') {
    $f = Join-Path $work 'x25519.json'
    $null = Run $php @((Join-Path $wt 'rv\x25519\x25519_gen.php'), $f)
    Driver 'rv_x25519_diff' @('the_crate_refuses_what_libsodium_refuses_and_agrees_on_the_rest') @{ RV_X_IN = $f }
}

# ---- The arithmetic of the pairing
Pipeline 'math' @('python') {
    $d = Join-Path $wt 'rv\pairing_math'
    $null = Run $py @('gen_math.py') $d
    Driver 'rv_pairing_math_diff' @('the_pairing_arithmetic_equals_an_independent_recomputation') @{ RV_MATH_IN = (Join-Path $d 'math_cases.json') }
}

# ---- Sealed boxes, both ways
Pipeline 'sealed' @('php') {
    $in = Join-Path $work 'seal_cases.json'
    $out = Join-Path $work 'rust_sealed.json'
    $null = Run $php @((Join-Path $wt 'rv\sealed\seal_gen.php'), $in)
    Driver 'rv_sealed_diff' @('libsodiums_sealed_boxes_open_here_and_the_crates_open_in_libsodium') @{ RV_SEAL_IN = $in; RV_SEAL_OUT = $out }
    $null = Run $php @((Join-Path $wt 'rv\sealed\seal_check.php'), $in, $out)
}

# ---- The poll decision core against an independent implementation of the README
Pipeline 'poll' @('python') {
    $d = Join-Path $work 'poll'
    New-Item -ItemType Directory -Path $d | Out-Null
    # The generator's own implementation of the README is strict by default (a calendar date that does not exist is no date, a seq above 2^53 - 1 counts, -0 is 0). The flags make
    # it follow the three readings the README now fixes and the repository's two readers share: dates are added as they stand (timegm), a seq above 2^53 - 1 is dropped, -0 is not an integer
    # (Interpretation 23), and a failure that the client itself caused (invalid_request, storage_failure) does not take the pause the relay asked for.
    $env:RV_FLAGS = 'lenient_date,seq_cap,no_d_on_own,neg_zero_not_int'
    try { $null = Run $py @((Join-Path $wt 'rv\poll\gen_poll.py'), "$PollCases", '1', $d) } finally { $env:RV_FLAGS = $null }
    Driver 'rv_poll_diff' @('rv_poll_differential') @{ RV_POLL_IN = (Join-Path $d 'cases.jsonl'); RV_POLL_OUT = (Join-Path $d 'crate.jsonl') }
    $report = Run $py @((Join-Path $wt 'rv\poll\compare_poll.py'), $d)
    $line = $report | Where-Object { $_ -match '^(\d+) cases; (\d+) disagree' } | Select-Object -First 1
    if (-not $line) { Fail 'compare_poll.py printed no summary' }
    if ($Matches[2] -ne '0') { Fail "the poll core disagrees with the independent implementation on $($Matches[2]) of $($Matches[1]) cases (see $d\disagree.txt)" }
}

# ---- The admission readers against the Python port of the shipped decoders
Pipeline 'admission' @('python') {
    $d = Join-Path $work 'adm'
    New-Item -ItemType Directory -Path $d | Out-Null
    $null = Run $py @((Join-Path $wt 'rv\admission\gen_adm.py'), "$AdmissionCases", '1', $d)
    Driver 'rv_admission_diff' @('rv_admission_differential') @{ RV_ADM_IN = (Join-Path $d 'cases.jsonl'); RV_ADM_OUT = (Join-Path $d 'rust.jsonl') }
    $report = Run $py @((Join-Path $wt 'rv\admission\cmp_adm.py'), $d)
    $line = $report | Where-Object { $_ -match '^(\d+) cases; (\d+) both accept; (\d+) both refuse; (\d+) disagree; (\d+) python crashes' } | Select-Object -First 1
    if (-not $line) { Fail 'cmp_adm.py printed no summary' }
    $disagree = [int]$Matches[4]
    # With the default 2000 copies per recorded case (20,000 cases, seed 1) the count is 1005 and was already 1005 when the crate was reviewed: 516 where the crate accepts and the
    # shipped decoder refuses, 485 the other way, and 4 where the relay advertisement is usable in one and not the other. A change in the count is a change in a reader, and is looked at.
    if ($ExpectedAdmissionDisagreements -lt 0 -and $AdmissionCases -eq 2000) { $ExpectedAdmissionDisagreements = 1005 }
    if ([int]$Matches[5] -ne 0) { Fail 'the python port crashed on a case' }
    if ($ExpectedAdmissionDisagreements -ge 0 -and $disagree -ne $ExpectedAdmissionDisagreements) {
        Fail "expected $ExpectedAdmissionDisagreements known disagreements, got $disagree (see $d\disagree.txt)"
    }
    Write-Host "    $disagree cases where the crate and the shipped decoders differ (stricter on the bearer and the scopes; looser on the echoed endpoint key of a plugin admission, the phone's device record, the content of an ICE credential and the gateway and relayOnly values): see the README of the crate"
}

# ---- Encoding: base64url
Pipeline 'b64' @('php', 'python', 'node') {
    $d = Join-Path $wt 'rv_enc'
    $null = Run $py @('drive_b64.py') $d
    Driver 'rv_enc_corpus' @('corpus_b64') @{ RV_DIR = (Join-Path $d 'work_b64') }
    $report = Run $py @('drive_b64.py') $d
    $line = $report | Where-Object { $_ -match "^\{'py_vs_php': (\d+), 'py_vs_node': (\d+), 'py_vs_rust': (\d+), 'rust_roundtrip_false': (\d+)" } | Select-Object -First 1
    if (-not $line) { Fail 'drive_b64.py printed no counts' }
    if ($Matches[1] -ne '0' -or $Matches[2] -ne '0' -or $Matches[3] -ne '0' -or $Matches[4] -ne '0') { Fail "base64url: $line" }
}

# ---- Encoding: JSON
Pipeline 'json' @('php', 'python', 'node') {
    $d = Join-Path $wt 'rv_enc'
    $null = Run $py @('drive_json.py', 'gen', "$JsonCases") $d
    $null = Run $py @('drive_json.py', 'refs') $d
    Driver 'rv_enc_corpus' @('corpus_json') @{ RV_DIR = (Join-Path $d 'work_json') }
    $report = Run $py @('drive_json.py', 'compare') $d
    # Where the crate is stricter than serde_json, Node and PHP's json_decode it is on purpose (README 1: a duplicate member, a lone surrogate and more than 64 levels are refused),
    # it accepts 1e999 as Python does and serde_json does not, and its canonical form sorts keys by UTF-8 bytes where RFC 8785 sorts by UTF-16 code units (839 keys of this corpus);
    # these are counted and printed, and stated in the README of the crate. What is asserted is the comparison with the strict Python reference of the README: the general reader
    # accepts exactly what it accepts and gives the same text, the canonical reader the same canonical text.
    $differs = @($report | Where-Object { $_ -cmatch '^\d+ (GEN|CANON) ' })   # case matters: "canon outputs where UTF-16" is the counted, documented difference
    if ($differs) { Fail "JSON: the crate differs from the strict reference: $($differs -join '; ')" }
    $acc = @{}
    foreach ($l in $report) { if ($l -match '^(\d+) (py_accept|rust_general_accept)$') { $acc[$Matches[2]] = $Matches[1] } }
    if (-not $acc['py_accept'] -or $acc['py_accept'] -ne $acc['rust_general_accept']) { Fail "JSON: accepted by the reference $($acc['py_accept']), by the crate $($acc['rust_general_accept'])" }
}

# ---- Encoding: the text rules (typed code, SAS entries, ids, names, urls)
Pipeline 'text' @('php', 'python') {
    $d = Join-Path $wt 'rv_enc'
    $null = Run $py @('drive_text.py', 'gen') $d
    $null = Run $py @('drive_text.py', 'php') $d
    Driver 'rv_enc_corpus' @('corpus_typed', 'corpus_sas_entry', 'corpus_ids', 'corpus_name', 'corpus_url') @{ RV_DIR = (Join-Path $d 'work_text') }
    $report = Run $py @('drive_text.py', 'compare') $d
    function Expect([string]$pattern, [string]$what, [int]$wanted = 0) {
        $line = $report | Where-Object { $_ -match $pattern } | Select-Object -First 1
        if (-not $line) { Fail "text: no line for $what in the report" }
        if ([int]$Matches[1] -ne $wanted) { Fail "text: $what is $($Matches[1]), expected $wanted ($line)" }
    }
    if (-not ($report | Where-Object { $_ -match '^typed: .* no disagreement with the reference' })) { Fail 'text: the typed codes disagree with the reference' }
    Expect '^sas: .*; disagreements (\d+)' 'the SAS entries'
    Expect '^ids: .*disagreements: (\d+)' 'the identifiers against Ids.php'
    Expect '^names: .*disagreements: (\d+)' 'the names against cleanName'
    # 16 relay URLs are refused or written otherwise by the crate than by the pattern of the schemas, all on purpose: an empty label (a..b), a trailing dot, a label that starts or ends
    # with a hyphen, a host or label that is too long, and a default port (:443), which is no port, so that https://h and https://h:443 are the same relay (and a port with a leading zero is no port).
    Expect '^urls: .*disagreements with the schema-pattern reference: (\d+)' 'the relay URLs against the pattern of the schemas' $ExpectedUrlDisagreements
}

Write-Host ''
Write-Host '==== summary'
$results.GetEnumerator() | ForEach-Object { Write-Host ("    {0,-10} {1}" -f $_.Key, $_.Value) }
if ($Keep) { Write-Host "scratch kept: $work" } else { [System.IO.Directory]::Delete($work, $true) }
if ($results.Count -eq 0) { Write-Host 'FAILED: no pipeline ran'; exit 1 }
$skipped = @($results.Values | Where-Object { $_ -eq 'SKIPPED' }).Count
if ($failed -gt 0) { Write-Host "FAILED: $failed pipeline(s)"; exit 1 }
if ($skipped -gt 0 -and -not $AllowSkips) { Write-Host "SKIPPED: $skipped pipeline(s) did not run: that is not a pass"; exit 3 }
Write-Host 'all pipelines that could run passed'
exit 0
