#!/usr/bin/env python3
"""An independent implementation of the client-address rules of design 4.5.4, and a generator of test cases.

The Rust code (`src-tauri/src/auth/clientip.rs`) decides who a request is *from*: which forwarded address is
believed, which peers are trusted proxies. A mistake there is an authentication bypass (a forged
`X-Forwarded-For` taken for a client, a network that trusts more than it says). This file re-implements the rules
from the design, in another language and on another library (Python's `ipaddress`, which shares no code with
Rust's standard library), and writes cases with the answers it computes. The Rust test
`auth::clientip_crosscheck` reads them and fails on any difference.

    python clientip-reference.py generate --count 36000 --seed 7 --out cases.json
    python clientip-reference.py summary cases.json

The rules, as the design gives them (4.5.4):

  * IPv4-mapped IPv6 addresses (`::ffff:a.b.c.d`) are unmapped, for the peer, for every forwarded entry and for
    every trusted network, before anything is compared.
  * If the peer is not a trusted proxy it is the client, and every forwarded header is ignored.
  * If it is, all `X-Forwarded-For` lines are joined with `,` and walked from the right; the first entry that is
    not a trusted proxy is the client. One entry that is not an address, a trusted peer with no header (or one
    with nothing but empty entries), or a header made only of trusted proxies falls back to the peer.
  * An IPv6 client is keyed by its /64 (16 hex digits), an IPv4 one by its dotted address.

What counts as an address is strict: an IPv4 or IPv6 literal with no zone, brackets, port, quotes or spaces inside,
and no leading zeros in an IPv4 octet. What counts as a network: an address (a host route) or `address/digits`,
with the prefix at most the width of the family; an address written in the mapped form with a prefix is the IPv4
network with the prefix 96 shorter, and with a prefix below 96 is refused.
"""

import argparse
import ipaddress
import json
import random
import sys

# ---- the rules ---------------------------------------------------------------------------------


def strict_ip(text):
    """An address exactly as written, or None."""
    if not isinstance(text, str) or not text or not text.isascii():
        return None
    if any(c in text for c in "%/[]\\ \t\r\n\x00\x0b\x0c\"'"):
        return None
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def unmap(ip):
    if ip.version == 6 and ip.ipv4_mapped is not None:
        return ip.ipv4_mapped
    return ip


def parse_network(text):
    """(version, network address as an int, prefix) or None."""
    t = text.strip()
    if "/" in t:
        addr, pfx = t.split("/", 1)
    else:
        addr, pfx = t, None
    written = strict_ip(addr)
    if written is None:
        return None
    ip = unmap(written)
    mapped = written.version == 6 and ip.version == 4
    width = 32 if ip.version == 4 else 128
    if pfx is None:
        prefix = width
    else:
        if pfx == "" or len(pfx) > 3 or any(c not in "0123456789" for c in pfx):
            return None
        n = int(pfx)
        prefix = n - 96 if mapped else n
        if prefix < 0 or prefix > width:
            return None
    net = ipaddress.ip_network((ip, prefix), strict=False)
    return (ip.version, int(net.network_address), prefix)


def net_contains(net, ip):
    version, base, prefix = net
    if ip.version != version:
        return False
    width = 32 if version == 4 else 128
    shift = width - prefix
    return (int(ip) >> shift) == (base >> shift)


def bucket_key(ip):
    if ip.version == 4:
        return str(ip)
    return "%016x" % (int(ip) >> 64)


def parse_trusted(entries):
    """The networks of OAIY_TRUSTED_PROXIES: the entries are one comma list (an entry that has a comma in it is
    two), each trimmed, the empty ones dropped, and an entry that is not a network is dropped (the server refuses to
    start with one; the rest is what this reads)."""
    nets = []
    for part in ",".join(entries).split(","):
        part = part.strip()
        if part == "":
            continue
        n = parse_network(part)
        if n is not None:
            nets.append(n)
    return nets


def client_ip(peer_text, xff_lines, trusted_entries):
    """(ip, via_proxy, fell_back) for the peer and header lines, given the entries of OAIY_TRUSTED_PROXIES."""
    trusted = parse_trusted(trusted_entries)

    def is_trusted(ip):
        return any(net_contains(n, ip) for n in trusted)

    peer = unmap(strict_ip(peer_text))
    if not is_trusted(peer):
        return peer, False, False
    entries = []
    for line in xff_lines:
        entries.extend(part.strip() for part in line.split(","))
    if not entries or all(e == "" for e in entries):
        return peer, False, False
    parsed = []
    for e in entries:
        ip = strict_ip(e)
        if ip is None:
            return peer, False, True
        parsed.append(unmap(ip))
    for ip in reversed(parsed):
        if not is_trusted(ip):
            return ip, True, False
    return peer, False, True


# ---- the cases ---------------------------------------------------------------------------------


def hexint(n):
    return "%x" % n


def rnd_v4(r):
    return ".".join(str(r.randrange(256)) for _ in range(4))


def rnd_v6(r):
    groups = []
    for _ in range(8):
        pick = r.random()
        if pick < 0.25:
            groups.append(0)
        elif pick < 0.35:
            groups.append(0xFFFF)
        else:
            groups.append(r.randrange(65536))
    return str(ipaddress.IPv6Address(sum(g << (16 * (7 - i)) for i, g in enumerate(groups))))


PUBLIC_V4_PREFIXES = ["203.0.113.", "198.51.100.", "192.0.2.", "8.8.", "1.1.1.", "45.33.", "185.199."]
PRIVATE_V4_PREFIXES = ["10.", "172.16.", "172.30.", "192.168.", "127.0.0.", "100.64.", "169.254."]


def near_v4(r):
    prefix = r.choice(PUBLIC_V4_PREFIXES + PRIVATE_V4_PREFIXES)
    parts = prefix.rstrip(".").split(".")
    while len(parts) < 4:
        parts.append(str(r.randrange(256)))
    return ".".join(parts)


def near_v6(r):
    base = r.choice(["2001:db8:", "2001:db8:1:", "fd12:3456:", "fe80:0:", "2606:4700:", "::"])
    if base == "::":
        return r.choice(["::1", "::", "::2", "::ffff:0:1"])
    tail = ":".join("%x" % r.randrange(65536) for _ in range(r.randrange(1, 4)))
    try:
        return str(ipaddress.IPv6Address(base + tail if base.endswith(":") else base + ":" + tail))
    except ValueError:
        return str(ipaddress.IPv6Address(rnd_v6(r)))


def address_text(r):
    """A valid address, in some form."""
    kind = r.random()
    if kind < 0.35:
        v4 = near_v4(r) if r.random() < 0.7 else rnd_v4(r)
        return v4
    if kind < 0.55:
        return near_v6(r) if r.random() < 0.6 else rnd_v6(r)
    if kind < 0.75:
        # the mapped forms of an IPv4 address
        v4 = near_v4(r) if r.random() < 0.7 else rnd_v4(r)
        form = r.random()
        if form < 0.5:
            return "::ffff:" + v4
        if form < 0.7:
            return "::FFFF:" + v4
        if form < 0.85:
            ip = ipaddress.IPv4Address(v4)
            return "::ffff:%x:%x" % (int(ip) >> 16, int(ip) & 0xFFFF)
        return "0:0:0:0:0:ffff:" + v4
    if kind < 0.9:
        ip = ipaddress.IPv6Address(near_v6(r) if r.random() < 0.6 else rnd_v6(r))
        form = r.random()
        if form < 0.35:
            return ip.exploded
        if form < 0.6:
            return ip.exploded.upper()
        if form < 0.8:
            return str(ip).upper()
        return str(ip)
    return r.choice(["127.0.0.1", "::1", "0.0.0.0", "::", "255.255.255.255", "::ffff:127.0.0.1", "::ffff:0.0.0.0"])


def garbage_entry(r):
    pool = [
        "unknown", "_hidden", "for=1.2.3.4", "\"1.2.3.4\"", "'1.2.3.4'", "1.2.3.4:80", "[::1]:80", "[::1]",
        "1.2.3", "1.2.3.4.5", "0x7f.0.0.1", "01.2.3.4", "1.2.3.04", "127.1", "2130706433", "1.2.3.4/24",
        "fe80::1%eth0", "fe80::1%1", "::1%lo", "1.2.3.256", "256.1.1.1", "-1.2.3.4", "+1.2.3.4", "1.2.3.4.",
        ".1.2.3.4", "1..2.3", "::g", ":::", "1:2:3:4:5:6:7:8:9", "12345::1", "::ffff:1.2.3", "::ffff:1.2.3.4.5",
        "::ffff:01.2.3.4", "１２７.0.0.1", "1.2.3.4\x00", "1.2.3.4\r", "1 .2.3.4", "1. 2.3.4",
        "localhost", "example.com", "null", "true", "0", "1", "-", "*", "::1::", "1::2::3", "0000:0000::1:",
        "unknown, 1.2.3.4", "obfuscated", "_abc123", "10.0.0.1 10.0.0.2", "10.0.0.1;10.0.0.2", "Bearer x",
    ]
    return r.choice(pool)


def decorate(r, text):
    """Spaces and tabs around an entry, as proxies write them."""
    return r.choice(["", "", "", " ", "  ", "\t", " \t"]) + text + r.choice(["", "", "", " ", "  ", "\t"])


def network_text(r):
    """An entry of OAIY_TRUSTED_PROXIES: mostly valid, sometimes hostile."""
    kind = r.random()
    if kind < 0.18:
        return address_text(r)
    if kind < 0.5:
        v4 = near_v4(r) if r.random() < 0.7 else rnd_v4(r)
        return "%s/%d" % (v4, r.choice([0, 8, 12, 16, 20, 24, 24, 24, 28, 30, 31, 32, r.randrange(0, 33)]))
    if kind < 0.68:
        v6 = near_v6(r) if r.random() < 0.6 else rnd_v6(r)
        return "%s/%d" % (v6, r.choice([0, 7, 32, 48, 56, 64, 64, 96, 112, 120, 127, 128, r.randrange(0, 129)]))
    if kind < 0.8:
        v4 = near_v4(r) if r.random() < 0.7 else rnd_v4(r)
        return "::ffff:%s/%d" % (v4, r.choice([96, 104, 112, 120, 128, r.randrange(0, 129)]))
    if kind < 0.9:
        return r.choice([
            "10.0.0.0/+8", "10.0.0.0/-1", "10.0.0.0/ 8", "10.0.0.0 /8", "10.0.0.0/8 ", " 10.0.0.0/8", "10.0.0.0/",
            "/8", "10.0.0.0/8/8", "10.0.0.0/0x8", "10.0.0.0/08", "10.0.0.0/008", "10.0.0.0/0008", "10.0.0.0/33",
            "::1/129", "::1/-1", "::1/+128", "::ffff:10.0.0.0/8", "::ffff:10.0.0.0/95", "::ffff:10.0.0.0/129",
            "::ffff:0.0.0.0/96", "::ffff:0:0/96", "0.0.0.0/0", "::/0", "::ffff:127.0.0.1/128", "fe80::1%eth0/64",
            "[::1]/128", "127.0.0.1:8080", "nginx", "", " ", ",", "10.0.0.5, 10.0.0.6", "2001:db8::/32/1",
            "1.2.3.4/24\x00", "１２７.0.0.1/8",
        ])
    return garbage_entry(r)


def member_of(r, net):
    """A valid address text inside `net` (version, base, prefix), or None."""
    version, base, prefix = net
    width = 32 if version == 4 else 128
    host_bits = width - prefix
    offset = r.getrandbits(host_bits) if host_bits > 0 else 0
    value = base | offset
    if version == 4:
        ip = ipaddress.IPv4Address(value)
        return str(ip) if r.random() < 0.8 else "::ffff:" + str(ip)
    ip = ipaddress.IPv6Address(value)
    form = r.random()
    return ip.exploded if form < 0.2 else (str(ip).upper() if form < 0.4 else str(ip))


def gen_ip(r):
    kind = r.random()
    if kind < 0.6:
        text = address_text(r)
    elif kind < 0.9:
        text = garbage_entry(r)
    else:
        text = address_text(r) + r.choice([":80", " ", "\t", "%eth0", "/32", ".", "\n", "\x00", "]"])
    ip = strict_ip(text)
    case = {"kind": "ip", "text": text, "ok": ip is not None}
    if ip is not None:
        ip = unmap(ip)
        case["v4"] = ip.version == 4
        case["value"] = hexint(int(ip))
    return case


def gen_cidr(r):
    text = network_text(r)
    net = parse_network(text)
    case = {"kind": "cidr", "text": text, "ok": net is not None}
    if net is not None:
        case["v4"] = net[0] == 4
        case["value"] = hexint(net[1])
        case["prefix"] = net[2]
    return case


def gen_client(r):
    entries = []
    for _ in range(r.choice([1, 1, 2, 2, 3, 4])):
        entries.append(network_text(r) if r.random() < 0.85 else garbage_entry(r))
    if r.random() < 0.3:
        entries.append(r.choice(["127.0.0.1/32", "::1/128", "127.0.0.1", "::1"]))
    nets = parse_trusted(entries)
    # A peer: often a member of a trusted network, sometimes not, sometimes the odd one.
    pick = r.random()
    if nets and pick < 0.55:
        peer = member_of(r, r.choice(nets))
    elif pick < 0.9:
        peer = address_text(r)
    else:
        peer = r.choice(["127.0.0.1", "::1", "::ffff:127.0.0.1", "203.0.113.50", "0.0.0.0", "::"])
    lines = []
    for _ in range(r.choice([0, 0, 1, 1, 1, 2, 2, 3])):
        parts = []
        for _ in range(r.choice([0, 1, 1, 2, 2, 3, 4])):
            pick = r.random()
            if nets and pick < 0.3:
                text = member_of(r, r.choice(nets))
            elif pick < 0.8:
                text = address_text(r)
            elif pick < 0.92:
                text = garbage_entry(r)
            else:
                text = ""
            parts.append(decorate(r, text))
        lines.append(",".join(parts))
    ip, via, fell = client_ip(peer, lines, entries)
    return {
        "kind": "client",
        "peer": peer,
        "xff": lines,
        "trusted": entries,
        "v4": ip.version == 4,
        "value": hexint(int(ip)),
        "key": bucket_key(ip),
        "via": via,
        "fell": fell,
    }


def generate(count, seed):
    r = random.Random(seed)
    cases = []
    n_client = count * 2 // 3
    n_ip = (count - n_client) // 2
    n_cidr = count - n_client - n_ip
    for _ in range(n_client):
        cases.append(gen_client(r))
    for _ in range(n_ip):
        cases.append(gen_ip(r))
    for _ in range(n_cidr):
        cases.append(gen_cidr(r))
    r.shuffle(cases)
    return cases


def summarize(cases):
    counts = {}
    for c in cases:
        k = c["kind"]
        counts[k] = counts.get(k, 0) + 1
        if k == "client":
            if c["via"]:
                key = "client: believed a forwarded address"
            elif c["fell"]:
                key = "client: fell back to a trusted peer (header unusable)"
            else:
                trusted = parse_trusted(c["trusted"])
                peer = unmap(strict_ip(c["peer"]))
                key = ("client: untrusted peer" if not any(net_contains(n, peer) for n in trusted)
                       else "client: trusted peer, no header")
            counts[key] = counts.get(key, 0) + 1
        elif k in ("ip", "cidr"):
            key = "%s: %s" % (k, "accepted" if c["ok"] else "refused")
            counts[key] = counts.get(key, 0) + 1
    return counts


def main(argv):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("generate")
    g.add_argument("--count", type=int, default=36000)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", required=True)
    s = sub.add_parser("summary")
    s.add_argument("file")
    args = p.parse_args(argv)
    if args.cmd == "generate":
        cases = generate(args.count, args.seed)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(cases, f, ensure_ascii=True, separators=(",", ":"))
        for k, v in sorted(summarize(cases).items()):
            print("%8d  %s" % (v, k))
        print("%8d  total -> %s" % (len(cases), args.out))
    else:
        with open(args.file, encoding="utf-8") as f:
            cases = json.load(f)
        for k, v in sorted(summarize(cases).items()):
            print("%8d  %s" % (v, k))
        print("%8d  total" % len(cases))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
