"""Tokens and random choices for secrets (pure Python, for OAIY's sandbox)."""
import os
import base64
import binascii

DEFAULT_ENTROPY = 32


def token_bytes(nbytes=None):
    return os.urandom(DEFAULT_ENTROPY if nbytes is None else nbytes)


def token_hex(nbytes=None):
    return binascii.hexlify(token_bytes(nbytes)).decode('ascii')


def token_urlsafe(nbytes=None):
    return base64.urlsafe_b64encode(token_bytes(nbytes)).rstrip(b'=').decode('ascii')


def randbelow(n):
    if n <= 0:
        raise ValueError('Upper bound must be positive.')
    data = os.urandom(8)
    value = 0
    for b in data:
        value = (value << 8) | b
    return value % n


def randbits(k):
    nbytes = (k + 7) // 8
    value = 0
    for b in os.urandom(nbytes):
        value = (value << 8) | b
    return value >> (nbytes * 8 - k)


def choice(seq):
    return seq[randbelow(len(seq))]


def compare_digest(a, b):
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= (ord(x) if isinstance(x, str) else x) ^ (ord(y) if isinstance(y, str) else y)
    return diff == 0


class SystemRandom:
    def random(self):
        return randbits(53) / (1 << 53)

    def randrange(self, start, stop=None):
        if stop is None:
            return randbelow(start)
        return start + randbelow(stop - start)

    def randint(self, a, b):
        return a + randbelow(b - a + 1)

    def choice(self, seq):
        return choice(seq)
