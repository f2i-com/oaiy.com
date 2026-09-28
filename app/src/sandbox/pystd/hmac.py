"""HMAC message authentication (pure Python over hashlib, for bot.computer's sandbox)."""
import hashlib
import binascii

digest_size = None


def _hash(digestmod):
    if isinstance(digestmod, str):
        return lambda data=b'': hashlib.new(digestmod, data)
    if hasattr(digestmod, 'new'):
        return digestmod.new
    return digestmod


class HMAC:
    blocksize = 64

    def __init__(self, key, msg=None, digestmod=''):
        if not digestmod:
            raise TypeError("Missing required parameter 'digestmod'.")
        self._make = _hash(digestmod)
        probe = self._make()
        self.digest_size = probe.digest_size if hasattr(probe, 'digest_size') else len(probe.digest())
        self.block_size = getattr(probe, 'block_size', 64)
        self.name = 'hmac-' + getattr(probe, 'name', 'hash')
        key = bytes(key)
        if len(key) > self.block_size:
            key = self._make(key).digest()
        key = key + bytes(self.block_size - len(key))
        self._okey = bytes([b ^ 0x5c for b in key])
        self._ikey = bytes([b ^ 0x36 for b in key])
        self._msg = b''
        if msg is not None:
            self.update(msg)

    def update(self, msg):
        self._msg = self._msg + bytes(msg)

    def copy(self):
        other = HMAC.__new__(HMAC)
        other.__dict__.update(self.__dict__)
        return other

    def digest(self):
        inner = self._make(self._ikey + self._msg).digest()
        return self._make(self._okey + inner).digest()

    def hexdigest(self):
        return binascii.hexlify(self.digest()).decode('ascii')


def new(key, msg=None, digestmod=''):
    return HMAC(key, msg, digestmod)


def digest(key, msg, digest):
    return HMAC(key, msg, digest).digest()


def compare_digest(a, b):
    if len(a) != len(b):
        return False
    diff = 0
    for x, y in zip(a, b):
        diff |= (ord(x) if isinstance(x, str) else x) ^ (ord(y) if isinstance(y, str) else y)
    return diff == 0
