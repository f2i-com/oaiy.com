"""UUID objects (pure Python, for OAIY's sandbox)."""
import os
import hashlib

RESERVED_NCS = 'reserved for NCS compatibility'
RFC_4122 = 'specified in RFC 4122'


class UUID:
    def __init__(self, hex=None, bytes=None, int=None, version=None):
        if hex is not None:
            h = hex.replace('urn:', '').replace('uuid:', '').strip('{}').replace('-', '')
            if len(h) != 32:
                raise ValueError('badly formed hexadecimal UUID string')
            value = _int(h, 16)
        elif bytes is not None:
            if len(bytes) != 16:
                raise ValueError('bytes is not a 16-char string')
            value = 0
            for b in bytes:
                value = (value << 8) | b
        elif int is not None:
            value = int
        else:
            raise TypeError('one of the hex, bytes or int arguments must be given')
        if version is not None:
            value &= ~(0xc000 << 48)
            value |= 0x8000 << 48
            value &= ~(0xf000 << 64)
            value |= version << 76
        self.int = value

    @property
    def hex(self):
        return '%032x' % self.int

    @property
    def bytes(self):
        out = []
        v = self.int
        for _ in range(16):
            out.append(v & 255)
            v >>= 8
        out.reverse()
        return _bytes(out)

    @property
    def urn(self):
        return 'urn:uuid:' + str(self)

    @property
    def version(self):
        return (self.int >> 76) & 0xf

    @property
    def variant(self):
        return RFC_4122

    def __str__(self):
        h = self.hex
        return '%s-%s-%s-%s-%s' % (h[:8], h[8:12], h[12:16], h[16:20], h[20:])

    def __repr__(self):
        return "UUID('%s')" % str(self)

    def __eq__(self, other):
        return isinstance(other, UUID) and self.int == other.int

    def __lt__(self, other):
        return self.int < other.int

    def __hash__(self):
        return hash(self.int)


_int = int
_bytes = bytes


def _from_bytes(data, version):
    value = 0
    for b in data[:16]:
        value = (value << 8) | b
    return UUID(int=value, version=version)


def uuid4():
    return _from_bytes(os.urandom(16), 4)


def uuid1(node=None, clock_seq=None):
    return _from_bytes(os.urandom(16), 1)


def uuid3(namespace, name):
    return _from_bytes(hashlib.md5(namespace.bytes + name.encode('utf-8')).digest(), 3)


def uuid5(namespace, name):
    return _from_bytes(hashlib.sha1(namespace.bytes + name.encode('utf-8')).digest(), 5)


NAMESPACE_DNS = UUID('6ba7b810-9dad-11d1-80b4-00c04fd430c8')
NAMESPACE_URL = UUID('6ba7b811-9dad-11d1-80b4-00c04fd430c8')
NAMESPACE_OID = UUID('6ba7b812-9dad-11d1-80b4-00c04fd430c8')
NAMESPACE_X500 = UUID('6ba7b814-9dad-11d1-80b4-00c04fd430c8')
NIL = UUID(int=0)
