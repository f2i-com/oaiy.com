"""Base16, Base32 and Base64 encodings (pure Python, for bot.computer's sandbox)."""
import binascii

_B64 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/'
_B32 = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ234567'


def _bytes(s):
    if isinstance(s, str):
        return s.encode('ascii')
    return bytes(s)


def b64encode(s, altchars=None):
    data = _bytes(s)
    out = []
    for i in range(0, len(data), 3):
        chunk = data[i:i + 3]
        n = chunk[0] << 16
        if len(chunk) > 1:
            n |= chunk[1] << 8
        if len(chunk) > 2:
            n |= chunk[2]
        out.append(_B64[(n >> 18) & 63])
        out.append(_B64[(n >> 12) & 63])
        out.append(_B64[(n >> 6) & 63] if len(chunk) > 1 else '=')
        out.append(_B64[n & 63] if len(chunk) > 2 else '=')
    text = ''.join(out)
    if altchars is not None:
        alt = _bytes(altchars).decode('ascii')
        text = text.replace('+', alt[0]).replace('/', alt[1])
    return text.encode('ascii')


def b64decode(s, altchars=None, validate=False):
    text = _bytes(s).decode('ascii')
    if altchars is not None:
        alt = _bytes(altchars).decode('ascii')
        text = text.replace(alt[0], '+').replace(alt[1], '/')
    clean = ''
    for c in text:
        if c in _B64:
            clean += c
        elif c == '=':
            clean += c
        elif validate and not c.isspace():
            raise binascii.Error('Non-base64 digit found')
    clean = clean.rstrip('=')
    if len(clean) % 4 == 1:
        raise binascii.Error('Incorrect padding')
    out = []
    buf = 0
    bits = 0
    for c in clean:
        buf = (buf << 6) | _B64.index(c)
        bits += 6
        if bits >= 8:
            bits -= 8
            out.append((buf >> bits) & 255)
    return bytes(out)


def standard_b64encode(s):
    return b64encode(s)


def standard_b64decode(s):
    return b64decode(s)


def urlsafe_b64encode(s):
    return b64encode(s, b'-_')


def urlsafe_b64decode(s):
    return b64decode(s, b'-_')


def encodebytes(s):
    text = b64encode(s).decode('ascii')
    lines = [text[i:i + 76] for i in range(0, len(text), 76)]
    return ('\n'.join(lines) + '\n').encode('ascii') if lines else b''


def decodebytes(s):
    return b64decode(s)


def b16encode(s):
    return ''.join(['%02X' % b for b in _bytes(s)]).encode('ascii')


def b16decode(s, casefold=False):
    text = _bytes(s).decode('ascii')
    if casefold:
        text = text.upper()
    return bytes([int(text[i:i + 2], 16) for i in range(0, len(text), 2)])


def b32encode(s):
    data = _bytes(s)
    out = ''
    buf = 0
    bits = 0
    for b in data:
        buf = (buf << 8) | b
        bits += 8
        while bits >= 5:
            bits -= 5
            out += _B32[(buf >> bits) & 31]
    if bits:
        out += _B32[(buf << (5 - bits)) & 31]
    while len(out) % 8:
        out += '='
    return out.encode('ascii')


def b32decode(s, casefold=False):
    text = _bytes(s).decode('ascii').rstrip('=')
    if casefold:
        text = text.upper()
    out = []
    buf = 0
    bits = 0
    for c in text:
        buf = (buf << 5) | _B32.index(c)
        bits += 5
        if bits >= 8:
            bits -= 8
            out.append((buf >> bits) & 255)
    return bytes(out)
