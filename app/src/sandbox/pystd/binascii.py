"""Conversions between binary data and ASCII (pure Python, for bot.computer's sandbox)."""


class Error(ValueError):
    pass


class Incomplete(Exception):
    pass


def hexlify(data, sep=None):
    text = ''.join(['%02x' % b for b in bytes(data)])
    if sep:
        s = sep if isinstance(sep, str) else sep.decode('ascii')
        text = s.join([text[i:i + 2] for i in range(0, len(text), 2)])
    return text.encode('ascii')


def unhexlify(data):
    text = data if isinstance(data, str) else bytes(data).decode('ascii')
    if len(text) % 2:
        raise Error('Odd-length string')
    try:
        return bytes([int(text[i:i + 2], 16) for i in range(0, len(text), 2)])
    except ValueError:
        raise Error('Non-hexadecimal digit found')


b2a_hex = hexlify
a2b_hex = unhexlify


def crc32(data, value=0):
    crc = value ^ 0xFFFFFFFF
    for b in bytes(data):
        crc ^= b
        for _ in range(8):
            if crc & 1:
                crc = (crc >> 1) ^ 0xEDB88320
            else:
                crc >>= 1
    return crc ^ 0xFFFFFFFF


def b2a_base64(data, newline=True):
    import base64
    out = base64.b64encode(data)
    return out + b'\n' if newline else out


def a2b_base64(data):
    import base64
    return base64.b64decode(data)
