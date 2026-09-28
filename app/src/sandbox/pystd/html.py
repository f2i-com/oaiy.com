"""HTML escaping (pure Python, for bot.computer's sandbox)."""

_ENTITIES = {
    'amp': '&', 'lt': '<', 'gt': '>', 'quot': '"', 'apos': "'", 'nbsp': '\xa0', 'copy': '\xa9', 'reg': '\xae',
    'trade': '™', 'hellip': '…', 'mdash': '—', 'ndash': '–', 'lsquo': '‘', 'rsquo': '’',
    'ldquo': '“', 'rdquo': '”', 'bull': '•', 'middot': '\xb7', 'euro': '€', 'pound': '\xa3',
    'yen': '\xa5', 'cent': '\xa2', 'deg': '\xb0', 'times': '\xd7', 'divide': '\xf7', 'laquo': '\xab', 'raquo': '\xbb',
    'sect': '\xa7', 'para': '\xb6', 'eacute': '\xe9', 'egrave': '\xe8', 'agrave': '\xe0', 'ccedil': '\xe7', 'uuml': '\xfc',
    'ouml': '\xf6', 'auml': '\xe4', 'szlig': '\xdf', 'larr': '←', 'rarr': '→', 'uarr': '↑', 'darr': '↓',
}


def escape(s, quote=True):
    s = s.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    if quote:
        s = s.replace('"', '&quot;').replace("'", '&#x27;')
    return s


def unescape(s):
    if '&' not in s:
        return s
    out = ''
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c != '&':
            out += c
            i += 1
            continue
        end = s.find(';', i + 1, i + 12)
        if end < 0:
            out += c
            i += 1
            continue
        name = s[i + 1:end]
        if name.startswith('#'):
            try:
                code = int(name[2:], 16) if name[1:2] in ('x', 'X') else int(name[1:])
                out += chr(code)
                i = end + 1
                continue
            except ValueError:
                pass
        elif name in _ENTITIES:
            out += _ENTITIES[name]
            i = end + 1
            continue
        out += c
        i += 1
    return out
