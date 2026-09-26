"""Parse URLs and query strings (pure Python, for bot.computer's sandbox)."""

_ALWAYS_SAFE = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_.-~'
uses_netloc = ['', 'ftp', 'http', 'https', 'ws', 'wss', 'file', 'git', 'ssh', 'sftp']


def _utf8(s):
    return s.encode('utf-8') if isinstance(s, str) else bytes(s)


def quote(string, safe='/', encoding=None, errors=None):
    ok = _ALWAYS_SAFE + (safe if isinstance(safe, str) else safe.decode('ascii'))
    out = ''
    for b in _utf8(string):
        c = chr(b)
        out += c if b < 128 and c in ok else '%%%02X' % b
    return out


def quote_plus(string, safe='', encoding=None, errors=None):
    if ' ' in string:
        return quote(string, safe + ' ').replace(' ', '+')
    return quote(string, safe)


def unquote(string, encoding='utf-8', errors='replace'):
    data = []
    i = 0
    n = len(string)
    while i < n:
        c = string[i]
        if c == '%' and i + 2 < n + 1 and i + 2 <= n - 1 + 1:
            h = string[i + 1:i + 3]
            if len(h) == 2 and all(ch in '0123456789abcdefABCDEF' for ch in h):
                data.append(int(h, 16))
                i += 3
                continue
        for b in c.encode('utf-8'):
            data.append(b)
        i += 1
    return bytes(data).decode(encoding, errors)


def unquote_plus(string, encoding='utf-8', errors='replace'):
    return unquote(string.replace('+', ' '), encoding, errors)


def urlencode(query, doseq=False, safe='', encoding=None, errors=None, quote_via=quote_plus):
    items = query.items() if hasattr(query, 'items') else query
    parts = []
    for k, v in items:
        if doseq and isinstance(v, (list, tuple)):
            for x in v:
                parts.append(quote_via(str(k), safe) + '=' + quote_via(str(x), safe))
        else:
            parts.append(quote_via(str(k), safe) + '=' + quote_via(str(v), safe))
    return '&'.join(parts)


def parse_qsl(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace', max_num_fields=None, separator='&'):
    out = []
    for pair in qs.split(separator):
        if not pair:
            continue
        if '=' in pair:
            k, v = pair.split('=', 1)
        else:
            k, v = pair, ''
        if v or keep_blank_values:
            out.append((unquote_plus(k, encoding, errors), unquote_plus(v, encoding, errors)))
    return out


def parse_qs(qs, keep_blank_values=False, strict_parsing=False, encoding='utf-8', errors='replace', max_num_fields=None, separator='&'):
    out = {}
    for k, v in parse_qsl(qs, keep_blank_values, strict_parsing, encoding, errors, max_num_fields, separator):
        out.setdefault(k, []).append(v)
    return out


class _Result:
    _fields = ()

    def __init__(self, *values):
        for name, value in zip(self._fields, values):
            setattr(self, name, value)

    def __iter__(self):
        return iter([getattr(self, f) for f in self._fields])

    def __getitem__(self, i):
        return getattr(self, self._fields[i])

    def __len__(self):
        return len(self._fields)

    def __eq__(self, other):
        return tuple(self) == tuple(other)

    def _replace(self, **kw):
        values = [kw.get(f, getattr(self, f)) for f in self._fields]
        return type(self)(*values)

    def _host_port(self):
        net = self.netloc.rsplit('@', 1)[-1]
        if net.startswith('['):
            end = net.find(']')
            host = net[1:end]
            rest = net[end + 1:]
            port = rest[1:] if rest.startswith(':') else ''
        elif ':' in net:
            host, port = net.rsplit(':', 1)
        else:
            host, port = net, ''
        return host.lower() or None, (int(port) if port.isdigit() else None)

    @property
    def hostname(self):
        return self._host_port()[0]

    @property
    def port(self):
        return self._host_port()[1]

    @property
    def username(self):
        if '@' not in self.netloc:
            return None
        return self.netloc.rsplit('@', 1)[0].split(':', 1)[0]

    @property
    def password(self):
        if '@' not in self.netloc:
            return None
        user = self.netloc.rsplit('@', 1)[0]
        return user.split(':', 1)[1] if ':' in user else None

    def __repr__(self):
        return '%s(%s)' % (type(self).__name__, ', '.join(['%s=%r' % (f, getattr(self, f)) for f in self._fields]))


class SplitResult(_Result):
    _fields = ('scheme', 'netloc', 'path', 'query', 'fragment')

    def geturl(self):
        return urlunsplit(self)


class ParseResult(_Result):
    _fields = ('scheme', 'netloc', 'path', 'params', 'query', 'fragment')

    def geturl(self):
        return urlunparse(self)


def urlsplit(url, scheme='', allow_fragments=True):
    fragment = ''
    query = ''
    netloc = ''
    s = scheme
    i = url.find(':')
    if i > 0 and url[:i].replace('+', '').replace('-', '').replace('.', '').isalnum() and url[0].isalpha():
        s = url[:i].lower()
        url = url[i + 1:]
    if url.startswith('//'):
        rest = url[2:]
        end = len(rest)
        for c in '/?#':
            k = rest.find(c)
            if k >= 0 and k < end:
                end = k
        netloc = rest[:end]
        url = rest[end:]
    if allow_fragments and '#' in url:
        url, fragment = url.split('#', 1)
    if '?' in url:
        url, query = url.split('?', 1)
    return SplitResult(s, netloc, url, query, fragment)


def urlparse(url, scheme='', allow_fragments=True):
    r = urlsplit(url, scheme, allow_fragments)
    path = r.path
    params = ''
    if ';' in path.rsplit('/', 1)[-1]:
        i = path.rfind(';')
        path, params = path[:i], path[i + 1:]
    return ParseResult(r.scheme, r.netloc, path, params, r.query, r.fragment)


def urlunsplit(components):
    scheme, netloc, url, query, fragment = tuple(components)
    if netloc or (scheme and scheme in uses_netloc and url[:2] != '//'):
        if url and url[:1] != '/':
            url = '/' + url
        url = '//' + (netloc or '') + url
    if scheme:
        url = scheme + ':' + url
    if query:
        url = url + '?' + query
    if fragment:
        url = url + '#' + fragment
    return url


def urlunparse(components):
    scheme, netloc, url, params, query, fragment = tuple(components)
    if params:
        url = url + ';' + params
    return urlunsplit((scheme, netloc, url, query, fragment))


def urljoin(base, url, allow_fragments=True):
    if not base:
        return url
    if not url:
        return base
    u = urlsplit(url)
    if u.scheme:
        return url
    b = urlsplit(base)
    if url.startswith('//'):
        return b.scheme + ':' + url
    if u.netloc:
        return urlunsplit((b.scheme, u.netloc, u.path, u.query, u.fragment))
    if not u.path:
        return urlunsplit((b.scheme, b.netloc, b.path, u.query or b.query, u.fragment))
    if u.path.startswith('/'):
        path = u.path
    else:
        path = b.path[:b.path.rfind('/') + 1] + u.path
    segs = []
    for seg in path.split('/'):
        if seg == '..':
            if len(segs) > 1:
                segs.pop()
        elif seg != '.':
            segs.append(seg)
    if path.endswith('/.') or path.endswith('/..'):
        segs.append('')
    return urlunsplit((b.scheme, b.netloc, '/'.join(segs) or '/', u.query, u.fragment))


def urldefrag(url):
    if '#' in url:
        u, f = url.split('#', 1)
        return u, f
    return url, ''
