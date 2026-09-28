"""Shell-like syntax: split, quote and join (pure Python, for bot.computer's sandbox)."""
import re

_unsafe = re.compile(r'[^\w@%+=:,./-]')


def quote(s):
    if not s:
        return "''"
    if _unsafe.search(s) is None:
        return s
    return "'" + s.replace("'", "'\"'\"'") + "'"


def join(split_command):
    return ' '.join([quote(arg) for arg in split_command])


def split(s, comments=False, posix=True):
    out = []
    cur = ''
    have = False
    i = 0
    n = len(s)
    while i < n:
        c = s[i]
        if c.isspace():
            if have:
                out.append(cur)
                cur = ''
                have = False
            i += 1
            continue
        if comments and c == '#' and not have:
            while i < n and s[i] != '\n':
                i += 1
            continue
        if c == "'":
            end = s.find("'", i + 1)
            if end < 0:
                raise ValueError('No closing quotation')
            cur += s[i + 1:end] if posix else s[i:end + 1]
            have = True
            i = end + 1
            continue
        if c == '"':
            i += 1
            if not posix:
                cur += '"'
            while i < n and s[i] != '"':
                if s[i] == '\\' and i + 1 < n and s[i + 1] in '"\\$`\n':
                    cur += s[i + 1]
                    i += 2
                    continue
                cur += s[i]
                i += 1
            if i >= n:
                raise ValueError('No closing quotation')
            if not posix:
                cur += '"'
            i += 1
            have = True
            continue
        if c == '\\' and posix:
            if i + 1 < n:
                cur += s[i + 1]
            i += 2
            have = True
            continue
        cur += c
        have = True
        i += 1
    if have:
        out.append(cur)
    return out


class shlex:
    def __init__(self, instream=None, posix=False, punctuation_chars=False):
        text = instream if isinstance(instream, str) else (instream.read() if instream is not None else '')
        self._tokens = split(text, comments=True, posix=posix)
        self.whitespace_split = True

    def __iter__(self):
        return iter(self._tokens)

    def get_token(self):
        return self._tokens.pop(0) if self._tokens else None
