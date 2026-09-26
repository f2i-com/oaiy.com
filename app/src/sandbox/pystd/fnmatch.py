"""Shell-style filename matching (pure Python, for bot.computer's sandbox)."""
import re

_cache = {}


def translate(pat):
    i = 0
    n = len(pat)
    res = ''
    while i < n:
        c = pat[i]
        i += 1
        if c == '*':
            res += '.*'
        elif c == '?':
            res += '.'
        elif c == '[':
            j = i
            if j < n and pat[j] == '!':
                j += 1
            if j < n and pat[j] == ']':
                j += 1
            while j < n and pat[j] != ']':
                j += 1
            if j >= n:
                res += '\\['
            else:
                stuff = pat[i:j].replace('\\', '\\\\')
                i = j + 1
                if stuff.startswith('!'):
                    stuff = '^' + stuff[1:]
                elif stuff.startswith('^'):
                    stuff = '\\' + stuff
                res += '[' + stuff + ']'
        else:
            res += re.escape(c)
    return '(?s:' + res + ')\\Z'


def _compiled(pat):
    r = _cache.get(pat)
    if r is None:
        r = re.compile(translate(pat))
        _cache[pat] = r
    return r


def fnmatchcase(name, pat):
    return _compiled(pat).match(name) is not None


def fnmatch(name, pat):
    return fnmatchcase(name, pat)


def filter(names, pat):
    r = _compiled(pat)
    return [n for n in names if r.match(n) is not None]
