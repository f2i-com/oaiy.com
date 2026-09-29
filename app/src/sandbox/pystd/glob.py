"""Filename pattern expansion (pure Python, for OAIY's sandbox)."""
import os
import re
import fnmatch

_magic = re.compile('([*?[])')


def has_magic(s):
    return _magic.search(s) is not None


def escape(pathname):
    out = ''
    for c in pathname:
        out += '[' + c + ']' if c in '*?[' else c
    return out


def _listdir(d):
    try:
        return os.listdir(d if d else '.')
    except OSError:
        return []


def _isdir(p):
    return os.path.isdir(p) if p else True


def _join(a, b):
    if not a:
        return b
    return a + b if a.endswith('/') else a + '/' + b


def _walk_all(base, hidden):
    out = []
    for name in _listdir(base):
        if not hidden and name.startswith('.'):
            continue
        p = _join(base, name)
        if _isdir(p):
            out.append(p)
            out.extend(_walk_all(p, hidden))
    return out


def glob(pathname, *, root_dir=None, recursive=False, include_hidden=False):
    return list(iglob(pathname, root_dir=root_dir, recursive=recursive, include_hidden=include_hidden))


def iglob(pathname, *, root_dir=None, recursive=False, include_hidden=False):
    prefix = ''
    if root_dir:
        prefix = root_dir.rstrip('/') + '/'
    absolute = pathname.startswith('/')
    parts = [p for p in pathname.split('/') if p != '']
    results = ['/' if absolute else '']
    for idx, part in enumerate(parts):
        last = idx == len(parts) - 1
        nxt = []
        for base in results:
            full = prefix + base if not absolute else base
            if recursive and part == '**':
                if not last:
                    nxt.append(base)
                    for d in _walk_all(full, include_hidden):
                        nxt.append(d[len(prefix):] if prefix and d.startswith(prefix) else d)
                else:
                    for name in _listdir(full):
                        if not include_hidden and name.startswith('.'):
                            continue
                        nxt.append(_join(base, name))
                    for d in _walk_all(full, include_hidden):
                        rel = d[len(prefix):] if prefix and d.startswith(prefix) else d
                        for name in _listdir(d):
                            if not include_hidden and name.startswith('.'):
                                continue
                            nxt.append(_join(rel, name))
                continue
            if not has_magic(part):
                cand = _join(base, part)
                if os.path.exists(prefix + cand if not absolute else cand) or (not last and _isdir(prefix + cand)):
                    nxt.append(cand)
                continue
            for name in _listdir(full):
                if name.startswith('.') and not part.startswith('.') and not include_hidden:
                    continue
                if fnmatch.fnmatchcase(name, part):
                    cand = _join(base, name)
                    if last or _isdir(prefix + cand if not absolute else cand):
                        nxt.append(cand)
        seen = []
        for r in nxt:
            if r not in seen:
                seen.append(r)
        results = seen
    for r in results:
        if r:
            yield r
