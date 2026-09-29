"""Temporary files and folders, under /tmp in the project (pure Python, for OAIY's sandbox)."""
import os
import random

tempdir = None
template = 'tmp'
_chars = 'abcdefghijklmnopqrstuvwxyz0123456789_'


def gettempdir():
    d = tempdir or '/tmp'
    os.makedirs(d, exist_ok=True)
    return d


def gettempprefix():
    return template


def _name(prefix, suffix):
    return (prefix if prefix is not None else template) + ''.join([random.choice(_chars) for _ in range(8)]) + (suffix or '')


def mkdtemp(suffix=None, prefix=None, dir=None):
    base = dir or gettempdir()
    while True:
        p = os.path.join(base, _name(prefix, suffix))
        if not os.path.exists(p):
            os.makedirs(p, exist_ok=True)
            return p


def mkstemp(suffix=None, prefix=None, dir=None, text=False):
    base = dir or gettempdir()
    while True:
        p = os.path.join(base, _name(prefix, suffix))
        if not os.path.exists(p):
            open(p, 'w').close()
            return (-1, p)


def mktemp(suffix='', prefix=template, dir=None):
    return os.path.join(dir or gettempdir(), _name(prefix, suffix))


class _TemporaryFile:
    def __init__(self, mode='w+b', suffix=None, prefix=None, dir=None, delete=True, encoding=None):
        self.name = mkstemp(suffix, prefix, dir)[1]
        self.delete = delete
        self.mode = mode
        self._f = open(self.name, mode if 'x' not in mode else mode.replace('x', 'w'))

    def write(self, data):
        return self._f.write(data)

    def read(self, *a):
        return self._f.read(*a)

    def readline(self):
        return self._f.readline()

    def seek(self, pos, whence=0):
        self._f.close()
        self._f = open(self.name, 'r+b' if 'b' in self.mode else 'r+')
        return self._f.seek(pos)

    def flush(self):
        self._f.flush()

    def close(self):
        try:
            self._f.close()
        finally:
            if self.delete and os.path.exists(self.name):
                os.remove(self.name)

    def __iter__(self):
        return iter(self._f)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False


def NamedTemporaryFile(mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, delete=True, **kw):
    return _TemporaryFile(mode, suffix, prefix, dir, delete, encoding)


def TemporaryFile(mode='w+b', buffering=-1, encoding=None, newline=None, suffix=None, prefix=None, dir=None, **kw):
    return _TemporaryFile(mode, suffix, prefix, dir, True, encoding)


class TemporaryDirectory:
    def __init__(self, suffix=None, prefix=None, dir=None, ignore_cleanup_errors=False, delete=True):
        self.name = mkdtemp(suffix, prefix, dir)
        self.delete = delete

    def cleanup(self):
        import shutil
        if os.path.isdir(self.name):
            shutil.rmtree(self.name, ignore_errors=True)

    def __enter__(self):
        return self.name

    def __exit__(self, *exc):
        if self.delete:
            self.cleanup()
        return False
