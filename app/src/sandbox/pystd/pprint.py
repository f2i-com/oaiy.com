"""Pretty printing of data structures (pure Python, for OAIY's sandbox)."""
import sys


def _fmt(obj, indent, width, depth, level, sort_dicts):
    text = repr(obj)
    if len(text) + level * indent <= width or depth is not None and level >= depth:
        return text
    pad = ' ' * ((level + 1) * indent)
    if isinstance(obj, dict) and obj:
        keys = sorted(obj.keys(), key=lambda k: repr(k)) if sort_dicts else list(obj.keys())
        items = [repr(k) + ': ' + _fmt(obj[k], indent, width, depth, level + 1, sort_dicts) for k in keys]
        return '{' + (',\n' + pad).join(items) + '}'
    if isinstance(obj, (list, tuple, set, frozenset)) and obj:
        if isinstance(obj, list):
            o, c = '[', ']'
        elif isinstance(obj, tuple):
            o, c = '(', ',)' if len(obj) == 1 else ')'
        else:
            o, c = '{', '}'
        seq = sorted(obj, key=lambda v: repr(v)) if isinstance(obj, (set, frozenset)) else obj
        items = [_fmt(v, indent, width, depth, level + 1, sort_dicts) for v in seq]
        return o + (',\n' + pad).join(items) + c
    return text


def pformat(obj, indent=1, width=80, depth=None, *, compact=False, sort_dicts=True, underscore_numbers=False):
    return _fmt(obj, indent, width, depth, 0, sort_dicts)


def pprint(obj, stream=None, indent=1, width=80, depth=None, *, compact=False, sort_dicts=True, underscore_numbers=False):
    (stream or sys.stdout).write(pformat(obj, indent, width, depth, sort_dicts=sort_dicts) + '\n')


def pp(obj, *args, sort_dicts=False, **kw):
    pprint(obj, *args, sort_dicts=sort_dicts, **kw)


def saferepr(obj):
    return repr(obj)


def isreadable(obj):
    return True


def isrecursive(obj):
    return False


class PrettyPrinter:
    def __init__(self, indent=1, width=80, depth=None, stream=None, *, compact=False, sort_dicts=True, underscore_numbers=False):
        self.indent = indent
        self.width = width
        self.depth = depth
        self.stream = stream
        self.sort_dicts = sort_dicts

    def pformat(self, obj):
        return pformat(obj, self.indent, self.width, self.depth, sort_dicts=self.sort_dicts)

    def pprint(self, obj):
        (self.stream or sys.stdout).write(self.pformat(obj) + '\n')
