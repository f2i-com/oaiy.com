"""Names for built-in types, and SimpleNamespace (pure Python, for OAIY's sandbox)."""


def _f():
    pass


def _g():
    yield 1


class _C:
    def _m(self):
        pass


FunctionType = type(_f)
LambdaType = type(lambda: None)
GeneratorType = type(_g())
MethodType = type(_C()._m)
BuiltinFunctionType = type(len)
BuiltinMethodType = type([].append)
NoneType = type(None)
EllipsisType = type(Ellipsis)
NotImplementedType = type(NotImplemented)


class SimpleNamespace:
    def __init__(self, mapping=None, **kwargs):
        if mapping:
            for k, v in dict(mapping).items():
                setattr(self, k, v)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        items = ['%s=%r' % (k, v) for k, v in self.__dict__.items()]
        return 'namespace(%s)' % ', '.join(items)

    def __eq__(self, other):
        return isinstance(other, SimpleNamespace) and self.__dict__ == other.__dict__


class MappingProxyType:
    def __init__(self, mapping):
        self._m = mapping

    def __getitem__(self, k):
        return self._m[k]

    def __iter__(self):
        return iter(self._m)

    def __len__(self):
        return len(self._m)

    def __contains__(self, k):
        return k in self._m

    def get(self, k, default=None):
        return self._m.get(k, default)

    def keys(self):
        return self._m.keys()

    def values(self):
        return self._m.values()

    def items(self):
        return self._m.items()

    def __repr__(self):
        return 'mappingproxy(%r)' % (self._m,)


def new_class(name, bases=(), kwds=None, exec_body=None):
    ns = {}
    if exec_body is not None:
        exec_body(ns)
    return type(name, tuple(bases) or (object,), ns)
