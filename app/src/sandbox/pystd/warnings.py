"""Warnings (pure Python, for bot.computer's sandbox): printed to stderr, filterable."""
import sys

filters = []
_seen = []


def _category_name(category):
    return getattr(category, '__name__', 'UserWarning')


def _action(message, category):
    for action, text, cat in filters:
        if text and text not in str(message):
            continue
        if cat is not None and not issubclass(category, cat):
            continue
        return action
    return 'default'


def formatwarning(message, category, filename, lineno, line=None):
    return '%s:%s: %s: %s\n' % (filename, lineno, _category_name(category), message)


def showwarning(message, category, filename, lineno, file=None, line=None):
    (file or sys.stderr).write(formatwarning(message, category, filename, lineno, line))


def warn(message, category=None, stacklevel=1, source=None, **kw):
    if isinstance(message, Warning):
        category = type(message)
    if category is None:
        category = UserWarning
    action = _action(message, category)
    if action == 'ignore':
        return
    if action == 'error':
        raise category(message)
    key = (str(message), _category_name(category))
    if action in ('default', 'once', 'module') and key in _seen:
        return
    _seen.append(key)
    showwarning(message, category, 'main.py', 0)


def warn_explicit(message, category, filename, lineno, module=None, registry=None, module_globals=None, source=None):
    warn(message, category)


def filterwarnings(action, message='', category=None, module='', lineno=0, append=False):
    entry = (action, message, category)
    if append:
        filters.append(entry)
    else:
        filters.insert(0, entry)


def simplefilter(action, category=None, lineno=0, append=False):
    filterwarnings(action, '', category, append=append)


def resetwarnings():
    del filters[:]


class catch_warnings:
    def __init__(self, *, record=False, module=None, action=None, category=Warning, lineno=0, append=False):
        self._record = record
        self._action = action
        self._category = category

    def __enter__(self):
        self._saved = list(filters)
        self._saved_show = globals()['showwarning']
        if self._action:
            simplefilter(self._action, self._category)
        if self._record:
            log = []

            def record(message, category, filename, lineno, file=None, line=None):
                log.append(WarningMessage(message, category, filename, lineno))
            globals()['showwarning'] = record
            return log
        return None

    def __exit__(self, *exc):
        del filters[:]
        filters.extend(self._saved)
        globals()['showwarning'] = self._saved_show
        return False


class WarningMessage:
    def __init__(self, message, category, filename, lineno, file=None, line=None):
        self.message = message
        self.category = category
        self.filename = filename
        self.lineno = lineno


def deprecated(msg, *, category=DeprecationWarning, stacklevel=1):
    def wrap(f):
        def inner(*a, **k):
            warn(msg, category)
            return f(*a, **k)
        return inner
    return wrap
