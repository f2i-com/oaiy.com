"""Logging (a pure-Python subset of CPython's logging, for bot.computer's sandbox)."""
import sys
import time

CRITICAL = 50
FATAL = CRITICAL
ERROR = 40
WARNING = 30
WARN = WARNING
INFO = 20
DEBUG = 10
NOTSET = 0

_names = {CRITICAL: 'CRITICAL', ERROR: 'ERROR', WARNING: 'WARNING', INFO: 'INFO', DEBUG: 'DEBUG', NOTSET: 'NOTSET'}
_levels = {'CRITICAL': CRITICAL, 'FATAL': FATAL, 'ERROR': ERROR, 'WARN': WARNING, 'WARNING': WARNING, 'INFO': INFO, 'DEBUG': DEBUG, 'NOTSET': NOTSET}
BASIC_FORMAT = '%(levelname)s:%(name)s:%(message)s'


def getLevelName(level):
    if isinstance(level, str):
        return _levels.get(level, 'Level %s' % level)
    return _names.get(level, 'Level %s' % level)


def addLevelName(level, name):
    _names[level] = name
    _levels[name] = level


def _level(v):
    if isinstance(v, str):
        return _levels[v.upper()]
    return v


class LogRecord:
    def __init__(self, name, level, msg, args, exc_info=None):
        self.name = name
        self.levelno = level
        self.levelname = getLevelName(level)
        self.msg = msg
        self.args = args
        self.exc_info = exc_info
        self.created = time.time()
        self.msecs = int((self.created % 1) * 1000)
        self.filename = 'main.py'
        self.module = 'main'
        self.funcName = ''
        self.lineno = 0
        self.process = 1
        self.thread = 1
        self.threadName = 'MainThread'

    def getMessage(self):
        msg = str(self.msg)
        if self.args:
            if len(self.args) == 1 and isinstance(self.args[0], dict):
                msg = msg % self.args[0]
            else:
                msg = msg % tuple(self.args)
        return msg


class Formatter:
    def __init__(self, fmt=None, datefmt=None, style='%'):
        self._fmt = fmt or '%(message)s'
        self.datefmt = datefmt
        self.style = style

    def formatTime(self, record, datefmt=None):
        t = time.localtime(record.created)
        if datefmt:
            return time.strftime(datefmt)
        return '%04d-%02d-%02d %02d:%02d:%02d,%03d' % (t[0], t[1], t[2], t[3], t[4], t[5], record.msecs)

    def format(self, record):
        record.message = record.getMessage()
        values = dict(record.__dict__)
        values['asctime'] = self.formatTime(record, self.datefmt)
        if self.style == '{':
            out = self._fmt.format(**values)
        elif self.style == '$':
            out = self._fmt
            for k, v in values.items():
                out = out.replace('${' + k + '}', str(v)).replace('$' + k, str(v))
        else:
            out = self._fmt % values
        if record.exc_info:
            import traceback
            exc = record.exc_info
            if exc is True:
                exc = sys.exc_info()
            if exc and exc[1] is not None:
                out += '\n' + ''.join(traceback.format_exception(exc[0], exc[1], exc[2])).rstrip('\n')
        return out


class Filter:
    def __init__(self, name=''):
        self.name = name

    def filter(self, record):
        return not self.name or record.name == self.name or record.name.startswith(self.name + '.')


class Handler:
    def __init__(self, level=NOTSET):
        self.level = _level(level)
        self.formatter = None
        self.filters = []

    def setLevel(self, level):
        self.level = _level(level)

    def setFormatter(self, fmt):
        self.formatter = fmt

    def addFilter(self, f):
        self.filters.append(f)

    def removeFilter(self, f):
        if f in self.filters:
            self.filters.remove(f)

    def format(self, record):
        return (self.formatter or Formatter()).format(record)

    def handle(self, record):
        for f in self.filters:
            ok = f.filter(record) if hasattr(f, 'filter') else f(record)
            if not ok:
                return False
        self.emit(record)
        return True

    def emit(self, record):
        pass

    def flush(self):
        pass

    def close(self):
        pass


class StreamHandler(Handler):
    terminator = '\n'

    def __init__(self, stream=None):
        Handler.__init__(self)
        self.stream = stream

    def emit(self, record):
        stream = self.stream if self.stream is not None else sys.stderr
        stream.write(self.format(record) + self.terminator)


class FileHandler(StreamHandler):
    def __init__(self, filename, mode='a', encoding=None, delay=False):
        StreamHandler.__init__(self)
        self.baseFilename = filename
        self.mode = mode
        if mode == 'w':
            open(filename, 'w').close()

    def emit(self, record):
        with open(self.baseFilename, 'a') as f:
            f.write(self.format(record) + self.terminator)


class NullHandler(Handler):
    pass


class Logger:
    def __init__(self, name, level=NOTSET):
        self.name = name
        self.level = _level(level)
        self.handlers = []
        self.parent = None
        self.propagate = True
        self.disabled = False
        self.filters = []

    def setLevel(self, level):
        self.level = _level(level)

    def getEffectiveLevel(self):
        logger = self
        while logger is not None:
            if logger.level:
                return logger.level
            logger = logger.parent
        return NOTSET

    def isEnabledFor(self, level):
        return level >= self.getEffectiveLevel() and not _disabled[0] >= level

    def addHandler(self, h):
        if h not in self.handlers:
            self.handlers.append(h)

    def removeHandler(self, h):
        if h in self.handlers:
            self.handlers.remove(h)

    def hasHandlers(self):
        logger = self
        while logger is not None:
            if logger.handlers:
                return True
            logger = logger.parent if logger.propagate else None
        return False

    def addFilter(self, f):
        self.filters.append(f)

    def getChild(self, suffix):
        return getLogger(self.name + '.' + suffix if self.name != 'root' else suffix)

    def _log(self, level, msg, args, exc_info=None, extra=None, stack_info=False, stacklevel=1):
        if self.disabled or not self.isEnabledFor(level):
            return
        record = LogRecord(self.name, level, msg, args, exc_info)
        if extra:
            for k, v in extra.items():
                setattr(record, k, v)
        for f in self.filters:
            if not (f.filter(record) if hasattr(f, 'filter') else f(record)):
                return
        logger = self
        handled = False
        while logger is not None:
            for h in logger.handlers:
                handled = True
                if record.levelno >= h.level:
                    h.handle(record)
            logger = logger.parent if logger.propagate else None
        if not handled and level >= WARNING:
            sys.stderr.write(record.getMessage() + '\n')

    def debug(self, msg, *args, **kw):
        self._log(DEBUG, msg, args, **kw)

    def info(self, msg, *args, **kw):
        self._log(INFO, msg, args, **kw)

    def warning(self, msg, *args, **kw):
        self._log(WARNING, msg, args, **kw)

    warn = warning

    def error(self, msg, *args, **kw):
        self._log(ERROR, msg, args, **kw)

    def exception(self, msg, *args, exc_info=True, **kw):
        self._log(ERROR, msg, args, exc_info=exc_info, **kw)

    def critical(self, msg, *args, **kw):
        self._log(CRITICAL, msg, args, **kw)

    fatal = critical

    def log(self, level, msg, *args, **kw):
        self._log(_level(level), msg, args, **kw)


root = Logger('root', WARNING)
_loggers = {}
_disabled = [NOTSET - 1]


def getLogger(name=None):
    if not name or name == 'root':
        return root
    if name in _loggers:
        return _loggers[name]
    logger = Logger(name)
    _loggers[name] = logger
    parent_name = name.rsplit('.', 1)[0] if '.' in name else None
    logger.parent = getLogger(parent_name) if parent_name else root
    return logger


def basicConfig(**kw):
    if root.handlers and not kw.get('force'):
        return
    if kw.get('force'):
        root.handlers = []
    if 'filename' in kw:
        h = FileHandler(kw['filename'], kw.get('filemode', 'a'))
    else:
        h = StreamHandler(kw.get('stream'))
    for handler in kw.get('handlers', []) or []:
        root.addHandler(handler)
    if not kw.get('handlers'):
        h.setFormatter(Formatter(kw.get('format', BASIC_FORMAT), kw.get('datefmt'), kw.get('style', '%')))
        root.addHandler(h)
    if 'level' in kw:
        root.setLevel(kw['level'])


def disable(level=CRITICAL):
    _disabled[0] = level


def _root_call(level):
    def call(msg, *args, **kw):
        if not root.handlers:
            basicConfig()
        root._log(level, msg, args, **kw)
    return call


debug = _root_call(DEBUG)
info = _root_call(INFO)
warning = _root_call(WARNING)
warn = warning
error = _root_call(ERROR)
critical = _root_call(CRITICAL)
fatal = critical


def exception(msg, *args, **kw):
    if not root.handlers:
        basicConfig()
    root._log(ERROR, msg, args, exc_info=True)


def log(level, msg, *args, **kw):
    if not root.handlers:
        basicConfig()
    root._log(_level(level), msg, args, **kw)


def shutdown():
    pass


class LoggerAdapter:
    def __init__(self, logger, extra=None):
        self.logger = logger
        self.extra = extra or {}

    def debug(self, msg, *a, **k):
        self.logger.debug(msg, *a, **k)

    def info(self, msg, *a, **k):
        self.logger.info(msg, *a, **k)

    def warning(self, msg, *a, **k):
        self.logger.warning(msg, *a, **k)

    def error(self, msg, *a, **k):
        self.logger.error(msg, *a, **k)

    def critical(self, msg, *a, **k):
        self.logger.critical(msg, *a, **k)
