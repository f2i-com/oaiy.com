"""Threads in bot.computer's sandbox: there is one thread, so start() runs the target to completion. Locks and events work as they would uncontended."""


class Lock:
    def __init__(self):
        self._locked = False

    def acquire(self, blocking=True, timeout=-1):
        if self._locked and not blocking:
            return False
        self._locked = True
        return True

    def release(self):
        if not self._locked:
            raise RuntimeError('release unlocked lock')
        self._locked = False

    def locked(self):
        return self._locked

    def __enter__(self):
        self.acquire()
        return True

    def __exit__(self, *exc):
        self.release()
        return False


class RLock(Lock):
    def __init__(self):
        Lock.__init__(self)
        self._count = 0

    def acquire(self, blocking=True, timeout=-1):
        self._count += 1
        self._locked = True
        return True

    def release(self):
        if self._count <= 0:
            raise RuntimeError('cannot release un-acquired lock')
        self._count -= 1
        self._locked = self._count > 0


class Event:
    def __init__(self):
        self._flag = False

    def is_set(self):
        return self._flag

    def set(self):
        self._flag = True

    def clear(self):
        self._flag = False

    def wait(self, timeout=None):
        return self._flag


class Condition:
    def __init__(self, lock=None):
        self._lock = lock or RLock()

    def __enter__(self):
        return self._lock.__enter__()

    def __exit__(self, *exc):
        return self._lock.__exit__(*exc)

    def wait(self, timeout=None):
        return True

    def wait_for(self, predicate, timeout=None):
        return predicate()

    def notify(self, n=1):
        pass

    def notify_all(self):
        pass


class Semaphore:
    def __init__(self, value=1):
        self._value = value

    def acquire(self, blocking=True, timeout=None):
        if self._value <= 0:
            return False
        self._value -= 1
        return True

    def release(self, n=1):
        self._value += n

    def __enter__(self):
        self.acquire()
        return True

    def __exit__(self, *exc):
        self.release()
        return False


BoundedSemaphore = Semaphore
_ids = [0]


class Thread:
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        _ids[0] += 1
        self._target = target
        self._args = args
        self._kwargs = kwargs or {}
        self.name = name or 'Thread-%d' % _ids[0]
        self.daemon = bool(daemon)
        self.ident = _ids[0]
        self._started = False
        self._done = False

    def start(self):
        if self._started:
            raise RuntimeError('threads can only be started once')
        self._started = True
        self.run()
        self._done = True

    def run(self):
        if self._target is not None:
            self._target(*self._args, **self._kwargs)

    def join(self, timeout=None):
        return None

    def is_alive(self):
        return self._started and not self._done


class Timer(Thread):
    def __init__(self, interval, function, args=None, kwargs=None):
        Thread.__init__(self, target=function, args=args or (), kwargs=kwargs or {})
        self.interval = interval

    def cancel(self):
        self._target = None


class local:
    pass


_main = Thread(name='MainThread')
_main._started = True


def current_thread():
    return _main


def main_thread():
    return _main


def active_count():
    return 1


def enumerate():
    return [_main]


def get_ident():
    return 1


def get_native_id():
    return 1
