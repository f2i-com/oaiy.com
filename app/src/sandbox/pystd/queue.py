"""Queues (pure Python, single-threaded, for OAIY's sandbox)."""
import heapq


class Empty(Exception):
    pass


class Full(Exception):
    pass


class ShutDown(Exception):
    pass


class Queue:
    def __init__(self, maxsize=0):
        self.maxsize = maxsize
        self._items = []
        self.unfinished_tasks = 0

    def _put(self, item):
        self._items.append(item)

    def _get(self):
        return self._items.pop(0)

    def qsize(self):
        return len(self._items)

    def empty(self):
        return not self._items

    def full(self):
        return self.maxsize > 0 and len(self._items) >= self.maxsize

    def put(self, item, block=True, timeout=None):
        if self.full():
            raise Full
        self._put(item)
        self.unfinished_tasks += 1

    def put_nowait(self, item):
        self.put(item, False)

    def get(self, block=True, timeout=None):
        if not self._items:
            raise Empty
        return self._get()

    def get_nowait(self):
        return self.get(False)

    def task_done(self):
        if self.unfinished_tasks <= 0:
            raise ValueError('task_done() called too many times')
        self.unfinished_tasks -= 1

    def join(self):
        return None


class LifoQueue(Queue):
    def _get(self):
        return self._items.pop()


class PriorityQueue(Queue):
    def _put(self, item):
        heapq.heappush(self._items, item)

    def _get(self):
        return heapq.heappop(self._items)


class SimpleQueue(Queue):
    pass
