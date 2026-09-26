"""Dates and times (a pure-Python subset of CPython's datetime for bot.computer's sandbox)."""
import time as _time

MINYEAR = 1
MAXYEAR = 9999

_DAYS_IN_MONTH = [0, 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
_MONTH_NAMES = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
_DAY_NAMES = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']


def _is_leap(y):
    return y % 4 == 0 and (y % 100 != 0 or y % 400 == 0)


def _days_in_month(y, m):
    if m == 2 and _is_leap(y):
        return 29
    return _DAYS_IN_MONTH[m]


def _days_before_year(y):
    y -= 1
    return y * 365 + y // 4 - y // 100 + y // 400


def _days_before_month(y, m):
    n = 0
    for k in range(1, m):
        n += _days_in_month(y, k)
    return n


def _ymd2ord(y, m, d):
    return _days_before_year(y) + _days_before_month(y, m) + d


def _ord2ymd(n):
    # Proleptic Gregorian ordinal (1 = 0001-01-01) to (year, month, day).
    n -= 1
    n400, n = divmod(n, 146097)
    year = n400 * 400 + 1
    n100, n = divmod(n, 36524)
    n4, n = divmod(n, 1461)
    n1, n = divmod(n, 365)
    year += n100 * 100 + n4 * 4 + n1
    if n1 == 4 or n100 == 4:
        return year - 1, 12, 31
    month = 1
    while True:
        dim = _days_in_month(year, month)
        if n < dim:
            break
        n -= dim
        month += 1
    return year, month, n + 1


def _check_date(y, m, d):
    if not MINYEAR <= y <= MAXYEAR:
        raise ValueError('year %d is out of range' % y)
    if not 1 <= m <= 12:
        raise ValueError('month must be in 1..12')
    if not 1 <= d <= _days_in_month(y, m):
        raise ValueError('day is out of range for month')


def _pad(n, width=2):
    s = str(abs(int(n)))
    while len(s) < width:
        s = '0' + s
    return ('-' if n < 0 else '') + s


class timedelta:
    def __init__(self, days=0, seconds=0, microseconds=0, milliseconds=0, minutes=0, hours=0, weeks=0):
        total = (days + weeks * 7) * 86400.0 + seconds + minutes * 60 + hours * 3600 + milliseconds / 1000.0 + microseconds / 1e6
        us = int(round(total * 1e6))
        d, rem = divmod(us, 86400 * 1000000)
        s, u = divmod(rem, 1000000)
        self.days = d
        self.seconds = s
        self.microseconds = u

    def _us(self):
        return (self.days * 86400 + self.seconds) * 1000000 + self.microseconds

    def total_seconds(self):
        return self._us() / 1e6

    def __add__(self, other):
        if isinstance(other, timedelta):
            return timedelta(microseconds=self._us() + other._us())
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return timedelta(microseconds=self._us() - other._us())
        return NotImplemented

    def __neg__(self):
        return timedelta(microseconds=-self._us())

    def __abs__(self):
        return timedelta(microseconds=abs(self._us()))

    def __mul__(self, k):
        return timedelta(microseconds=self._us() * k)

    def __rmul__(self, k):
        return self.__mul__(k)

    def __truediv__(self, other):
        if isinstance(other, timedelta):
            return self._us() / other._us()
        return timedelta(microseconds=self._us() / other)

    def __floordiv__(self, other):
        if isinstance(other, timedelta):
            return self._us() // other._us()
        return timedelta(microseconds=self._us() // other)

    def __eq__(self, other):
        return isinstance(other, timedelta) and self._us() == other._us()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self._us() < other._us()

    def __le__(self, other):
        return self._us() <= other._us()

    def __gt__(self, other):
        return self._us() > other._us()

    def __ge__(self, other):
        return self._us() >= other._us()

    def __hash__(self):
        return hash(self._us())

    def __bool__(self):
        return self._us() != 0

    def __str__(self):
        h, rem = divmod(self.seconds, 3600)
        m, s = divmod(rem, 60)
        out = '%d:%s:%s' % (h, _pad(m), _pad(s))
        if self.microseconds:
            out += '.' + _pad(self.microseconds, 6)
        if self.days:
            out = '%d day%s, %s' % (self.days, '' if abs(self.days) == 1 else 's', out)
        return out

    def __repr__(self):
        parts = []
        if self.days:
            parts.append('days=%d' % self.days)
        if self.seconds:
            parts.append('seconds=%d' % self.seconds)
        if self.microseconds:
            parts.append('microseconds=%d' % self.microseconds)
        return 'datetime.timedelta(%s)' % ', '.join(parts)


timedelta.min = timedelta(days=-999999999)
timedelta.max = timedelta(days=999999999, hours=23, minutes=59, seconds=59, microseconds=999999)
timedelta.resolution = timedelta(microseconds=1)


class tzinfo:
    def utcoffset(self, dt):
        return None

    def tzname(self, dt):
        return None

    def dst(self, dt):
        return None


class timezone(tzinfo):
    def __init__(self, offset, name=None):
        self._offset = offset
        self._name = name

    def utcoffset(self, dt):
        return self._offset

    def tzname(self, dt):
        if self._name is not None:
            return self._name
        if self._offset._us() == 0:
            return 'UTC'
        total = int(self._offset.total_seconds())
        sign = '+' if total >= 0 else '-'
        total = abs(total)
        return 'UTC%s%s:%s' % (sign, _pad(total // 3600), _pad(total % 3600 // 60))

    def __repr__(self):
        return 'datetime.timezone.utc' if self._offset._us() == 0 else 'datetime.timezone(%r)' % self._offset


timezone.utc = timezone(timedelta(0))
UTC = timezone.utc


def _fmt(fmt, y, mo, d, h, mi, s, us, wd, yday, tz):
    out = ''
    i = 0
    while i < len(fmt):
        c = fmt[i]
        if c != '%' or i + 1 >= len(fmt):
            out += c
            i += 1
            continue
        k = fmt[i + 1]
        i += 2
        if k == 'Y':
            out += _pad(y, 4)
        elif k == 'y':
            out += _pad(y % 100)
        elif k == 'm':
            out += _pad(mo)
        elif k == 'd':
            out += _pad(d)
        elif k == 'e':
            out += str(d).rjust(2)
        elif k == 'H':
            out += _pad(h)
        elif k == 'I':
            out += _pad(12 if h % 12 == 0 else h % 12)
        elif k == 'M':
            out += _pad(mi)
        elif k == 'S':
            out += _pad(s)
        elif k == 'f':
            out += _pad(us, 6)
        elif k == 'p':
            out += 'AM' if h < 12 else 'PM'
        elif k == 'B':
            out += _MONTH_NAMES[mo - 1]
        elif k == 'b' or k == 'h':
            out += _MONTH_NAMES[mo - 1][:3]
        elif k == 'A':
            out += _DAY_NAMES[wd]
        elif k == 'a':
            out += _DAY_NAMES[wd][:3]
        elif k == 'w':
            out += str((wd + 1) % 7)
        elif k == 'u':
            out += str(wd + 1)
        elif k == 'j':
            out += _pad(yday, 3)
        elif k == 'Z':
            out += tz.tzname(None) if tz is not None else ''
        elif k == 'z':
            if tz is not None:
                total = int(tz.utcoffset(None).total_seconds())
                sign = '+' if total >= 0 else '-'
                total = abs(total)
                out += sign + _pad(total // 3600) + _pad(total % 3600 // 60)
        elif k == 'F':
            out += '%s-%s-%s' % (_pad(y, 4), _pad(mo), _pad(d))
        elif k == 'T':
            out += '%s:%s:%s' % (_pad(h), _pad(mi), _pad(s))
        elif k == 'D':
            out += '%s/%s/%s' % (_pad(mo), _pad(d), _pad(y % 100))
        elif k == 'c':
            out += '%s %s %s %s:%s:%s %d' % (_DAY_NAMES[wd][:3], _MONTH_NAMES[mo - 1][:3], str(d).rjust(2), _pad(h), _pad(mi), _pad(s), y)
        elif k == 'x':
            out += '%s/%s/%s' % (_pad(mo), _pad(d), _pad(y % 100))
        elif k == 'X':
            out += '%s:%s:%s' % (_pad(h), _pad(mi), _pad(s))
        elif k == '%':
            out += '%'
        else:
            out += '%' + k
    return out


class date:
    def __init__(self, year, month, day):
        _check_date(year, month, day)
        self.year = year
        self.month = month
        self.day = day

    @classmethod
    def today(cls):
        t = _time.localtime()
        return cls(t[0], t[1], t[2])

    @classmethod
    def fromordinal(cls, n):
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    @classmethod
    def fromtimestamp(cls, ts):
        d = datetime.fromtimestamp(ts)
        return cls(d.year, d.month, d.day)

    @classmethod
    def fromisoformat(cls, s):
        s = s.strip()
        if len(s) == 8 and s.isdigit():
            return cls(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        parts = s[:10].split('-')
        if len(parts) != 3:
            raise ValueError('Invalid isoformat string: %r' % s)
        return cls(int(parts[0]), int(parts[1]), int(parts[2]))

    def toordinal(self):
        return _ymd2ord(self.year, self.month, self.day)

    def weekday(self):
        return (self.toordinal() + 6) % 7

    def isoweekday(self):
        return self.weekday() + 1

    def isocalendar(self):
        y = self.year
        week1 = _iso_week1_monday(y)
        today = self.toordinal()
        week, day = divmod(today - week1, 7)
        if week < 0:
            y -= 1
            week1 = _iso_week1_monday(y)
            week, day = divmod(today - week1, 7)
        elif week >= 52 and today >= _iso_week1_monday(y + 1):
            y += 1
            week = 0
        return (y, week + 1, day + 1)

    def replace(self, year=None, month=None, day=None):
        return type(self)(self.year if year is None else year, self.month if month is None else month, self.day if day is None else day)

    def isoformat(self):
        return '%s-%s-%s' % (_pad(self.year, 4), _pad(self.month), _pad(self.day))

    def strftime(self, fmt):
        return _fmt(fmt, self.year, self.month, self.day, 0, 0, 0, 0, self.weekday(), self.toordinal() - _ymd2ord(self.year, 1, 1) + 1, None)

    def timetuple(self):
        return (self.year, self.month, self.day, 0, 0, 0, self.weekday(), self.toordinal() - _ymd2ord(self.year, 1, 1) + 1, -1)

    def ctime(self):
        return self.strftime('%c')

    def __format__(self, spec):
        return self.strftime(spec) if spec else str(self)

    def __str__(self):
        return self.isoformat()

    def __repr__(self):
        return 'datetime.date(%d, %d, %d)' % (self.year, self.month, self.day)

    def __add__(self, other):
        if isinstance(other, timedelta):
            return type(self).fromordinal(self.toordinal() + other.days)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return type(self).fromordinal(self.toordinal() - other.days)
        if isinstance(other, date):
            return timedelta(days=self.toordinal() - other.toordinal())
        return NotImplemented

    def _key(self):
        return (self.year, self.month, self.day)

    def __eq__(self, other):
        return isinstance(other, date) and not isinstance(other, datetime) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self._key() < other._key()

    def __le__(self, other):
        return self._key() <= other._key()

    def __gt__(self, other):
        return self._key() > other._key()

    def __ge__(self, other):
        return self._key() >= other._key()

    def __hash__(self):
        return hash(self._key())


def _iso_week1_monday(year):
    first = _ymd2ord(year, 1, 1)
    wd = (first + 6) % 7
    monday = first - wd
    if wd > 3:
        monday += 7
    return monday


class time:
    def __init__(self, hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
        if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59 or not 0 <= microsecond <= 999999:
            raise ValueError('time out of range')
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.tzinfo = tzinfo

    def isoformat(self, timespec='auto'):
        out = '%s:%s:%s' % (_pad(self.hour), _pad(self.minute), _pad(self.second))
        if self.microsecond and timespec == 'auto' or timespec == 'microseconds':
            out += '.' + _pad(self.microsecond, 6)
        elif timespec == 'milliseconds':
            out += '.' + _pad(self.microsecond // 1000, 3)
        elif timespec == 'minutes':
            out = out[:5]
        elif timespec == 'hours':
            out = out[:2]
        return out

    @classmethod
    def fromisoformat(cls, s):
        return datetime.fromisoformat('1970-01-01T' + s).time()

    def strftime(self, fmt):
        return _fmt(fmt, 1900, 1, 1, self.hour, self.minute, self.second, self.microsecond, 0, 1, self.tzinfo)

    def replace(self, hour=None, minute=None, second=None, microsecond=None):
        return time(self.hour if hour is None else hour, self.minute if minute is None else minute, self.second if second is None else second, self.microsecond if microsecond is None else microsecond, self.tzinfo)

    def _key(self):
        return (self.hour, self.minute, self.second, self.microsecond)

    def __eq__(self, other):
        return isinstance(other, time) and self._key() == other._key()

    def __lt__(self, other):
        return self._key() < other._key()

    def __le__(self, other):
        return self._key() <= other._key()

    def __gt__(self, other):
        return self._key() > other._key()

    def __ge__(self, other):
        return self._key() >= other._key()

    def __hash__(self):
        return hash(self._key())

    def __format__(self, spec):
        return self.strftime(spec) if spec else str(self)

    def __str__(self):
        return self.isoformat()

    def __repr__(self):
        return 'datetime.time(%d, %d, %d%s)' % (self.hour, self.minute, self.second, ', %d' % self.microsecond if self.microsecond else '')


class datetime(date):
    def __init__(self, year, month, day, hour=0, minute=0, second=0, microsecond=0, tzinfo=None):
        _check_date(year, month, day)
        if not 0 <= hour <= 23 or not 0 <= minute <= 59 or not 0 <= second <= 59 or not 0 <= microsecond <= 999999:
            raise ValueError('time out of range')
        self.year = year
        self.month = month
        self.day = day
        self.hour = hour
        self.minute = minute
        self.second = second
        self.microsecond = microsecond
        self.tzinfo = tzinfo

    @classmethod
    def now(cls, tz=None):
        if tz is not None:
            return cls.fromtimestamp(_time.time(), tz)
        t = _time.localtime()
        frac = _time.time() % 1
        return cls(t[0], t[1], t[2], t[3], t[4], t[5], int(frac * 1000000))

    @classmethod
    def today(cls):
        return cls.now()

    @classmethod
    def utcnow(cls):
        return cls._from_epoch(_time.time(), None)

    @classmethod
    def _from_epoch(cls, ts, tz):
        us = int(round(ts * 1000000))
        days, rem = divmod(us, 86400 * 1000000)
        y, m, d = _ord2ymd(_ymd2ord(1970, 1, 1) + days)
        secs, micro = divmod(rem, 1000000)
        return cls(y, m, d, secs // 3600, secs % 3600 // 60, secs % 60, micro, tz)

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        if tz is not None:
            base = cls._from_epoch(ts, None) + tz.utcoffset(None)
            return base.replace(tzinfo=tz)
        return cls._from_epoch(ts, None)

    @classmethod
    def utcfromtimestamp(cls, ts):
        return cls._from_epoch(ts, None)

    @classmethod
    def combine(cls, d, t, tzinfo=None):
        return cls(d.year, d.month, d.day, t.hour, t.minute, t.second, t.microsecond, tzinfo if tzinfo is not None else t.tzinfo)

    @classmethod
    def fromordinal(cls, n):
        y, m, d = _ord2ymd(n)
        return cls(y, m, d)

    @classmethod
    def fromisoformat(cls, s):
        s = s.strip()
        if len(s) < 10:
            raise ValueError('Invalid isoformat string: %r' % s)
        d = date.fromisoformat(s[:10])
        rest = s[10:]
        if not rest:
            return cls(d.year, d.month, d.day)
        rest = rest[1:]
        tz = None
        if rest.endswith('Z'):
            tz = timezone.utc
            rest = rest[:-1]
        else:
            for sign in ('+', '-'):
                at = rest.rfind(sign)
                if at > 0:
                    off = rest[at + 1:].replace(':', '')
                    mins = int(off[:2]) * 60 + (int(off[2:4]) if len(off) >= 4 else 0)
                    tz = timezone(timedelta(minutes=mins if sign == '+' else -mins))
                    rest = rest[:at]
                    break
        hms = rest.split('.')
        bits = hms[0].split(':')
        h = int(bits[0]) if bits and bits[0] else 0
        mi = int(bits[1]) if len(bits) > 1 else 0
        sec = int(bits[2]) if len(bits) > 2 else 0
        us = int((hms[1] + '000000')[:6]) if len(hms) > 1 else 0
        return cls(d.year, d.month, d.day, h, mi, sec, us, tz)

    @classmethod
    def strptime(cls, text, fmt):
        return _strptime(cls, text, fmt)

    def date(self):
        return date(self.year, self.month, self.day)

    def time(self):
        return time(self.hour, self.minute, self.second, self.microsecond)

    def timetz(self):
        return time(self.hour, self.minute, self.second, self.microsecond, self.tzinfo)

    def replace(self, year=None, month=None, day=None, hour=None, minute=None, second=None, microsecond=None, tzinfo=True):
        return datetime(
            self.year if year is None else year, self.month if month is None else month, self.day if day is None else day,
            self.hour if hour is None else hour, self.minute if minute is None else minute, self.second if second is None else second,
            self.microsecond if microsecond is None else microsecond, self.tzinfo if tzinfo is True else tzinfo)

    def utcoffset(self):
        return None if self.tzinfo is None else self.tzinfo.utcoffset(self)

    def tzname(self):
        return None if self.tzinfo is None else self.tzinfo.tzname(self)

    def astimezone(self, tz=None):
        if tz is None:
            return self.replace(tzinfo=None)
        if self.tzinfo is None:
            return self.replace(tzinfo=tz)
        utc = self - self.tzinfo.utcoffset(self)
        return (utc + tz.utcoffset(None)).replace(tzinfo=tz)

    def timestamp(self):
        base = self._us_since_epoch() / 1e6
        if self.tzinfo is not None:
            base -= self.tzinfo.utcoffset(self).total_seconds()
        return base

    def _us_since_epoch(self):
        days = self.toordinal() - _ymd2ord(1970, 1, 1)
        return ((days * 86400 + self.hour * 3600 + self.minute * 60 + self.second) * 1000000) + self.microsecond

    def isoformat(self, sep='T', timespec='auto'):
        out = date.isoformat(self) + sep + self.time().isoformat(timespec)
        if self.tzinfo is not None:
            total = int(self.tzinfo.utcoffset(self).total_seconds())
            sign = '+' if total >= 0 else '-'
            total = abs(total)
            out += '%s%s:%s' % (sign, _pad(total // 3600), _pad(total % 3600 // 60))
        return out

    def strftime(self, fmt):
        return _fmt(fmt, self.year, self.month, self.day, self.hour, self.minute, self.second, self.microsecond, self.weekday(), self.toordinal() - _ymd2ord(self.year, 1, 1) + 1, self.tzinfo)

    def timetuple(self):
        return (self.year, self.month, self.day, self.hour, self.minute, self.second, self.weekday(), self.toordinal() - _ymd2ord(self.year, 1, 1) + 1, -1)

    def __str__(self):
        return self.isoformat(' ')

    def __repr__(self):
        fields = [self.year, self.month, self.day, self.hour, self.minute]
        if self.second or self.microsecond:
            fields.append(self.second)
        if self.microsecond:
            fields.append(self.microsecond)
        out = 'datetime.datetime(%s' % ', '.join(str(f) for f in fields)
        if self.tzinfo is not None:
            out += ', tzinfo=%r' % self.tzinfo
        return out + ')'

    def __add__(self, other):
        if isinstance(other, timedelta):
            us = self._us_since_epoch() + other._us()
            days, rem = divmod(us, 86400 * 1000000)
            y, m, d = _ord2ymd(_ymd2ord(1970, 1, 1) + days)
            secs, micro = divmod(rem, 1000000)
            return datetime(y, m, d, secs // 3600, secs % 3600 // 60, secs % 60, micro, self.tzinfo)
        return NotImplemented

    def __radd__(self, other):
        return self.__add__(other)

    def __sub__(self, other):
        if isinstance(other, timedelta):
            return self.__add__(-other)
        if isinstance(other, datetime):
            a = self._us_since_epoch()
            b = other._us_since_epoch()
            if self.tzinfo is not None and other.tzinfo is not None:
                a -= self.tzinfo.utcoffset(self)._us()
                b -= other.tzinfo.utcoffset(other)._us()
            return timedelta(microseconds=a - b)
        return NotImplemented

    def _key(self):
        us = self._us_since_epoch()
        if self.tzinfo is not None:
            us -= self.tzinfo.utcoffset(self)._us()
        return us

    def __eq__(self, other):
        return isinstance(other, datetime) and self._key() == other._key()

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        return self._key() < other._key()

    def __le__(self, other):
        return self._key() <= other._key()

    def __gt__(self, other):
        return self._key() > other._key()

    def __ge__(self, other):
        return self._key() >= other._key()

    def __hash__(self):
        return hash(self._key())


def _strptime(cls, text, fmt):
    vals = {'Y': 1900, 'm': 1, 'd': 1, 'H': 0, 'M': 0, 'S': 0, 'f': 0}
    pm = None
    tz = None
    i = 0
    j = 0

    def take_digits(maxlen):
        start = j_ref[0]
        k = start
        while k < len(text) and k - start < maxlen and text[k].isdigit():
            k += 1
        if k == start:
            raise ValueError("time data %r does not match format %r" % (text, fmt))
        j_ref[0] = k
        return int(text[start:k])

    j_ref = [0]
    while i < len(fmt):
        c = fmt[i]
        if c == '%' and i + 1 < len(fmt):
            k = fmt[i + 1]
            i += 2
            if k in ('Y',):
                vals['Y'] = take_digits(4)
            elif k == 'y':
                y = take_digits(2)
                vals['Y'] = 2000 + y if y < 69 else 1900 + y
            elif k in ('m', 'd', 'H', 'M', 'S'):
                vals[k] = take_digits(2)
            elif k == 'I':
                vals['H'] = take_digits(2)
            elif k == 'f':
                start = j_ref[0]
                n = take_digits(6)
                digits = j_ref[0] - start
                vals['f'] = n * (10 ** (6 - digits))
            elif k == 'j':
                vals['j'] = take_digits(3)
            elif k in ('b', 'B', 'h'):
                rest = text[j_ref[0]:].lower()
                found = False
                for n, name in enumerate(_MONTH_NAMES):
                    for cand in (name.lower(), name[:3].lower()):
                        if rest.startswith(cand):
                            vals['m'] = n + 1
                            j_ref[0] += len(cand)
                            found = True
                            break
                    if found:
                        break
                if not found:
                    raise ValueError("time data %r does not match format %r" % (text, fmt))
            elif k in ('a', 'A'):
                rest = text[j_ref[0]:].lower()
                for name in _DAY_NAMES:
                    for cand in (name.lower(), name[:3].lower()):
                        if rest.startswith(cand):
                            j_ref[0] += len(cand)
                            rest = ''
                            break
                    if rest == '':
                        break
            elif k == 'p':
                word = text[j_ref[0]:j_ref[0] + 2].upper()
                pm = word == 'PM'
                j_ref[0] += 2
            elif k == 'z':
                s = text[j_ref[0]:]
                if s.startswith('Z'):
                    tz = timezone.utc
                    j_ref[0] += 1
                else:
                    sign = 1 if s[0] == '+' else -1
                    digits = s[1:6].replace(':', '')
                    mins = int(digits[:2]) * 60 + int(digits[2:4])
                    tz = timezone(timedelta(minutes=sign * mins))
                    j_ref[0] += 6 if s[3:4] == ':' else 5
            elif k == '%':
                j_ref[0] += 1
            else:
                raise ValueError("'%s' is a bad directive in format %r" % (k, fmt))
        else:
            if j_ref[0] >= len(text) or text[j_ref[0]] != c:
                if c == ' ' and j_ref[0] < len(text) and text[j_ref[0]].isspace():
                    pass
                else:
                    raise ValueError("time data %r does not match format %r" % (text, fmt))
            j_ref[0] += 1
            i += 1
    if j_ref[0] != len(text):
        raise ValueError('unconverted data remains: %s' % text[j_ref[0]:])
    h = vals['H']
    if pm is not None:
        h = h % 12 + (12 if pm else 0)
    if 'j' in vals:
        d = date.fromordinal(_ymd2ord(vals['Y'], 1, 1) + vals['j'] - 1)
        vals['m'] = d.month
        vals['d'] = d.day
    return cls(vals['Y'], vals['m'], vals['d'], h, vals['M'], vals['S'], vals['f'], tz)


datetime.min = datetime(1, 1, 1)
datetime.max = datetime(9999, 12, 31, 23, 59, 59, 999999)
date.min = date(1, 1, 1)
date.max = date(9999, 12, 31)
