"""Calendar functions (pure Python, for OAIY's sandbox)."""
import datetime

MONDAY, TUESDAY, WEDNESDAY, THURSDAY, FRIDAY, SATURDAY, SUNDAY = 0, 1, 2, 3, 4, 5, 6
day_name = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
day_abbr = [d[:3] for d in day_name]
month_name = ['', 'January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December']
month_abbr = [''] + [m[:3] for m in month_name[1:]]
_first = [0]


class IllegalMonthError(ValueError):
    pass


def isleap(year):
    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)


def leapdays(y1, y2):
    y1 -= 1
    y2 -= 1
    return (y2 // 4 - y1 // 4) - (y2 // 100 - y1 // 100) + (y2 // 400 - y1 // 400)


def weekday(year, month, day):
    return datetime.date(year, month, day).weekday()


def monthrange(year, month):
    if not 1 <= month <= 12:
        raise IllegalMonthError('bad month number %r; must be 1-12' % month)
    days = [31, 29 if isleap(year) else 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31][month - 1]
    return weekday(year, month, 1), days


def setfirstweekday(firstweekday):
    _first[0] = firstweekday


def firstweekday():
    return _first[0]


def monthcalendar(year, month):
    start, days = monthrange(year, month)
    offset = (start - _first[0]) % 7
    weeks = []
    week = [0] * offset
    for d in range(1, days + 1):
        week.append(d)
        if len(week) == 7:
            weeks.append(week)
            week = []
    if week:
        weeks.append(week + [0] * (7 - len(week)))
    return weeks


def month(theyear, themonth, w=0, l=0):
    width = max(2, w)
    head = ' '.join([day_abbr[(i + _first[0]) % 7][:width] for i in range(7)])
    title = ('%s %d' % (month_name[themonth], theyear)).center(len(head)).rstrip()
    rows = [title, head]
    for week in monthcalendar(theyear, themonth):
        rows.append(' '.join([(str(d) if d else '').rjust(width) for d in week]).rstrip())
    return '\n'.join(rows) + '\n'


def prmonth(theyear, themonth, w=0, l=0):
    print(month(theyear, themonth, w, l), end='')


def timegm(tup):
    y, m, d, hh, mm, ss = tup[:6]
    days = datetime.date(y, m, d).toordinal() - datetime.date(1970, 1, 1).toordinal()
    return ((days * 24 + hh) * 60 + mm) * 60 + ss


class Calendar:
    def __init__(self, firstweekday=0):
        self.firstweekday = firstweekday

    def monthdayscalendar(self, year, month):
        saved = _first[0]
        _first[0] = self.firstweekday
        try:
            return monthcalendar(year, month)
        finally:
            _first[0] = saved

    def itermonthdates(self, year, month):
        for week in self.monthdayscalendar(year, month):
            for d in week:
                if d:
                    yield datetime.date(year, month, d)


class TextCalendar(Calendar):
    def formatmonth(self, theyear, themonth, w=0, l=0):
        saved = _first[0]
        _first[0] = self.firstweekday
        try:
            return month(theyear, themonth, w, l)
        finally:
            _first[0] = saved
