"""CSV files (a pure-Python csv module for bot.computer's sandbox)."""

QUOTE_MINIMAL = 0
QUOTE_ALL = 1
QUOTE_NONNUMERIC = 2
QUOTE_NONE = 3

__version__ = '1.0'


class Error(Exception):
    pass


class Dialect:
    delimiter = ','
    quotechar = '"'
    escapechar = None
    doublequote = True
    skipinitialspace = False
    lineterminator = '\r\n'
    quoting = QUOTE_MINIMAL
    strict = False


class excel(Dialect):
    pass


class excel_tab(excel):
    delimiter = '\t'


class unix_dialect(Dialect):
    lineterminator = '\n'
    quoting = QUOTE_ALL


_dialects = {'excel': excel, 'excel-tab': excel_tab, 'unix': unix_dialect}
_limit = [131072]


def register_dialect(name, dialect=None, **kw):
    base = dialect if dialect is not None else Dialect
    d = _Settings(base, kw)
    _dialects[name] = d


def unregister_dialect(name):
    if name not in _dialects:
        raise Error('unknown dialect')
    del _dialects[name]


def get_dialect(name):
    if name not in _dialects:
        raise Error('unknown dialect')
    return _dialects[name]


def list_dialects():
    return list(_dialects.keys())


def field_size_limit(new_limit=None):
    old = _limit[0]
    if new_limit is not None:
        _limit[0] = new_limit
    return old


class _Settings:
    def __init__(self, dialect, kw):
        if isinstance(dialect, str):
            dialect = get_dialect(dialect)
        for name in ('delimiter', 'quotechar', 'escapechar', 'doublequote', 'skipinitialspace', 'lineterminator', 'quoting', 'strict'):
            value = kw[name] if name in kw else getattr(dialect, name, getattr(Dialect, name))
            setattr(self, name, value)


def _lines(source):
    if isinstance(source, str):
        return iter(source.splitlines(True))
    return iter(source)


class reader:
    def __init__(self, csvfile, dialect='excel', **kw):
        self._src = _lines(csvfile)
        self.dialect = _Settings(dialect, kw)
        self.line_num = 0

    def __iter__(self):
        return self

    def __next__(self):
        d = self.dialect
        row = []
        field = ''
        quoted = False
        in_quotes = False
        started = False
        while True:
            line = next(self._src)
            self.line_num += 1
            i = 0
            n = len(line)
            while i < n:
                c = line[i]
                if in_quotes:
                    if d.escapechar is not None and c == d.escapechar and i + 1 < n:
                        field += line[i + 1]
                        i += 2
                        continue
                    if c == d.quotechar:
                        if d.doublequote and i + 1 < n and line[i + 1] == d.quotechar:
                            field += c
                            i += 2
                            continue
                        in_quotes = False
                        i += 1
                        continue
                    field += c
                    i += 1
                    continue
                if c == '\r' or c == '\n':
                    i += 1
                    continue
                if d.escapechar is not None and c == d.escapechar and i + 1 < n:
                    field += line[i + 1]
                    started = True
                    i += 2
                    continue
                if c == d.delimiter:
                    row.append(self._value(field, quoted))
                    field = ''
                    quoted = False
                    started = False
                    i += 1
                    if d.skipinitialspace:
                        while i < n and line[i] == ' ':
                            i += 1
                    continue
                if c == d.quotechar and d.quoting != QUOTE_NONE and not started:
                    in_quotes = True
                    quoted = True
                    started = True
                    i += 1
                    continue
                field += c
                started = True
                i += 1
            if in_quotes:
                field += '\n'
                continue
            if not row and field == '' and not quoted:
                if line.strip('\r\n') == '':
                    return []
            row.append(self._value(field, quoted))
            return row

    def _value(self, field, quoted):
        if self.dialect.quoting == QUOTE_NONNUMERIC and not quoted and field != '':
            return float(field)
        return field


class writer:
    def __init__(self, csvfile, dialect='excel', **kw):
        self._out = csvfile
        self.dialect = _Settings(dialect, kw)

    def _field(self, value):
        d = self.dialect
        if value is None:
            text = ''
        elif isinstance(value, float) and value == int(value) and False:
            text = str(value)
        else:
            text = str(value)
        numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
        need = d.quoting == QUOTE_ALL or (d.quoting == QUOTE_NONNUMERIC and not numeric)
        if d.quoting == QUOTE_MINIMAL:
            need = text == '' and False
            for ch in (d.delimiter, d.quotechar, '\n', '\r'):
                if ch and ch in text:
                    need = True
            if text.startswith(' ') and d.skipinitialspace:
                need = True
        if d.quoting == QUOTE_NONE:
            if d.escapechar is not None:
                for ch in (d.escapechar, d.delimiter, d.quotechar):
                    if ch:
                        text = text.replace(ch, d.escapechar + ch)
            return text
        if need:
            if d.doublequote:
                text = text.replace(d.quotechar, d.quotechar + d.quotechar)
            elif d.escapechar is not None:
                text = text.replace(d.quotechar, d.escapechar + d.quotechar)
            return d.quotechar + text + d.quotechar
        return text

    def writerow(self, row):
        line = self.dialect.delimiter.join([self._field(v) for v in row]) + self.dialect.lineterminator
        self._out.write(line)
        return len(line)

    def writerows(self, rows):
        for row in rows:
            self.writerow(row)


class DictReader:
    def __init__(self, f, fieldnames=None, restkey=None, restval=None, dialect='excel', **kw):
        self.reader = reader(f, dialect, **kw)
        self._fieldnames = fieldnames
        self.restkey = restkey
        self.restval = restval
        self.line_num = 0

    @property
    def fieldnames(self):
        if self._fieldnames is None:
            try:
                self._fieldnames = next(self.reader)
            except StopIteration:
                pass
        self.line_num = self.reader.line_num
        return self._fieldnames

    def __iter__(self):
        return self

    def __next__(self):
        names = self.fieldnames
        row = next(self.reader)
        while row == []:
            row = next(self.reader)
        self.line_num = self.reader.line_num
        out = {}
        for k, name in enumerate(names):
            out[name] = row[k] if k < len(row) else self.restval
        if len(row) > len(names):
            out[self.restkey] = row[len(names):]
        return out


class DictWriter:
    def __init__(self, f, fieldnames, restval='', extrasaction='raise', dialect='excel', **kw):
        self.fieldnames = list(fieldnames)
        self.restval = restval
        self.extrasaction = extrasaction
        self.writer = writer(f, dialect, **kw)

    def writeheader(self):
        return self.writer.writerow(self.fieldnames)

    def writerow(self, rowdict):
        if self.extrasaction == 'raise':
            wrong = [k for k in rowdict if k not in self.fieldnames]
            if wrong:
                raise ValueError('dict contains fields not in fieldnames: ' + ', '.join([repr(x) for x in wrong]))
        return self.writer.writerow([rowdict.get(k, self.restval) for k in self.fieldnames])

    def writerows(self, rows):
        for r in rows:
            self.writerow(r)


class Sniffer:
    def sniff(self, sample, delimiters=None):
        candidates = delimiters or ',;\t|'
        lines = [l for l in sample.splitlines() if l.strip()][:10]
        best = ','
        best_score = -1
        for c in candidates:
            counts = [l.count(c) for l in lines]
            if counts and counts[0] > 0 and all(x == counts[0] for x in counts):
                if counts[0] > best_score:
                    best = c
                    best_score = counts[0]
        d = _Settings(excel, {'delimiter': best})
        return d

    def has_header(self, sample):
        rows = [r for r in reader(sample.splitlines(True))][:5]
        if len(rows) < 2:
            return False
        head = rows[0]
        for k in range(len(head)):
            try:
                float(head[k])
                return False
            except ValueError:
                pass
        for r in rows[1:]:
            for v in r:
                try:
                    float(v)
                    return True
                except ValueError:
                    pass
        return False
