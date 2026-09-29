"""Comparing sequences: SequenceMatcher, unified and context diffs, close matches (pure Python, for OAIY's sandbox)."""


class Match:
    def __init__(self, a, b, size):
        self.a = a
        self.b = b
        self.size = size

    def __iter__(self):
        return iter((self.a, self.b, self.size))

    def __getitem__(self, i):
        return (self.a, self.b, self.size)[i]

    def __repr__(self):
        return 'Match(a=%d, b=%d, size=%d)' % (self.a, self.b, self.size)


class SequenceMatcher:
    def __init__(self, isjunk=None, a='', b='', autojunk=True):
        self.isjunk = isjunk
        self.a = a
        self.b = b
        self._blocks = None
        self._b2j = None

    def set_seqs(self, a, b):
        self.set_seq1(a)
        self.set_seq2(b)

    def set_seq1(self, a):
        self.a = a
        self._blocks = None

    def set_seq2(self, b):
        self.b = b
        self._blocks = None
        self._b2j = None

    def _index(self):
        if self._b2j is None:
            b2j = {}
            for j, elt in enumerate(self.b):
                if self.isjunk is not None and self.isjunk(elt):
                    continue
                b2j.setdefault(elt, []).append(j)
            self._b2j = b2j
        return self._b2j

    def find_longest_match(self, alo=0, ahi=None, blo=0, bhi=None):
        if ahi is None:
            ahi = len(self.a)
        if bhi is None:
            bhi = len(self.b)
        b2j = self._index()
        besti, bestj, bestsize = alo, blo, 0
        j2len = {}
        for i in range(alo, ahi):
            newj2len = {}
            for j in b2j.get(self.a[i], []):
                if j < blo:
                    continue
                if j >= bhi:
                    break
                k = j2len.get(j - 1, 0) + 1
                newj2len[j] = k
                if k > bestsize:
                    besti, bestj, bestsize = i - k + 1, j - k + 1, k
            j2len = newj2len
        return Match(besti, bestj, bestsize)

    def get_matching_blocks(self):
        if self._blocks is not None:
            return self._blocks
        la, lb = len(self.a), len(self.b)
        queue = [(0, la, 0, lb)]
        blocks = []
        while queue:
            alo, ahi, blo, bhi = queue.pop()
            m = self.find_longest_match(alo, ahi, blo, bhi)
            i, j, k = m.a, m.b, m.size
            if k:
                blocks.append((i, j, k))
                if alo < i and blo < j:
                    queue.append((alo, i, blo, j))
                if i + k < ahi and j + k < bhi:
                    queue.append((i + k, ahi, j + k, bhi))
        blocks.sort()
        merged = []
        i1 = j1 = k1 = 0
        for i2, j2, k2 in blocks:
            if i1 + k1 == i2 and j1 + k1 == j2:
                k1 += k2
            else:
                if k1:
                    merged.append((i1, j1, k1))
                i1, j1, k1 = i2, j2, k2
        if k1:
            merged.append((i1, j1, k1))
        merged.append((la, lb, 0))
        self._blocks = [Match(x, y, z) for x, y, z in merged]
        return self._blocks

    def get_opcodes(self):
        i = j = 0
        out = []
        for m in self.get_matching_blocks():
            ai, bj, size = m.a, m.b, m.size
            tag = ''
            if i < ai and j < bj:
                tag = 'replace'
            elif i < ai:
                tag = 'delete'
            elif j < bj:
                tag = 'insert'
            if tag:
                out.append((tag, i, ai, j, bj))
            i, j = ai + size, bj + size
            if size:
                out.append(('equal', ai, i, bj, j))
        return out

    def get_grouped_opcodes(self, n=3):
        codes = self.get_opcodes()
        if not codes:
            codes = [('equal', 0, 1, 0, 1)]
        if codes[0][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[0]
            codes[0] = (tag, max(i1, i2 - n), i2, max(j1, j2 - n), j2)
        if codes[-1][0] == 'equal':
            tag, i1, i2, j1, j2 = codes[-1]
            codes[-1] = (tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n))
        nn = n + n
        group = []
        for tag, i1, i2, j1, j2 in codes:
            if tag == 'equal' and i2 - i1 > nn:
                group.append((tag, i1, min(i2, i1 + n), j1, min(j2, j1 + n)))
                yield group
                group = []
                i1, j1 = max(i1, i2 - n), max(j1, j2 - n)
            group.append((tag, i1, i2, j1, j2))
        if group and not (len(group) == 1 and group[0][0] == 'equal'):
            yield group

    def ratio(self):
        matches = sum([m.size for m in self.get_matching_blocks()])
        total = len(self.a) + len(self.b)
        return 2.0 * matches / total if total else 1.0

    def quick_ratio(self):
        return self.ratio()

    def real_quick_ratio(self):
        la, lb = len(self.a), len(self.b)
        return 2.0 * min(la, lb) / (la + lb) if la + lb else 1.0


def get_close_matches(word, possibilities, n=3, cutoff=0.6):
    scored = []
    s = SequenceMatcher()
    s.set_seq2(word)
    for x in possibilities:
        s.set_seq1(x)
        r = s.ratio()
        if r >= cutoff:
            scored.append((r, x))
    scored.sort(key=lambda t: -t[0])
    return [x for r, x in scored[:n]]


def _range(start, stop):
    beginning = start + 1
    length = stop - start
    if length == 1:
        return '%d' % beginning
    if not length:
        beginning -= 1
    return '%d,%d' % (beginning, length)


def unified_diff(a, b, fromfile='', tofile='', fromfiledate='', tofiledate='', n=3, lineterm='\n'):
    started = False
    for group in SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            yield '--- %s%s%s' % (fromfile, '\t' + fromfiledate if fromfiledate else '', lineterm)
            yield '+++ %s%s%s' % (tofile, '\t' + tofiledate if tofiledate else '', lineterm)
        first, last = group[0], group[-1]
        yield '@@ -%s +%s @@%s' % (_range(first[1], last[2]), _range(first[3], last[4]), lineterm)
        for tag, i1, i2, j1, j2 in group:
            if tag == 'equal':
                for line in a[i1:i2]:
                    yield ' ' + line
                continue
            if tag in ('replace', 'delete'):
                for line in a[i1:i2]:
                    yield '-' + line
            if tag in ('replace', 'insert'):
                for line in b[j1:j2]:
                    yield '+' + line


def context_diff(a, b, fromfile='', tofile='', fromfiledate='', tofiledate='', n=3, lineterm='\n'):
    started = False
    prefix = {'insert': '+ ', 'delete': '- ', 'replace': '! ', 'equal': '  '}
    for group in SequenceMatcher(None, a, b).get_grouped_opcodes(n):
        if not started:
            started = True
            yield '*** %s%s' % (fromfile, lineterm)
            yield '--- %s%s' % (tofile, lineterm)
        first, last = group[0], group[-1]
        yield '***************' + lineterm
        yield '*** %d,%d ****%s' % (first[1] + 1, last[2], lineterm)
        for tag, i1, i2, _, _ in group:
            if tag != 'insert':
                for line in a[i1:i2]:
                    yield prefix[tag] + line
        yield '--- %d,%d ----%s' % (first[3] + 1, last[4], lineterm)
        for tag, _, _, j1, j2 in group:
            if tag != 'delete':
                for line in b[j1:j2]:
                    yield prefix[tag] + line


def ndiff(a, b, linejunk=None, charjunk=None):
    for tag, i1, i2, j1, j2 in SequenceMatcher(None, a, b).get_opcodes():
        if tag == 'equal':
            for line in a[i1:i2]:
                yield '  ' + line
            continue
        for line in a[i1:i2]:
            yield '- ' + line
        for line in b[j1:j2]:
            yield '+ ' + line


def restore(delta, which):
    tag = {1: '- ', 2: '+ '}[which]
    for line in delta:
        if line[:2] in (tag, '  '):
            yield line[2:]


class Differ:
    def __init__(self, linejunk=None, charjunk=None):
        pass

    def compare(self, a, b):
        return ndiff(a, b)


class HtmlDiff:
    def make_table(self, a, b, fromdesc='', todesc='', context=False, numlines=5):
        rows = ['<table class="diff">']
        for line in ndiff(a, b):
            rows.append('<tr><td>%s</td></tr>' % line.replace('&', '&amp;').replace('<', '&lt;'))
        rows.append('</table>')
        return '\n'.join(rows)

    def make_file(self, a, b, fromdesc='', todesc='', context=False, numlines=5):
        return '<html><body>%s</body></html>' % self.make_table(a, b, fromdesc, todesc, context, numlines)
