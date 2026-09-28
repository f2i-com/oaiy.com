"""INI-style configuration files (pure Python, for bot.computer's sandbox)."""
import re

DEFAULTSECT = 'DEFAULT'


class Error(Exception):
    pass


class NoSectionError(Error):
    def __init__(self, section):
        Error.__init__(self, 'No section: %r' % (section,))
        self.section = section


class NoOptionError(Error):
    def __init__(self, option, section):
        Error.__init__(self, 'No option %r in section: %r' % (option, section))
        self.option = option
        self.section = section


class DuplicateSectionError(Error):
    pass


class MissingSectionHeaderError(Error):
    pass


_BOOLEAN = {'1': True, 'yes': True, 'true': True, 'on': True, '0': False, 'no': False, 'false': False, 'off': False}
_unset = object()


class SectionProxy:
    def __init__(self, parser, name):
        self._parser = parser
        self.name = name

    def __getitem__(self, key):
        if not self._parser.has_option(self.name, key):
            raise KeyError(key)
        return self._parser.get(self.name, key)

    def __setitem__(self, key, value):
        self._parser.set(self.name, key, value)

    def __contains__(self, key):
        return self._parser.has_option(self.name, key)

    def __iter__(self):
        return iter(self._parser.options(self.name))

    def keys(self):
        return self._parser.options(self.name)

    def items(self):
        return self._parser.items(self.name)

    def get(self, key, fallback=None):
        return self._parser.get(self.name, key, fallback=fallback)

    def getint(self, key, fallback=None):
        return self._parser.getint(self.name, key, fallback=fallback)

    def getfloat(self, key, fallback=None):
        return self._parser.getfloat(self.name, key, fallback=fallback)

    def getboolean(self, key, fallback=None):
        return self._parser.getboolean(self.name, key, fallback=fallback)


class ConfigParser:
    def __init__(self, defaults=None, allow_no_value=False, delimiters=('=', ':'), comment_prefixes=('#', ';'), interpolation=None, **kw):
        self._defaults = {}
        self._sections = {}
        self._allow_no_value = allow_no_value
        self._delimiters = delimiters
        self._comments = comment_prefixes
        if defaults:
            for k, v in defaults.items():
                self._defaults[self.optionxform(k)] = str(v)

    def optionxform(self, optionstr):
        return optionstr.lower()

    def defaults(self):
        return self._defaults

    def sections(self):
        return list(self._sections.keys())

    def add_section(self, section):
        if section in self._sections:
            raise DuplicateSectionError('Section %r already exists' % section)
        self._sections[section] = {}

    def has_section(self, section):
        return section in self._sections

    def options(self, section):
        if section not in self._sections:
            raise NoSectionError(section)
        keys = list(self._sections[section].keys())
        for k in self._defaults:
            if k not in keys:
                keys.append(k)
        return keys

    def has_option(self, section, option):
        option = self.optionxform(option)
        if section == DEFAULTSECT or not section:
            return option in self._defaults
        return section in self._sections and (option in self._sections[section] or option in self._defaults)

    def read_string(self, string, source='<string>'):
        section = None
        last = None
        for raw in string.splitlines():
            line = raw.strip()
            if not line or line[0] in self._comments:
                continue
            if raw[:1].isspace() and last is not None and section is not None:
                target = self._defaults if section == DEFAULTSECT else self._sections[section]
                target[last] = (target[last] or '') + '\n' + line
                continue
            m = re.match(r'^\[([^\]]+)\]$', line)
            if m:
                section = m.group(1)
                if section != DEFAULTSECT and section not in self._sections:
                    self._sections[section] = {}
                last = None
                continue
            if section is None:
                raise MissingSectionHeaderError('File contains no section headers: %r' % line)
            at = -1
            for d in self._delimiters:
                i = line.find(d)
                if i >= 0 and (at < 0 or i < at):
                    at = i
            if at < 0:
                if not self._allow_no_value:
                    raise Error('Source contains parsing errors: %r' % line)
                key, value = line, None
            else:
                key, value = line[:at].strip(), line[at + 1:].strip()
            key = self.optionxform(key)
            if section == DEFAULTSECT:
                self._defaults[key] = value
            else:
                self._sections[section][key] = value
            last = key

    def read(self, filenames, encoding=None):
        if isinstance(filenames, str):
            filenames = [filenames]
        done = []
        for name in filenames:
            try:
                text = open(name).read()
            except OSError:
                continue
            self.read_string(text, name)
            done.append(name)
        return done

    def read_file(self, f, source=None):
        self.read_string(f.read())

    def read_dict(self, dictionary, source='<dict>'):
        for section, values in dictionary.items():
            if section != DEFAULTSECT and section not in self._sections:
                self._sections[section] = {}
            for k, v in values.items():
                self.set(section, k, str(v))

    def get(self, section, option, *, raw=False, vars=None, fallback=_unset):
        option = self.optionxform(option)
        if section != DEFAULTSECT and section not in self._sections:
            if fallback is not _unset:
                return fallback
            raise NoSectionError(section)
        values = self._sections.get(section, {})
        if option in values:
            return values[option]
        if option in self._defaults:
            return self._defaults[option]
        if fallback is not _unset:
            return fallback
        raise NoOptionError(option, section)

    def getint(self, section, option, *, fallback=_unset, **kw):
        v = self.get(section, option, fallback=fallback)
        return v if v is fallback else int(v)

    def getfloat(self, section, option, *, fallback=_unset, **kw):
        v = self.get(section, option, fallback=fallback)
        return v if v is fallback else float(v)

    def getboolean(self, section, option, *, fallback=_unset, **kw):
        v = self.get(section, option, fallback=fallback)
        if v is fallback:
            return v
        if str(v).lower() not in _BOOLEAN:
            raise ValueError('Not a boolean: %s' % v)
        return _BOOLEAN[str(v).lower()]

    def items(self, section=_unset, raw=False, vars=None):
        if section is _unset:
            return [(s, self[s]) for s in [DEFAULTSECT] + self.sections()]
        out = dict(self._defaults)
        out.update(self._sections.get(section, {}))
        return list(out.items())

    def set(self, section, option, value=None):
        if section == DEFAULTSECT or not section:
            self._defaults[self.optionxform(option)] = value
            return
        if section not in self._sections:
            raise NoSectionError(section)
        self._sections[section][self.optionxform(option)] = value

    def remove_option(self, section, option):
        option = self.optionxform(option)
        if section not in self._sections:
            raise NoSectionError(section)
        existed = option in self._sections[section]
        if existed:
            del self._sections[section][option]
        return existed

    def remove_section(self, section):
        existed = section in self._sections
        if existed:
            del self._sections[section]
        return existed

    def write(self, fp, space_around_delimiters=True):
        d = ' = ' if space_around_delimiters else '='
        if self._defaults:
            fp.write('[%s]\n' % DEFAULTSECT)
            for k, v in self._defaults.items():
                fp.write('%s%s%s\n' % (k, d, v) if v is not None else '%s\n' % k)
            fp.write('\n')
        for section, values in self._sections.items():
            fp.write('[%s]\n' % section)
            for k, v in values.items():
                fp.write('%s%s%s\n' % (k, d, v) if v is not None else '%s\n' % k)
            fp.write('\n')

    def __getitem__(self, key):
        if key != DEFAULTSECT and not self.has_section(key):
            raise KeyError(key)
        return SectionProxy(self, key)

    def __contains__(self, key):
        return key == DEFAULTSECT or self.has_section(key)

    def __iter__(self):
        return iter([DEFAULTSECT] + self.sections())


RawConfigParser = ConfigParser
SafeConfigParser = ConfigParser
