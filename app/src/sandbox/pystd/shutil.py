"""High-level file operations (pure Python, for bot.computer's sandbox)."""
import os
import fnmatch


class Error(OSError):
    pass


class SameFileError(Error):
    pass


def copyfileobj(fsrc, fdst, length=0):
    fdst.write(fsrc.read())


def copyfile(src, dst, *, follow_symlinks=True):
    if os.path.abspath(src) == os.path.abspath(dst):
        raise SameFileError('%r and %r are the same file' % (src, dst))
    with open(src, 'rb') as f:
        data = f.read()
    with open(dst, 'wb') as f:
        f.write(data)
    return dst


def copymode(src, dst, *, follow_symlinks=True):
    return None


def copystat(src, dst, *, follow_symlinks=True):
    return None


def copy(src, dst, *, follow_symlinks=True):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src))
    return copyfile(src, dst)


copy2 = copy


def ignore_patterns(*patterns):
    def _ignore(path, names):
        out = set()
        for p in patterns:
            for n in fnmatch.filter(names, p):
                out.add(n)
        return out
    return _ignore


def copytree(src, dst, symlinks=False, ignore=None, copy_function=copy2, ignore_dangling_symlinks=False, dirs_exist_ok=False):
    names = os.listdir(src)
    ignored = ignore(src, names) if ignore is not None else set()
    if os.path.isdir(dst) and not dirs_exist_ok:
        raise FileExistsError('[Errno 17] File exists: %r' % dst)
    os.makedirs(dst, exist_ok=True)
    for name in names:
        if name in ignored:
            continue
        s = os.path.join(src, name)
        d = os.path.join(dst, name)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore, copy_function, ignore_dangling_symlinks, dirs_exist_ok)
        else:
            copy_function(s, d)
    return dst


def rmtree(path, ignore_errors=False, onerror=None, *, onexc=None):
    if not os.path.isdir(path):
        if ignore_errors:
            return
        raise NotADirectoryError('[Errno 20] Not a directory: %r' % path)
    for name in os.listdir(path):
        p = os.path.join(path, name)
        if os.path.isdir(p):
            rmtree(p, ignore_errors)
        else:
            os.remove(p)
    try:
        os.rmdir(path)
    except OSError:
        if not ignore_errors:
            raise


def move(src, dst, copy_function=copy2):
    if os.path.isdir(dst):
        dst = os.path.join(dst, os.path.basename(src.rstrip('/')))
    if os.path.isdir(src):
        copytree(src, dst, copy_function=copy_function)
        rmtree(src)
    else:
        copyfile(src, dst)
        os.remove(src)
    return dst


def which(cmd, mode=None, path=None):
    known = ('python', 'python3', 'node', 'sh', 'bash', 'ls', 'cat', 'grep', 'sed', 'awk', 'find', 'curl')
    return '/bin/' + cmd if cmd in known else None


class _usage:
    def __init__(self, total, used, free):
        self.total = total
        self.used = used
        self.free = free

    def __iter__(self):
        return iter((self.total, self.used, self.free))

    def __repr__(self):
        return 'usage(total=%d, used=%d, free=%d)' % (self.total, self.used, self.free)


def disk_usage(path):
    return _usage(256 << 20, 0, 256 << 20)


def get_terminal_size(fallback=(80, 24)):
    return os.terminal_size(fallback) if hasattr(os, 'terminal_size') else fallback


def make_archive(base_name, format, root_dir=None, base_dir=None, **kw):
    if format != 'zip':
        raise ValueError("unknown archive format '%s' (the sandbox makes zip archives)" % format)
    import zipfile
    root = root_dir or '.'
    target = base_name + '.zip'
    z = zipfile.ZipFile(target, 'w')
    for dirpath, dirs, files in os.walk(os.path.join(root, base_dir) if base_dir else root):
        for f in files:
            full = os.path.join(dirpath, f)
            z.write(full, os.path.relpath(full, root))
    z.close()
    return target
