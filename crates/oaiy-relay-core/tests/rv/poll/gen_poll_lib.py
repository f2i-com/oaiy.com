import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import my_poll as mp

def parse_body(text):
    if text is None:
        return None
    def hook(pairs):
        d = {}
        for k, v in pairs:
            if k in d:
                raise ValueError("dup")
            d[k] = v
        return d
    def bad(x):
        raise ValueError("const")
    try:
        return json.loads(text, object_pairs_hook=hook, parse_constant=bad)
    except Exception:
        return None
