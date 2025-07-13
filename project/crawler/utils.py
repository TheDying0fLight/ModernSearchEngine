import re
from urllib.parse import urlparse
import tldextract
from langcodes import Language
import os
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Callable, Any
import threading

_LANG_REG = re.compile(r'^[a-z]{2,3}$', re.IGNORECASE)
_LANG_REG_SHORT = re.compile(r'^[a-z]{2}$', re.IGNORECASE)
_GENERIC = {
    "com", "org", "net", "edu", "gov", "io", "info", "biz", "co", "uk", "www"
}


def predict_language_from_url(url: str) -> str:
    parsed = urlparse(url)
    ext = tldextract.extract(url)

    def normalize(code: str) -> str | None:
        try: return Language.get(code).to_tag()
        except Exception: return None

    # 1) Path prefix: /de/... or /pt-br/...
    path_parts = [p for p in parsed.path.split('/') if p]
    if path_parts:
        first = path_parts[0].lower()
        if _LANG_REG_SHORT.match(first) or '-' in first:
            tag = normalize(first.replace('_', '-'))
            if tag: return tag

    # 2) First subdomain segment: "de".example.com
    sub_parts = ext.subdomain.split('.') if ext.subdomain else []
    if sub_parts:
        candidate = sub_parts[-1].lower()
        if _LANG_REG.match(candidate) and candidate not in _GENERIC:
            tag = normalize(candidate)
            if tag: return tag

    # 3) Country‐code TLD: example."de"
    suffix = ext.suffix.lower()
    if _LANG_REG.match(suffix) and suffix not in _GENERIC:
        tag = normalize(suffix)
        if tag: return tag

    # else undefined
    return 'und'


# https://stackoverflow.com/questions/13852700/create-file-but-if-name-exists-add-number
def uniquify(path):
    filename, extension = os.path.splitext(path)
    counter = 1

    while os.path.exists(path):
        path = filename + " (" + str(counter) + ")" + extension
        counter += 1

    return path


class TrackingThreadPoolExecutor(ThreadPoolExecutor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._active = 0
        self._lock = threading.Lock()

    def submit(self, fn: Callable[..., Any], *args, **kwargs) -> Future:
        def wrapper(*args, **kwargs):
            with self._lock: self._active += 1
            try:
                return fn(*args, **kwargs)
            finally:
                with self._lock:
                    self._active -= 1

        return super().submit(wrapper, *args, **kwargs)

    @property
    def active_count(self) -> int:
        with self._lock: return self._active