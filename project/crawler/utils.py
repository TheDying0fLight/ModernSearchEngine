import re
from urllib.parse import urlparse
import tldextract
from langcodes import Language
import os

_CC_TLD_RE = re.compile(r'^[a-z]{2,3}$', re.IGNORECASE)
_GENERIC_TLDS = {
    "com", "org", "net", "edu", "gov", "io", "info", "biz", "co", "uk"
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
        if _CC_TLD_RE.match(first) or '-' in first:
            tag = normalize(first.replace('_', '-'))
            if tag: return tag

    # 2) First subdomain segment: de.example.com
    sub_parts = ext.subdomain.split('.') if ext.subdomain else []
    if sub_parts:
        candidate = sub_parts[-1].lower()
        if _CC_TLD_RE.match(candidate):
            tag = normalize(candidate)
            if tag: return tag

    # 3) Country‐code TLD: example.de
    suffix = ext.suffix.lower()
    if _CC_TLD_RE.match(suffix) and suffix not in _GENERIC_TLDS:
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