"""Text canonicalisation and model-input preprocessing (versioned).

Two separate functions, both pure and deterministic:

* ``preprocess(text, mode)`` -- the text a model sees.
    - ``none``    : raw text.
    - ``minimal`` : minimal sentiment preprocessing rule
                    (URLs, mentions, '#', Alef, emojis, whitespace).
    - ``alg3``    : Algorithm 3 (alg:preprocess) of the manuscript, implemented
                    step by step in the order printed in the paper.  The three
                    worked examples of Table tab:preprocess_examples are unit
                    tests (tests/test_textnorm.py).
    - ``mhamed``  : the CNN preprocessing
                    (punctuation, diacritics, Alef/Yeh/Hamza/Heh/Gaf
                    normalisation, stop-word removal with a vendored list).

* ``canonical_key(text)`` -- the de-duplication key (``CANON_VERSION``).
  key = alg3(NFKC(text)) followed by folding of Persian Yeh/Kaf variants.
  It is a function of the raw text, and every model-input mode above is
  *finer* than it for the Arabic content, so two texts that are equal under
  raw/minimal/alg3 input always share a key.  Grouping by key therefore
  guarantees that no identical model input crosses partitions.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import unicodedata

CANON_VERSION = "canon_v1"
ALG3_VERSION = "alg3_v1"

# ---------------------------------------------------------------- alg3 steps
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_MENTION_RE = re.compile(r"@\w+")
# Emoji / pictograph / symbol blocks + variation selectors + ZWJ + keycaps.
_EMOJI_RE = re.compile(
    "["
    "\U0001F000-\U0001FAFF"   # mahjong .. symbols & pictographs ext-A
    "\U00002600-\U000027BF"   # misc symbols, dingbats
    "\U00002300-\U000023FF"   # misc technical (watch, hourglass ...)
    "\U00002B00-\U00002BFF"   # arrows / stars
    "\U0000FE00-\U0000FE0F"   # variation selectors
    "\U0000200D"              # zero-width joiner
    "\U000020E3"              # combining keycap
    "\U0001F1E6-\U0001F1FF"   # flags
    "]+",
    flags=re.UNICODE,
)
# Tashkeel U+064B..U+0652, superscript Alef U+0670, Quranic marks U+06D6..U+06ED
_DIACRITICS_RE = re.compile("[ً-ْٰۖ-ۭ]")
_TATWEEL = "ـ"
_ALEF_RE = re.compile("[آأإٱ]")      # آ أ إ ٱ -> ا
_YEH_RE = re.compile("[ىئ]")                   # ى ئ -> ي (paper example 3: دائما -> دايما)
_TEH_MARBUTA = "ة"                                   # ة -> ه
_REPEAT_RE = re.compile(r"(.)\1{2,}")                    # >2 repeats -> 1 (paper: زووول -> زول)
# Arabic letters kept by RemoveNonArabic: U+0621-U+063A, U+0641-U+064A, U+0671-U+06D3
_NON_ARABIC_RE = re.compile("[^ء-غف-يٱ-ۓ\\s]")
_WS_RE = re.compile(r"\s+")


def alg3(text: str) -> str:
    """Manuscript Algorithm 3, in the printed order."""
    if not isinstance(text, str):
        return ""
    t = _URL_RE.sub(" ", text)                 # RemoveURLs
    t = _MENTION_RE.sub(" ", t)                # RemoveMentions
    t = t.replace("#", " ")                    # RemoveHashSymbol (keep hashtag text)
    t = _EMOJI_RE.sub(" ", t)                  # RemoveEmojis
    t = _DIACRITICS_RE.sub("", t)              # RemoveDiacritics
    t = _ALEF_RE.sub("ا", t)              # NormalizeAlef
    t = _YEH_RE.sub("ي", t)               # NormalizeYeh
    t = t.replace(_TEH_MARBUTA, "ه")      # NormalizeTehMarbuta
    t = t.replace(_TATWEEL, "")                # RemoveElongation (a): kashida
    t = _REPEAT_RE.sub(r"\1", t)               # RemoveElongation (b): runs > 2 -> 1
    t = _NON_ARABIC_RE.sub(" ", t)             # RemoveNonArabic (Latin, digits, punctuation, '_')
    t = _WS_RE.sub(" ", t).strip()             # NormalizeWhitespace
    return t


# ------------------------------------------------------------- minimal
_MIN_EMOJI_RE = re.compile("["
    "\U0001F600-\U0001F64F" "\U0001F300-\U0001F5FF" "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF" "\U00002702-\U000027B0" "\U000024C2-\U0001F251"
    "]+", flags=re.UNICODE)


def minimal(text: str) -> str:
    """Minimal sentiment preprocessing rule."""
    if not text or not isinstance(text, str):
        return ""
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = text.replace('#', '')
    text = re.sub(r'[إأآا]', 'ا', text)
    text = _MIN_EMOJI_RE.sub('', text)
    text = ' '.join(text.split())
    return text.strip()


# ------------------------------------------------------------- mhamed (CNN)
_RES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "resources")
_STOP_PATH = os.path.join(_RES, "stopwords_cnn.json")
_STOPWORDS = None
_PUNCT = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!"…"–ـ''' + '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
_PUNCT_TR = str.maketrans('', '', _PUNCT)
_MH_DIAC = re.compile("[ًٌٍَُِّْـ]")


def _stopwords():
    global _STOPWORDS
    if _STOPWORDS is None:
        with open(_STOP_PATH, encoding="utf-8") as f:
            d = json.load(f)
        _STOPWORDS = set(d["words"])
    return _STOPWORDS


def mhamed(text: str) -> str:
    """CNN preprocessing with a vendored stop list."""
    if not text or not isinstance(text, str):
        return ""
    text = text.translate(_PUNCT_TR)
    text = _MH_DIAC.sub('', text)
    text = re.sub("[إأآا]", "ا", text)
    text = re.sub("ى", "ي", text)
    text = re.sub("ؤ", "ء", text)
    text = re.sub("ئ", "ء", text)
    text = re.sub("ة", "ه", text)
    text = re.sub("گ", "ك", text)
    sw = _stopwords()
    return ' '.join(w for w in text.split() if w not in sw).strip()


MODES = {"none": lambda t: t if isinstance(t, str) else "", "minimal": minimal,
         "alg3": alg3, "mhamed": mhamed}


def preprocess(text: str, mode: str) -> str:
    return MODES[mode](text)


# ------------------------------------------------------------ canonical key
def canonical_key(text: str) -> str:
    t = unicodedata.normalize("NFKC", text if isinstance(text, str) else "")
    t = alg3(t)
    t = t.replace("ی", "ي").replace("ک", "ك")   # Persian Yeh/Kaf
    return t


def sha256(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def text_id(text: str) -> str:
    """SHA-256 of the canonical key (the row/group identifier)."""
    return sha256(canonical_key(text))
