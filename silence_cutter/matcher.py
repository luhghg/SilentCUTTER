"""Profanity word matching: normalization, root/whitelist lookup, and the
padding+lead math that turns matched words into mute intervals.

Pure logic module - no ffmpeg, no whisper - so it can be unit tested on
plain lists of Word objects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / "data"
DEFAULT_PROFANITY_PATH = _DATA_DIR / "profanity.txt"
DEFAULT_WHITELIST_PATH = _DATA_DIR / "whitelist.txt"

DEFAULT_PADDING = 0.06
DEFAULT_LEAD = 0.05
MIN_INTERVAL_AFTER_LEAD = 0.05

# Real continuous cursing - even a rapid-fire rant - essentially always has a
# micro-pause somewhere within this many seconds. Whisper has a known failure
# mode on long audio where it hallucinates a *loop* of short repeated
# "words" near the end of a stream: individually each one is brief (so the
# per-word duration guard in transcriber.py doesn't catch them), but they
# chain together with no gap and, if the repeated text happens to normalize
# to a profanity root, merge into one mute interval spanning minutes. A
# merged interval this long is far more likely to be that artifact than
# real speech, so it's dropped rather than muted - silently missing a
# genuinely long rant is a much smaller cost than wiping out minutes of
# real audio.
MAX_MUTE_DURATION = 12.0

_LETTER_MAP = str.maketrans({"ё": "е", "і": "и", "ї": "и", "є": "е"})
_NON_LETTER_RE = re.compile(r"[^a-zа-я]")
_REPEAT_RE = re.compile(r"(.)\1+")


@dataclass(frozen=True)
class Word:
    """A single transcribed word with its timing, in seconds."""

    text: str
    start: float
    end: float


def normalize_word(word: str) -> str:
    """Lowercase, fold ё/і/ї/є, strip punctuation/separators, collapse repeats."""
    w = word.lower().translate(_LETTER_MAP)
    w = _NON_LETTER_RE.sub("", w)
    w = _REPEAT_RE.sub(r"\1", w)
    return w


def load_word_list(path: str | Path) -> list[str]:
    """Read a list of words/roots from ``path``: one per line, '#' comments and
    blank lines ignored, each entry normalized the same way as transcribed words.
    """
    words: list[str] = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        norm = normalize_word(line)
        if norm:
            words.append(norm)
    return words


def filter_profane_words(
    words: list[Word], roots: list[str], whitelist: list[str], hardcore: bool = False
) -> list[Word]:
    """Return the subset of ``words`` that are profane, before padding/merge.

    A word is profane if its normalized form starts with one of ``roots`` (or,
    in ``hardcore`` mode, *contains* one of ``roots`` anywhere - catches a
    root buried after a prefix, e.g. "распиздяй" or "выблядок", at the cost
    of more false positives on ordinary words that happen to contain a root
    mid-word) and is not itself in ``whitelist`` (checked first, always
    wins). Useful on its own for reporting how many words were matched.
    """
    norm_roots = [normalize_word(r) for r in roots if normalize_word(r)]
    norm_whitelist = {normalize_word(w) for w in whitelist}

    matched: list[Word] = []
    for word in words:
        norm = normalize_word(word.text)
        if not norm or norm in norm_whitelist:
            continue
        if hardcore:
            is_match = any(root in norm for root in norm_roots)
        else:
            is_match = any(norm.startswith(root) for root in norm_roots)
        if is_match:
            matched.append(word)
    return matched


def find_profanity(
    words: list[Word],
    roots: list[str],
    whitelist: list[str],
    padding: float = DEFAULT_PADDING,
    lead: float = DEFAULT_LEAD,
    hardcore: bool = False,
) -> list[tuple[float, float]]:
    """Find mute intervals for profane words.

    Each match (see ``filter_profane_words``; ``hardcore`` is passed through)
    is padded on both sides, then ``lead`` shifts the interval's start
    forward in time so the first sound of the word stays audible - unless
    that would shrink the interval below ``MIN_INTERVAL_AFTER_LEAD``, in
    which case lead is skipped for that word. Overlapping/touching intervals
    are merged.
    """
    raw_intervals: list[tuple[float, float]] = []
    for word in filter_profane_words(words, roots, whitelist, hardcore=hardcore):
        start = max(0.0, word.start - padding)
        end = word.end + padding

        shifted_start = start + lead
        if end - shifted_start >= MIN_INTERVAL_AFTER_LEAD:
            start = shifted_start

        raw_intervals.append((start, end))

    merged = _merge_intervals(raw_intervals)
    return [(s, e) for s, e in merged if e - s <= MAX_MUTE_DURATION]


def _merge_intervals(intervals: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if not intervals:
        return []

    ordered = sorted(intervals)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged
