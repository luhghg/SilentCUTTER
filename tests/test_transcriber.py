import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Importing the module itself must not require faster-whisper to be
# installed - only calling transcribe() does (see transcriber.py's docstring).
from silence_cutter.transcriber import MAX_WORD_DURATION, _sanitize_word
from silence_cutter.matcher import Word


def test_normal_word_passes_through_unchanged():
    word = _sanitize_word("бля", 1.0, 1.3, total_duration=100.0)
    assert word == Word("бля", 1.0, 1.3)


def test_empty_text_rejected():
    assert _sanitize_word("   ", 1.0, 1.3, total_duration=100.0) is None


def test_end_before_start_rejected():
    assert _sanitize_word("бля", 5.0, 4.0, total_duration=100.0) is None


def test_implausibly_long_word_rejected():
    # Regression guard: a Whisper hallucination near the end of long audio
    # can emit a "word" spanning minutes - this must never reach the matcher.
    assert _sanitize_word("бля", 100.0, 100.0 + MAX_WORD_DURATION + 1, total_duration=1000.0) is None


def test_word_at_exactly_max_duration_accepted():
    word = _sanitize_word("бля", 100.0, 100.0 + MAX_WORD_DURATION, total_duration=1000.0)
    assert word is not None


def test_word_end_clamped_to_total_duration():
    word = _sanitize_word("бля", 9.9, 10.5, total_duration=10.0)
    assert word == Word("бля", 9.9, 10.0)


def test_word_start_clamped_to_total_duration():
    # Degenerate but shouldn't happen in practice; make sure it doesn't blow up.
    word = _sanitize_word("бля", 10.5, 10.5 + 0.2, total_duration=10.0)
    assert word is not None
    assert word.start <= 10.0
    assert word.end <= 10.0


def test_no_clamping_when_total_duration_unknown():
    # total_duration=0 means "unknown" (e.g. faster-whisper didn't report one) -
    # don't clamp in that case, there's nothing reliable to clamp against.
    word = _sanitize_word("бля", 1.0, 1.3, total_duration=0.0)
    assert word == Word("бля", 1.0, 1.3)
