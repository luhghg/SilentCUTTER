import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silence_cutter.matcher import (
    MAX_MUTE_DURATION,
    Word,
    filter_profane_words,
    find_profanity,
    load_word_list,
    normalize_word,
)

ROOTS = ["бля", "хуй", "хуе", "ебан", "ебат", "пизд", "сука"]
WHITELIST = ["хуже", "художник", "сукно", "мудрый"]


# ---------------------------------------------------------------------------
# normalize_word
# ---------------------------------------------------------------------------

def test_normalize_lowercases():
    assert normalize_word("БЛЯ") == "бля"


def test_normalize_yo_to_e():
    assert normalize_word("ёбаный") == "ебаный"


def test_normalize_ukrainian_i_to_i():
    assert normalize_word("піздець") == "пиздець"


def test_normalize_ukrainian_yi_to_i():
    assert normalize_word("їбати") == "ибати"


def test_normalize_ukrainian_ye_to_e():
    assert normalize_word("гандєн") == "ганден"


def test_normalize_collapses_repeated_letters():
    assert normalize_word("бляяяя") == "бля"
    assert normalize_word("сукаааа") == "сука"


def test_normalize_strips_dash_separators():
    assert normalize_word("б-л-я") == "бля"


def test_normalize_strips_dot_separators():
    assert normalize_word("б.л.я") == "бля"


def test_normalize_strips_asterisk_separators():
    assert normalize_word("б*ля") == "бля"


def test_normalize_strips_edge_punctuation():
    assert normalize_word("«бля!»") == "бля"
    assert normalize_word("бля,") == "бля"


def test_normalize_empty_string():
    assert normalize_word("") == ""


def test_normalize_punctuation_only():
    assert normalize_word("...") == ""


# ---------------------------------------------------------------------------
# find_profanity - whitelist wins
# ---------------------------------------------------------------------------

def test_whitelisted_word_not_censored_even_if_root_prefixed():
    words = [Word("хуже", 1.0, 1.5)]
    assert find_profanity(words, ROOTS, WHITELIST) == []


def test_whitelisted_word_художник_not_censored():
    words = [Word("художник", 1.0, 1.8)]
    assert find_profanity(words, ROOTS, WHITELIST) == []


def test_whitelist_collision_after_repeat_collapse():
    # "суккулент" collapses to "сукулент", which *does* start with a "суку"-like
    # root once repeats are folded - whitelist must still protect it.
    words = [Word("суккулент", 1.0, 2.0)]
    roots = ["сука", "суку"]
    whitelist = ["суккулент"]
    assert find_profanity(words, roots, whitelist) == []


# ---------------------------------------------------------------------------
# find_profanity - prefix-only matching (not substring anywhere)
# ---------------------------------------------------------------------------

def test_root_not_at_start_is_not_censored_trebovat():
    words = [Word("требовать", 1.0, 1.9)]
    assert find_profanity(words, ["еб"], []) == []


def test_root_not_at_start_is_not_censored_khleb():
    words = [Word("хлеб", 1.0, 1.4)]
    assert find_profanity(words, ["еб"], []) == []


def test_root_at_start_is_censored():
    words = [Word("ебать", 1.0, 1.5)]
    assert find_profanity(words, ["ебат"], []) != []


# ---------------------------------------------------------------------------
# find_profanity - padding and lead
# ---------------------------------------------------------------------------

def test_padding_applied_both_sides():
    words = [Word("бля", 2.0, 2.3)]
    intervals = find_profanity(words, ROOTS, [], padding=0.1, lead=0.0)
    assert intervals == [(1.9, 2.4)]


def test_lead_shifts_start_forward_after_padding():
    # padded interval: [1.9, 2.4] (0.5s wide); lead=0.1 -> start becomes 2.0,
    # still >= 0.05s wide (0.4s), so lead applies.
    words = [Word("бля", 2.0, 2.3)]
    intervals = find_profanity(words, ROOTS, [], padding=0.1, lead=0.1)
    assert intervals == [(2.0, 2.4)]


def test_lead_skipped_when_it_would_shrink_interval_too_much():
    # word is very short: padded interval [0.94, 1.06] (0.12s wide).
    # lead=0.1 -> candidate start 1.04, remaining width 0.02s < 0.05s -> skip lead.
    words = [Word("бля", 1.0, 1.0)]
    intervals = find_profanity(words, ROOTS, [], padding=0.06, lead=0.1)
    assert intervals == [(0.94, 1.06)]


def test_lead_applied_exactly_at_minimum_threshold():
    # padded interval [0.0, 0.1] (0.1s). lead=0.05 -> candidate start 0.05,
    # remaining width exactly 0.05s -> boundary case, lead should apply (>=).
    words = [Word("бля", 0.05, 0.05)]
    intervals = find_profanity(words, ROOTS, [], padding=0.05, lead=0.05)
    assert intervals == [(0.05, 0.1)]


def test_start_never_negative():
    words = [Word("бля", 0.02, 0.1)]
    intervals = find_profanity(words, ROOTS, [], padding=0.1, lead=0.0)
    assert intervals[0][0] == 0.0


# ---------------------------------------------------------------------------
# find_profanity - merging overlapping intervals
# ---------------------------------------------------------------------------

def test_overlapping_intervals_are_merged():
    words = [
        Word("бля", 1.0, 1.3),
        Word("сука", 1.35, 1.6),
    ]
    intervals = find_profanity(words, ROOTS, [], padding=0.1, lead=0.0)
    assert len(intervals) == 1
    assert intervals[0] == pytest.approx((0.9, 1.7))


def test_non_overlapping_intervals_stay_separate():
    words = [
        Word("бля", 1.0, 1.2),
        Word("сука", 5.0, 5.3),
    ]
    intervals = find_profanity(words, ROOTS, [], padding=0.05, lead=0.0)
    assert intervals == [(0.95, 1.25), (4.95, 5.35)]


def test_touching_intervals_are_merged():
    words = [
        Word("бля", 1.0, 1.2),
        Word("сука", 1.3, 1.6),
    ]
    intervals = find_profanity(words, ROOTS, [], padding=0.05, lead=0.0)
    # [0.95, 1.25] and [1.25, 1.65] touch exactly at 1.25 -> merge.
    assert len(intervals) == 1
    assert intervals[0] == pytest.approx((0.95, 1.65))


def test_unsorted_words_still_merge_correctly():
    words = [
        Word("сука", 5.0, 5.3),
        Word("бля", 1.0, 1.2),
    ]
    intervals = find_profanity(words, ROOTS, [], padding=0.05, lead=0.0)
    assert intervals == [(0.95, 1.25), (4.95, 5.35)]


# ---------------------------------------------------------------------------
# find_profanity - dropping implausibly long merged intervals
#
# Whisper has a known failure mode on long audio: it can hallucinate a loop
# of short repeated "words" near the end of a stream. Individually each one
# is brief, but chained with no gaps they merge into one interval spanning
# minutes - a real symptom hit in production (a whole video tail muted).
# ---------------------------------------------------------------------------

def test_chain_of_short_words_merging_past_max_duration_is_dropped():
    # Simulate a hallucination loop: 200 tiny "бля" words, each overlapping
    # into the next (spacing 0.1s, duration 0.15s each - a real symptom of
    # Whisper repeating itself with no real gap), merging into one ~20s
    # interval - past MAX_MUTE_DURATION.
    words = [Word("бля", i * 0.1, i * 0.1 + 0.15) for i in range(200)]
    intervals = find_profanity(words, ROOTS, [], padding=0.0, lead=0.0)
    assert intervals == []


def test_merged_interval_within_max_duration_is_kept():
    # A short-ish overlapping chain (well under MAX_MUTE_DURATION) is
    # plausible real speech and must still be muted normally.
    words = [Word("бля", i * 1.0, i * 1.0 + 1.5) for i in range(10)]
    intervals = find_profanity(words, ROOTS, [], padding=0.0, lead=0.0)
    assert len(intervals) == 1
    start, end = intervals[0]
    assert end - start < MAX_MUTE_DURATION


def test_merged_interval_exactly_at_max_duration_is_kept():
    intervals = find_profanity(
        [Word("бля", 0.0, MAX_MUTE_DURATION)], ROOTS, [], padding=0.0, lead=0.0
    )
    assert intervals == [(0.0, MAX_MUTE_DURATION)]


def test_only_the_oversized_interval_is_dropped_others_survive():
    # A hallucination-length chain near the end shouldn't take out an
    # earlier, legitimate, normal-length match.
    words = [Word("сука", 1.0, 1.2)] + [
        Word("бля", 100.0 + i * 0.1, 100.0 + i * 0.1 + 0.15) for i in range(200)
    ]
    intervals = find_profanity(words, ROOTS, [], padding=0.0, lead=0.0)
    assert intervals == [(1.0, 1.2)]


# ---------------------------------------------------------------------------
# find_profanity - misc
# ---------------------------------------------------------------------------

def test_empty_word_list_gives_empty_result():
    assert find_profanity([], ROOTS, WHITELIST) == []


def test_no_matches_gives_empty_result():
    words = [Word("привет", 1.0, 1.5), Word("мир", 2.0, 2.3)]
    assert find_profanity(words, ROOTS, WHITELIST) == []


def test_mixed_clean_and_profane_words():
    words = [
        Word("привет", 1.0, 1.5),
        Word("бля", 2.0, 2.2),
        Word("мир", 3.0, 3.3),
    ]
    intervals = find_profanity(words, ROOTS, [], padding=0.05, lead=0.0)
    assert intervals == [(1.95, 2.25)]


# ---------------------------------------------------------------------------
# filter_profane_words - the raw per-word match, before padding/merge
# ---------------------------------------------------------------------------

def test_filter_profane_words_counts_each_word_separately():
    words = [
        Word("бля", 1.0, 1.2),
        Word("привет", 2.0, 2.3),
        Word("сука", 3.0, 3.3),
    ]
    matched = filter_profane_words(words, ROOTS, [])
    assert [w.text for w in matched] == ["бля", "сука"]


def test_filter_profane_words_respects_whitelist():
    words = [Word("хуже", 1.0, 1.5), Word("хуй", 2.0, 2.3)]
    matched = filter_profane_words(words, ROOTS, WHITELIST)
    assert [w.text for w in matched] == ["хуй"]


def test_filter_profane_words_count_survives_interval_merge():
    # Two adjacent profane words end up merged into a single beep interval,
    # but the per-word count must still be 2.
    words = [Word("бля", 1.0, 1.2), Word("сука", 1.21, 1.5)]
    matched = filter_profane_words(words, ROOTS, [])
    assert len(matched) == 2
    intervals = find_profanity(words, ROOTS, [], padding=0.05, lead=0.0)
    assert len(intervals) == 1


# ---------------------------------------------------------------------------
# load_word_list
# ---------------------------------------------------------------------------

def test_load_word_list_skips_comments_and_blanks(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("# comment\n\nбля\n\nХУЙ\n# another\nсука\n", encoding="utf-8")
    assert load_word_list(p) == ["бля", "хуй", "сука"]


def test_load_word_list_normalizes_entries(tmp_path):
    p = tmp_path / "list.txt"
    p.write_text("ЁБАНЫЙ\nсуккулент\n", encoding="utf-8")
    assert load_word_list(p) == ["ебаный", "сукулент"]


def test_load_default_profanity_and_whitelist_files_exist():
    roots = load_word_list(Path(__file__).resolve().parents[1] / "silence_cutter" / "data" / "profanity.txt")
    whitelist = load_word_list(Path(__file__).resolve().parents[1] / "silence_cutter" / "data" / "whitelist.txt")
    assert "хуй" in roots
    assert "хуже" in whitelist


def test_real_wordlists_whitelist_beats_profanity():
    roots = load_word_list(
        Path(__file__).resolve().parents[1] / "silence_cutter" / "data" / "profanity.txt"
    )
    whitelist = load_word_list(
        Path(__file__).resolve().parents[1] / "silence_cutter" / "data" / "whitelist.txt"
    )
    words = [
        Word("хуже", 1.0, 1.5),
        Word("художник", 2.0, 2.8),
        Word("сукно", 3.0, 3.5),
        Word("мудрый", 4.0, 4.6),
        Word("хер", 5.0, 5.3),
    ]
    assert find_profanity(words, roots, whitelist) == []


def test_real_wordlists_catch_actual_profanity():
    roots = load_word_list(
        Path(__file__).resolve().parents[1] / "silence_cutter" / "data" / "profanity.txt"
    )
    whitelist = load_word_list(
        Path(__file__).resolve().parents[1] / "silence_cutter" / "data" / "whitelist.txt"
    )
    words = [Word("бля", 1.0, 1.2), Word("сука", 2.0, 2.3)]
    intervals = find_profanity(words, roots, whitelist)
    assert len(intervals) == 2
