import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silence_cutter.detector import (
    Segment,
    apply_padding,
    compute_speech_segments,
    format_timecode,
    invert_silences,
    merge_close_segments,
    parse_silencedetect_output,
    total_kept_duration,
)


# ---------------------------------------------------------------------------
# parse_silencedetect_output
# ---------------------------------------------------------------------------

def test_parse_single_silence_range():
    stderr = """
[silencedetect @ 0x7f] silence_start: 1.5
[silencedetect @ 0x7f] silence_end: 3.2 | silence_duration: 1.7
"""
    result = parse_silencedetect_output(stderr, duration=10.0)
    assert result == [Segment(1.5, 3.2)]


def test_parse_multiple_silence_ranges():
    stderr = """
[silencedetect @ 0x1] silence_start: 0
[silencedetect @ 0x1] silence_end: 0.5 | silence_duration: 0.5
[silencedetect @ 0x1] silence_start: 5.0
[silencedetect @ 0x1] silence_end: 6.25 | silence_duration: 1.25
"""
    result = parse_silencedetect_output(stderr, duration=10.0)
    assert result == [Segment(0, 0.5), Segment(5.0, 6.25)]


def test_parse_trailing_silence_with_no_end_uses_duration():
    stderr = """
[silencedetect @ 0x1] silence_start: 8.0
"""
    result = parse_silencedetect_output(stderr, duration=10.0)
    assert result == [Segment(8.0, 10.0)]


def test_parse_ignores_unrelated_lines():
    stderr = """
frame=  100 fps=25 q=-1.0 size=    256kB time=00:00:04.00 bitrate= 524.3kbits/s
[silencedetect @ 0x1] silence_start: 2.0
random noise line
[silencedetect @ 0x1] silence_end: 2.5 | silence_duration: 0.5
"""
    result = parse_silencedetect_output(stderr, duration=10.0)
    assert result == [Segment(2.0, 2.5)]


def test_parse_no_silence_returns_empty():
    result = parse_silencedetect_output("no silence here", duration=10.0)
    assert result == []


# ---------------------------------------------------------------------------
# invert_silences
# ---------------------------------------------------------------------------

def test_invert_silence_in_the_middle():
    silences = [Segment(4.0, 6.0)]
    speech = invert_silences(silences, duration=10.0)
    assert speech == [Segment(0.0, 4.0), Segment(6.0, 10.0)]


def test_invert_silence_at_start():
    silences = [Segment(0.0, 2.0)]
    speech = invert_silences(silences, duration=10.0)
    assert speech == [Segment(2.0, 10.0)]


def test_invert_silence_at_end():
    silences = [Segment(8.0, 10.0)]
    speech = invert_silences(silences, duration=10.0)
    assert speech == [Segment(0.0, 8.0)]


def test_invert_no_silence_keeps_whole_clip():
    speech = invert_silences([], duration=10.0)
    assert speech == [Segment(0.0, 10.0)]


def test_invert_entire_clip_silent_yields_nothing():
    silences = [Segment(0.0, 10.0)]
    speech = invert_silences(silences, duration=10.0)
    assert speech == []


def test_invert_multiple_silences():
    silences = [Segment(1.0, 2.0), Segment(5.0, 5.5), Segment(9.0, 10.0)]
    speech = invert_silences(silences, duration=10.0)
    assert speech == [Segment(0.0, 1.0), Segment(2.0, 5.0), Segment(5.5, 9.0)]


def test_invert_unsorted_silences_are_handled():
    silences = [Segment(5.0, 5.5), Segment(1.0, 2.0)]
    speech = invert_silences(silences, duration=10.0)
    assert speech == [Segment(0.0, 1.0), Segment(2.0, 5.0), Segment(5.5, 10.0)]


# ---------------------------------------------------------------------------
# apply_padding
# ---------------------------------------------------------------------------

def test_apply_padding_expands_both_sides():
    segments = [Segment(2.0, 4.0)]
    padded = apply_padding(segments, padding=0.15, duration=10.0)
    assert padded == [Segment(1.85, 4.15)]


def test_apply_padding_clamped_at_zero():
    segments = [Segment(0.05, 4.0)]
    padded = apply_padding(segments, padding=0.15, duration=10.0)
    assert padded == [Segment(0.0, 4.15)]


def test_apply_padding_clamped_at_duration():
    segments = [Segment(2.0, 9.95)]
    padded = apply_padding(segments, padding=0.15, duration=10.0)
    assert padded == [Segment(1.85, 10.0)]


# ---------------------------------------------------------------------------
# merge_close_segments
# ---------------------------------------------------------------------------

def test_merge_close_segments_merges_small_gap():
    segments = [Segment(0.0, 2.0), Segment(2.05, 4.0)]
    merged = merge_close_segments(segments, min_gap=0.1)
    assert merged == [Segment(0.0, 4.0)]


def test_merge_close_segments_keeps_large_gap_separate():
    segments = [Segment(0.0, 2.0), Segment(2.5, 4.0)]
    merged = merge_close_segments(segments, min_gap=0.1)
    assert merged == [Segment(0.0, 2.0), Segment(2.5, 4.0)]


def test_merge_close_segments_chains_multiple():
    segments = [Segment(0.0, 1.0), Segment(1.05, 2.0), Segment(2.08, 3.0)]
    merged = merge_close_segments(segments, min_gap=0.1)
    assert merged == [Segment(0.0, 3.0)]


def test_merge_close_segments_handles_overlap():
    segments = [Segment(0.0, 2.0), Segment(1.5, 4.0)]
    merged = merge_close_segments(segments, min_gap=0.1)
    assert merged == [Segment(0.0, 4.0)]


def test_merge_close_segments_empty_input():
    assert merge_close_segments([], min_gap=0.1) == []


# ---------------------------------------------------------------------------
# compute_speech_segments (full pipeline)
# ---------------------------------------------------------------------------

def test_compute_speech_segments_basic():
    silences = [Segment(4.0, 6.0)]
    segments = compute_speech_segments(
        silences, duration=10.0, padding=0.15, min_clip=0.2, min_gap=0.1
    )
    assert segments == [Segment(0.0, 4.15), Segment(5.85, 10.0)]


def test_compute_speech_segments_drops_short_clicks():
    # A tiny 0.1s speech blip between two silences should be dropped
    # (it's below min_clip=0.2), leaving the two silences merged into one gap.
    silences = [Segment(0.0, 3.0), Segment(3.1, 6.0)]
    segments = compute_speech_segments(
        silences, duration=6.0, padding=0.0, min_clip=0.2, min_gap=0.1
    )
    assert segments == []


def test_compute_speech_segments_keeps_clip_at_exactly_min_clip():
    silences = [Segment(0.0, 3.0), Segment(3.2, 6.0)]
    segments = compute_speech_segments(
        silences, duration=6.0, padding=0.0, min_clip=0.2, min_gap=0.1
    )
    assert segments == [Segment(3.0, 3.2)]


def test_compute_speech_segments_padding_closes_gap_and_merges():
    # Two speech chunks separated by 0.2s of silence; with padding=0.15 on
    # each side the padded segments touch/overlap and must merge into one.
    silences = [Segment(2.0, 2.2)]
    segments = compute_speech_segments(
        silences, duration=5.0, padding=0.15, min_clip=0.2, min_gap=0.1
    )
    assert segments == [Segment(0.0, 5.0)]


def test_compute_speech_segments_no_silence_at_all():
    segments = compute_speech_segments(
        [], duration=8.0, padding=0.15, min_clip=0.2, min_gap=0.1
    )
    assert segments == [Segment(0.0, 8.0)]


def test_compute_speech_segments_all_silence():
    segments = compute_speech_segments(
        [Segment(0.0, 8.0)], duration=8.0, padding=0.15, min_clip=0.2, min_gap=0.1
    )
    assert segments == []


def test_compute_speech_segments_realistic_multi_cut():
    silences = [
        Segment(0.0, 1.0),   # leading silence
        Segment(3.0, 4.0),   # mid silence
        Segment(4.05, 4.1),  # tiny gap, becomes a dropped click after invert
        Segment(9.0, 10.0),  # trailing silence
    ]
    segments = compute_speech_segments(
        silences, duration=10.0, padding=0.15, min_clip=0.2, min_gap=0.1
    )
    # raw speech (from invert): [1,3], [4,4.05] (0.05s -> dropped), [4.1,9]
    # after padding: [0.85,3.15], [3.95,9.15]
    # gap between them is 3.95-3.15=0.8 > min_gap, stays separate
    assert len(segments) == 2
    assert segments[0].start == pytest.approx(0.85)
    assert segments[0].end == pytest.approx(3.15)
    assert segments[1].start == pytest.approx(3.95)
    assert segments[1].end == pytest.approx(9.15)


# ---------------------------------------------------------------------------
# total_kept_duration
# ---------------------------------------------------------------------------

def test_total_kept_duration():
    segments = [Segment(0.0, 2.0), Segment(3.0, 3.5)]
    assert total_kept_duration(segments) == 2.5


def test_total_kept_duration_empty():
    assert total_kept_duration([]) == 0


# ---------------------------------------------------------------------------
# format_timecode
# ---------------------------------------------------------------------------

def test_format_timecode_basic():
    assert format_timecode(0.0) == "00:00:00.000"


def test_format_timecode_seconds_and_ms():
    assert format_timecode(65.5) == "00:01:05.500"


def test_format_timecode_hours():
    assert format_timecode(3661.234) == "01:01:01.234"


def test_format_timecode_rounds_ms():
    assert format_timecode(1.9999) == "00:00:02.000"


def test_format_timecode_negative_clamped_to_zero():
    assert format_timecode(-5.0) == "00:00:00.000"
