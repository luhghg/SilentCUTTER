import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from silence_cutter.censor import (
    FADE_STEPS,
    _build_sendcmd_script,
    _escape_filter_path,
)


# ---------------------------------------------------------------------------
# _escape_filter_path
# ---------------------------------------------------------------------------

def test_escape_simple_path():
    assert _escape_filter_path("/tmp/foo.cmds") == "'/tmp/foo.cmds'"


def test_escape_path_with_windows_drive_letter():
    # A bare colon inside single quotes is NOT safe on its own - ffmpeg's
    # per-filter option parser still reads it as a key=value separator and
    # errors out ("Invalid argument"), confirmed by reproducing it against a
    # real ffmpeg binary. Backslashes and the colon must be backslash-escaped
    # *before* the single-quote wrapping.
    assert _escape_filter_path("C:\\Users\\foo.cmds") == "'C\\:\\\\Users\\\\foo.cmds'"


def test_escape_path_with_backslash_only():
    assert _escape_filter_path("a\\b") == "'a\\\\b'"


def test_escape_path_with_single_quote():
    assert _escape_filter_path("it's/a/path.cmds") == "'it'\\''s/a/path.cmds'"


# ---------------------------------------------------------------------------
# _build_sendcmd_script
# ---------------------------------------------------------------------------

def test_hard_cut_interval_has_exactly_two_commands():
    # width = 0.05s, fade=0.02 -> 3*fade=0.06 > width -> hard cut.
    script = _build_sendcmd_script([(1.0, 1.05)], fade=0.02)
    lines = script.splitlines()
    assert lines == [
        "1.000000 volume volume 0.000000;",
        "1.050000 volume volume 1.000000;",
    ]


def test_faded_interval_has_ramp_out_and_ramp_in():
    script = _build_sendcmd_script([(1.0, 2.0)], fade=0.02)
    lines = script.splitlines()
    # (FADE_STEPS+1) steps ramping out at the start + same count ramping in at the end
    assert len(lines) == 2 * (FADE_STEPS + 1)
    assert lines[0] == "1.000000 volume volume 1.000000;"
    assert lines[FADE_STEPS] == "1.020000 volume volume 0.000000;"
    assert lines[FADE_STEPS + 1] == "1.980000 volume volume 0.000000;"
    assert lines[-1] == "2.000000 volume volume 1.000000;"


def test_interval_exactly_at_fade_threshold_gets_fade():
    # width = 0.06s == 3*fade exactly -> should still get the fade treatment.
    script = _build_sendcmd_script([(1.0, 1.06)], fade=0.02)
    assert len(script.splitlines()) == 2 * (FADE_STEPS + 1)


def test_zero_fade_always_cuts_hard():
    script = _build_sendcmd_script([(1.0, 5.0)], fade=0.0)
    lines = script.splitlines()
    assert lines == [
        "1.000000 volume volume 0.000000;",
        "5.000000 volume volume 1.000000;",
    ]


def test_commands_are_globally_sorted_by_time():
    # Intervals passed out of order must still yield a chronologically sorted script.
    script = _build_sendcmd_script([(5.0, 5.05), (1.0, 1.05)], fade=0.02)
    times = [float(line.split()[0]) for line in script.splitlines()]
    assert times == sorted(times)


def test_multiple_intervals_all_present():
    script = _build_sendcmd_script([(1.0, 1.05), (10.0, 10.05)], fade=0.02)
    assert "1.000000 volume volume 0.000000;" in script
    assert "1.050000 volume volume 1.000000;" in script
    assert "10.000000 volume volume 0.000000;" in script
    assert "10.050000 volume volume 1.000000;" in script


def test_no_sine_beep_amix_concat_or_expression_syntax():
    script = _build_sendcmd_script([(1.0, 2.0), (3.0, 3.03)], fade=0.02)
    assert "sine" not in script
    assert "beep" not in script
    assert "amix" not in script
    assert "concat" not in script
    assert "between(" not in script
    assert "eval=frame" not in script


def test_scales_to_thousands_of_intervals_without_blowing_up():
    # Regression guard: neither the old single-expression approach (parser
    # depth limit around ~100 terms) nor the trim+concat approach (became
    # impractically slow into the hundreds of segments) held up here.
    intervals = [(float(i), float(i) + 0.1) for i in range(1, 4000, 2)]
    script = _build_sendcmd_script(intervals, fade=0.02)
    # each interval is width 0.1 >= 3*0.02=0.06 -> gets the full fade treatment
    assert len(script.splitlines()) == len(intervals) * 2 * (FADE_STEPS + 1)
