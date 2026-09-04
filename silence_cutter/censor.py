"""Applying an audio drop-out censor over matched profanity intervals via ffmpeg.

A second ffmpeg pass on an already-cut file: the video stream is
stream-copied (no re-encode). The audio is processed in a single streaming
pass by the ``volume`` filter, driven by an ``asendcmd`` command list that
schedules volume changes at specific timestamps - one command to mute, one
to restore, plus a handful of intermediate steps at each edge to approximate
a short fade instead of an instant (audibly clicky) cut.

Two other approaches were tried and rejected before this one, both for the
same reason: they don't scale to a real video with hundreds of profane
words.

- A single algebraic gain expression (one term per interval, summed) fed to
  ``volume=<expr>:eval=frame``. Works for a handful of words, but ffmpeg's
  expression parser (libavutil/eval.c) is a recursive-descent parser with a
  bounded recursion depth - a few hundred chained terms is enough to break
  it ("Cannot allocate memory", which isn't an obvious message for what's
  actually a parser depth limit).
- Splitting the audio into small trimmed keep/mute pieces (with ``afade``
  on the mute pieces' edges) and reassembling them with ``concat`` - the
  same trim+concat pattern cutter.py uses for silence cutting. That one
  doesn't hit a hard ceiling, but empirically got very slow as the segment
  count grew into the hundreds (many independent trims into the same
  source stream), unacceptably so for a long video with lots of matches.

``asendcmd`` sidesteps both problems: the filter graph itself stays a fixed
small size regardless of how many words are matched (only the *command
list* file grows, and ffmpeg reads that as a flat schedule, not an
expression to parse or a graph to rebuild), and it processes audio in one
linear pass instead of many small re-seeks - confirmed empirically to run
at 40-160x realtime even with thousands of scheduled changes.
"""

from __future__ import annotations

import subprocess
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from .cutter import run_filter_complex

MIN_FADE_INTERVAL_RATIO = 3  # below (fade * this), an interval is cut hard instead
FADE_STEPS = 8  # intermediate volume steps used to approximate a smooth fade


def _escape_filter_path(path: str) -> str:
    """Quote a filesystem path for safe use as an ffmpeg filter option value.

    Single quotes alone are *not* enough here: a colon inside a single-quoted
    option value (e.g. a Windows drive letter, "C:\\Users\\...") still gets
    read by ffmpeg's per-filter option parser as a key=value separator,
    breaking the parse with "Invalid argument" - confirmed by reproducing it
    locally. Backslash-escaping the backslashes and the colon first, and
    *then* wrapping the result in single quotes (for literal single quotes
    and safety in general) is what actually works.
    """
    escaped = path.replace("\\", "\\\\").replace(":", "\\:")
    escaped = escaped.replace("'", "'\\''")
    return "'" + escaped + "'"


def _fade_steps(edge_time: float, fade: float, rising: bool) -> list[tuple[float, float]]:
    """Volume steps ramping 1->0 (rising=False) or 0->1 (rising=True) over
    ``fade`` seconds starting at ``edge_time``.
    """
    steps = []
    for i in range(FADE_STEPS + 1):
        frac = i / FADE_STEPS
        value = frac if rising else 1.0 - frac
        steps.append((edge_time + frac * fade, value))
    return steps


def _build_sendcmd_script(intervals: list[tuple[float, float]], fade: float) -> str:
    commands: list[tuple[float, float]] = []
    for start, end in intervals:
        width = end - start
        if fade > 0 and width >= MIN_FADE_INTERVAL_RATIO * fade:
            commands.extend(_fade_steps(start, fade, rising=False))
            commands.extend(_fade_steps(end - fade, fade, rising=True))
        else:
            commands.append((start, 0.0))
            commands.append((end, 1.0))

    commands.sort(key=lambda c: c[0])
    return "\n".join(f"{t:.6f} volume volume {v:.6f};" for t, v in commands)


def censor(
    video_path: str,
    intervals: list[tuple[float, float]],
    out_path: str,
    fade: float = 0.02,
    duration: float = 0.0,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[subprocess.Popen], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Drop audio to silence over ``intervals`` (seconds) in ``video_path``.

    If ``intervals`` is empty, no ffmpeg pass is needed - ``video_path`` is
    simply moved to ``out_path`` as-is. On success ``video_path`` no longer
    exists (either moved or consumed by the ffmpeg pass); on cancellation or
    failure it is left in place.
    """
    if not intervals:
        if on_log is not None:
            on_log("Мат не найден, заглушение не требуется.")
        Path(video_path).replace(out_path)
        return

    if duration > 0:
        # Defense in depth: a mistimed word (e.g. a Whisper hallucination near
        # the end of long audio - see MAX_WORD_DURATION in transcriber.py)
        # should never be able to silence past the actual end of the file.
        intervals = [
            (start, min(end, duration)) for start, end in intervals if start < duration
        ]

    if not intervals:
        if on_log is not None:
            on_log("Мат не найден, заглушение не требуется.")
        Path(video_path).replace(out_path)
        return

    cmd_script_content = _build_sendcmd_script(intervals, fade)

    cmd_script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".cmds", prefix="silence_cutter_", delete=False, encoding="utf-8"
        ) as cmd_file:
            cmd_file.write(cmd_script_content)
            cmd_script_path = cmd_file.name

        filter_graph = (
            f"[0:a]asendcmd=f={_escape_filter_path(cmd_script_path)},"
            "volume=volume=1[aout]"
        )

        run_filter_complex(
            input_args=["-i", video_path],
            filter_script_content=filter_graph,
            map_args=["-map", "0:v", "-map", "[aout]"],
            output_args=["-c:v", "copy", "-c:a", "aac", "-b:a", "192k", out_path],
            total_duration=duration,
            on_progress=on_progress,
            on_log=on_log,
            register_process=register_process,
            cancel_event=cancel_event,
        )
    finally:
        if cmd_script_path is not None:
            Path(cmd_script_path).unlink(missing_ok=True)

    Path(video_path).unlink(missing_ok=True)
