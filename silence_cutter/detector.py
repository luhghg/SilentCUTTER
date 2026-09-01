"""Silence detection and speech-segment calculation.

Pure logic module: parses ffmpeg ``silencedetect`` output and turns it into
a list of "keep" (speech) segments ready to be cut with ffmpeg. Contains no
subprocess calls of its own so it can be unit tested with plain timecode
lists.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

# Matches lines like:
#   [silencedetect @ 0x...] silence_start: 12.345
#   [silencedetect @ 0x...] silence_end: 14.01 | silence_duration: 1.665
_SILENCE_START_RE = re.compile(r"silence_start:\s*(-?[\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*(-?[\d.]+)")


@dataclass(frozen=True)
class Segment:
    """A half-open time range ``[start, end)`` in seconds."""

    start: float
    end: float

    @property
    def duration(self) -> float:
        return self.end - self.start

    def as_tuple(self) -> tuple[float, float]:
        return (self.start, self.end)


def parse_silencedetect_output(stderr_text: str, duration: float) -> list[Segment]:
    """Parse ffmpeg ``silencedetect`` stderr into a list of silence Segments.

    ffmpeg prints ``silence_start`` followed by a matching ``silence_end`` for
    every silent range it finds. If the audio ends while still silent, the
    trailing ``silence_end`` line is never printed, so an open silence_start
    is closed at ``duration``.
    """
    silences: list[Segment] = []
    pending_start: float | None = None

    for line in stderr_text.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            pending_start = float(start_match.group(1))
            continue

        end_match = _SILENCE_END_RE.search(line)
        if end_match and pending_start is not None:
            end = float(end_match.group(1))
            silences.append(Segment(pending_start, end))
            pending_start = None

    if pending_start is not None:
        silences.append(Segment(pending_start, duration))

    return silences


def invert_silences(silences: list[Segment], duration: float) -> list[Segment]:
    """Turn a list of silence segments into the complementary speech segments."""
    speech: list[Segment] = []
    cursor = 0.0

    for silence in sorted(silences, key=lambda s: s.start):
        start = max(0.0, min(silence.start, duration))
        end = max(0.0, min(silence.end, duration))
        if start > cursor:
            speech.append(Segment(cursor, start))
        cursor = max(cursor, end)

    if cursor < duration:
        speech.append(Segment(cursor, duration))

    return speech


def apply_padding(segments: list[Segment], padding: float, duration: float) -> list[Segment]:
    """Expand each segment by ``padding`` seconds on both sides, clamped to bounds."""
    padded = []
    for seg in segments:
        start = max(0.0, seg.start - padding)
        end = min(duration, seg.end + padding)
        padded.append(Segment(start, end))
    return padded


def merge_close_segments(segments: list[Segment], min_gap: float) -> list[Segment]:
    """Merge consecutive segments whose gap is smaller than ``min_gap``."""
    if not segments:
        return []

    ordered = sorted(segments, key=lambda s: s.start)
    merged = [ordered[0]]

    for seg in ordered[1:]:
        last = merged[-1]
        if seg.start - last.end < min_gap:
            merged[-1] = Segment(last.start, max(last.end, seg.end))
        else:
            merged.append(seg)

    return merged


def compute_speech_segments(
    silences: list[Segment],
    duration: float,
    padding: float = 0.15,
    min_clip: float = 0.2,
    min_gap: float = 0.1,
) -> list[Segment]:
    """Full pipeline: silences -> speech segments ready to keep.

    Order matters:
      1. Invert silence ranges into raw speech segments.
      2. Drop raw segments shorter than ``min_clip`` (clicks/breaths) *before*
         padding, so padding can't inflate noise into something that looks
         long enough to keep.
      3. Pad the surviving segments.
      4. Merge segments left with a gap smaller than ``min_gap`` after padding
         (padding often closes small gaps between words).
    """
    speech = invert_silences(silences, duration)
    speech = [s for s in speech if s.duration >= min_clip]
    speech = apply_padding(speech, padding, duration)
    speech = merge_close_segments(speech, min_gap)
    return speech


def total_kept_duration(segments: list[Segment]) -> float:
    return sum(s.duration for s in segments)


def format_timecode(seconds: float) -> str:
    """Format seconds as ``HH:MM:SS.mmm``."""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    hours, rem_ms = divmod(total_ms, 3_600_000)
    minutes, rem_ms = divmod(rem_ms, 60_000)
    secs, ms = divmod(rem_ms, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"
