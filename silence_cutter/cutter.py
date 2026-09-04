"""Building and running ffmpeg commands: silence detection pass and the
frame-accurate trim+concat cut pass.
"""

from __future__ import annotations

import re
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Callable, Optional

from .detector import Segment
from .ffmpeg_locate import ffmpeg_path

_TIME_RE = re.compile(r"time=(\d+):(\d+):(\d+\.?\d*)")

# Hide the console window ffmpeg would otherwise flash open on Windows.
_STARTUPINFO = None
_CREATIONFLAGS = 0
if sys.platform == "win32":
    _STARTUPINFO = subprocess.STARTUPINFO()
    _STARTUPINFO.dwFlags |= subprocess.STARTF_USESHOWWINDOW
    _CREATIONFLAGS = subprocess.CREATE_NO_WINDOW


class CutterError(RuntimeError):
    """Raised for ffmpeg failures, with a human-readable message."""


class CancelledError(Exception):
    """Raised when the user cancels a running ffmpeg operation."""


def _parse_time(line: str) -> Optional[float]:
    match = _TIME_RE.search(line)
    if not match:
        return None
    hours, minutes, seconds = match.groups()
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def _run_streaming(
    cmd: list[str],
    total_duration: float,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[subprocess.Popen], None]] = None,
) -> tuple[int, str]:
    """Run an ffmpeg command, streaming stderr line by line.

    Returns (returncode, full_stderr_text). Progress is reported as a
    fraction in [0, 1] based on the ``time=`` markers ffmpeg prints, relative
    to ``total_duration``.
    """
    try:
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            startupinfo=_STARTUPINFO,
            creationflags=_CREATIONFLAGS,
        )
    except FileNotFoundError as exc:
        raise CutterError("ffmpeg не найден в PATH") from exc

    if register_process is not None:
        register_process(proc)

    lines: list[str] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        lines.append(line)
        stripped = line.rstrip("\n")

        current = _parse_time(stripped)
        if current is not None and total_duration > 0 and on_progress is not None:
            on_progress(min(1.0, max(0.0, current / total_duration)))

        if on_log is not None and ("silence_" in stripped or "Error" in stripped or "error" in stripped):
            on_log(stripped)

    returncode = proc.wait()
    return returncode, "".join(lines)


def run_silence_detection(
    input_path: str,
    noise_db: float,
    min_silence_duration: float,
    total_duration: float,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[subprocess.Popen], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> str:
    """Run ffmpeg's silencedetect filter and return the raw stderr text."""
    cmd = [
        ffmpeg_path(),
        "-hide_banner",
        "-i",
        input_path,
        "-af",
        f"silencedetect=noise={noise_db}dB:d={min_silence_duration}",
        "-f",
        "null",
        "-",
    ]
    returncode, stderr_text = _run_streaming(
        cmd, total_duration, on_progress, on_log, register_process
    )
    if cancel_event is not None and cancel_event.is_set():
        raise CancelledError()
    if returncode != 0:
        raise CutterError(_humanize_ffmpeg_error(stderr_text, returncode))
    return stderr_text


def build_filter_complex_script(segments: list[Segment], has_audio: bool) -> str:
    """Build the -filter_complex_script contents for a frame-accurate trim+concat cut."""
    if not segments:
        raise CutterError("Не осталось сегментов речи для сборки видео")

    chains: list[str] = []
    for i, seg in enumerate(segments):
        chains.append(
            f"[0:v]trim=start={seg.start:.6f}:end={seg.end:.6f},"
            f"setpts=PTS-STARTPTS[v{i}]"
        )
        if has_audio:
            chains.append(
                f"[0:a]atrim=start={seg.start:.6f}:end={seg.end:.6f},"
                f"asetpts=PTS-STARTPTS[a{i}]"
            )

    n = len(segments)
    if has_audio:
        concat_inputs = "".join(f"[v{i}][a{i}]" for i in range(n))
        concat_chain = f"{concat_inputs}concat=n={n}:v=1:a=1[outv][outa]"
    else:
        concat_inputs = "".join(f"[v{i}]" for i in range(n))
        concat_chain = f"{concat_inputs}concat=n={n}:v=1:a=0[outv]"

    chains.append(concat_chain)
    return ";\n".join(chains)


def run_filter_complex(
    input_args: list[str],
    filter_script_content: str,
    map_args: list[str],
    output_args: list[str],
    total_duration: float,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[subprocess.Popen], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Run ffmpeg with a (possibly large) filter_complex graph.

    The graph is written to a temp script file and passed via
    ``-filter_complex_script`` first (needed for very large graphs that
    wouldn't fit on the command line). Some ffmpeg builds don't recognize
    that option though, so on that specific failure this falls back to
    passing the graph inline via ``-filter_complex``.
    """

    def build_cmd(filter_args: list[str]) -> list[str]:
        return (
            [ffmpeg_path(), "-hide_banner", "-y"]
            + input_args
            + filter_args
            + map_args
            + output_args
        )

    script_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", prefix="silence_cutter_", delete=False, encoding="utf-8"
        ) as script_file:
            script_file.write(filter_script_content)
            script_path = script_file.name

        cmd = build_cmd(["-filter_complex_script", script_path])
        returncode, stderr_text = _run_streaming(
            cmd, total_duration, on_progress, on_log, register_process
        )
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()

        if returncode != 0 and "filter_complex_script" in stderr_text and (
            "Unrecognized option" in stderr_text or "Error splitting the argument list" in stderr_text
        ):
            # This ffmpeg build doesn't support -filter_complex_script (seen on some
            # very recent builds) - fall back to passing the graph inline.
            if on_log is not None:
                on_log(
                    "Эта сборка ffmpeg не поддерживает -filter_complex_script, "
                    "пробую передать фильтр напрямую..."
                )
            cmd = build_cmd(["-filter_complex", filter_script_content])
            returncode, stderr_text = _run_streaming(
                cmd, total_duration, on_progress, on_log, register_process
            )
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()

        if returncode != 0:
            raise CutterError(_humanize_ffmpeg_error(stderr_text, returncode))
    finally:
        if script_path is not None:
            Path(script_path).unlink(missing_ok=True)


# A single trim+concat filter graph handling hundreds of segments has been
# observed (on real footage, both H.264 and HEVC) to silently truncate the
# video track while the audio track keeps going full length - the video
# encoder falls behind, and somewhere between ffmpeg's muxing queue, filter
# graph buffering and encoder backpressure it starts dropping video frames
# without ever raising an error. -max_muxing_queue_size alone didn't fully
# fix it at large segment counts. Splitting into bounded batches sidesteps
# the problem category entirely instead of chasing the exact internal limit:
# each ffmpeg invocation only ever has to hold BATCH_SIZE segments open at
# once, well inside sizes that have run clean in testing (up to several
# hundred), and the batches are stitched back together with the concat
# demuxer, which is a fast stream copy - no re-encode, no filter graph.
BATCH_SIZE = 40


def run_cut(
    input_path: str,
    output_path: str,
    segments: list[Segment],
    fps: float,
    has_audio: bool,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[subprocess.Popen], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Cut and concatenate ``segments`` from ``input_path`` into ``output_path``."""
    if len(segments) <= BATCH_SIZE:
        _run_cut_batch(
            input_path,
            output_path,
            segments,
            fps,
            has_audio,
            on_progress=on_progress,
            on_log=on_log,
            register_process=register_process,
            cancel_event=cancel_event,
        )
        return

    batches = [segments[i : i + BATCH_SIZE] for i in range(0, len(segments), BATCH_SIZE)]
    total_duration = sum(seg.duration for seg in segments)

    if on_log is not None:
        on_log(
            f"Сегментов много ({len(segments)}) — режу пачками по {BATCH_SIZE}, "
            "чтобы видео не оборвалось раньше времени."
        )

    batch_paths: list[str] = []
    concat_list_path: Optional[str] = None
    try:
        elapsed_duration = 0.0
        for i, batch in enumerate(batches):
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()

            with tempfile.NamedTemporaryFile(
                suffix=".mp4", prefix="silence_cutter_batch_", delete=False
            ) as batch_file:
                batch_path = batch_file.name
            batch_paths.append(batch_path)

            batch_duration = sum(seg.duration for seg in batch)
            batch_start = elapsed_duration

            def batch_progress(fraction: float, _start: float = batch_start, _dur: float = batch_duration) -> None:
                if on_progress is not None and total_duration > 0:
                    on_progress(min(1.0, (_start + fraction * _dur) / total_duration))

            if on_log is not None:
                on_log(f"Пачка {i + 1}/{len(batches)} ({len(batch)} сегментов)...")

            _run_cut_batch(
                input_path,
                batch_path,
                batch,
                fps,
                has_audio,
                on_progress=batch_progress,
                on_log=on_log,
                register_process=register_process,
                cancel_event=cancel_event,
            )
            elapsed_duration += batch_duration

        concat_list_path = _write_concat_list(batch_paths)
        cmd = [
            ffmpeg_path(),
            "-hide_banner",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path,
            "-c",
            "copy",
            output_path,
        ]
        returncode, stderr_text = _run_streaming(cmd, total_duration, None, on_log, register_process)
        if cancel_event is not None and cancel_event.is_set():
            raise CancelledError()
        if returncode != 0:
            raise CutterError(_humanize_ffmpeg_error(stderr_text, returncode))
    finally:
        for batch_path in batch_paths:
            Path(batch_path).unlink(missing_ok=True)
        if concat_list_path is not None:
            Path(concat_list_path).unlink(missing_ok=True)


def _write_concat_list(paths: list[str]) -> str:
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".txt", prefix="silence_cutter_concat_", delete=False, encoding="utf-8"
    ) as list_file:
        for path in paths:
            escaped = path.replace("'", "'\\''")
            list_file.write(f"file '{escaped}'\n")
        return list_file.name


def _run_cut_batch(
    input_path: str,
    output_path: str,
    segments: list[Segment],
    fps: float,
    has_audio: bool,
    on_progress: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    register_process: Optional[Callable[[subprocess.Popen], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> None:
    """Cut and concatenate ``segments`` (assumed small enough for one filter graph)."""
    script_content = build_filter_complex_script(segments, has_audio)
    output_duration = sum(seg.duration for seg in segments)

    map_args = ["-map", "[outv]"]
    if has_audio:
        map_args += ["-map", "[outa]"]

    # On slow-to-encode source (e.g. HEVC at high fps), video packets can back
    # up behind the much cheaper audio stream faster than the muxer's default
    # queue allows; past that limit ffmpeg silently *drops* video packets and
    # still exits 0, producing a short video track next to a full-length audio
    # one. A bigger queue lets the encoder catch up instead of losing frames.
    output_args = ["-max_muxing_queue_size", "9999"]
    if fps > 0:
        output_args += ["-r", f"{fps:.6f}"]
    elif on_log is not None:
        on_log(
            "Не удалось определить частоту кадров исходного видео — "
            "пропускаю принудительный fps, ffmpeg подберёт сам."
        )
    output_args += ["-c:v", "libx264", "-crf", "20", "-preset", "veryfast"]
    if has_audio:
        output_args += ["-c:a", "aac", "-b:a", "192k"]
    else:
        output_args += ["-an"]
    output_args.append(output_path)

    run_filter_complex(
        input_args=["-i", input_path],
        filter_script_content=script_content,
        map_args=map_args,
        output_args=output_args,
        total_duration=output_duration,
        on_progress=on_progress,
        on_log=on_log,
        register_process=register_process,
        cancel_event=cancel_event,
    )


def _humanize_ffmpeg_error(stderr_text: str, returncode: int) -> str:
    error_lines = [
        line.strip()
        for line in stderr_text.splitlines()
        if line.strip() and ("Error" in line or "error" in line or "No such file" in line)
    ]
    if error_lines:
        # The line right before the final error often names *which* option or
        # stream it was about (e.g. "Error setting option r ... to value 0"),
        # while the last line alone is sometimes just a bare "Invalid
        # argument" - keep up to the last two for more useful context.
        detail = " | ".join(error_lines[-2:])
    else:
        tail = [line.strip() for line in stderr_text.splitlines() if line.strip()]
        detail = tail[-1] if tail else "нет дополнительной информации"
    return f"ffmpeg завершился с ошибкой (код {returncode}): {detail}"
