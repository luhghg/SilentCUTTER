"""Video metadata via ffprobe (duration, fps, resolution)."""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass

from .ffmpeg_locate import ffmpeg_path, ffprobe_path


class FFToolNotFoundError(RuntimeError):
    """Raised when ffmpeg or ffprobe is not available in PATH."""


class FFProbeError(RuntimeError):
    """Raised when ffprobe fails to read the given file."""


@dataclass(frozen=True)
class VideoInfo:
    duration: float
    fps: float
    width: int
    height: int
    has_audio: bool


def check_tools_available() -> list[str]:
    """Return a list of missing tool names among ['ffmpeg', 'ffprobe'].

    Checks the bundled copies next to the app first (for a packaged build
    that ships its own ffmpeg), then falls back to PATH.
    """
    missing = []
    for tool, resolved in (("ffmpeg", ffmpeg_path()), ("ffprobe", ffprobe_path())):
        if shutil.which(resolved) is None:
            missing.append(tool)
    return missing


def _parse_fps(rate: str) -> float:
    # ffprobe reports frame rate as "num/den", e.g. "30000/1001" or "25/1".
    if "/" in rate:
        num_str, den_str = rate.split("/", 1)
        num, den = float(num_str), float(den_str)
        return num / den if den else 0.0
    return float(rate)


def get_video_info(path: str) -> VideoInfo:
    """Run ffprobe on ``path`` and return duration, fps, resolution, audio presence."""
    cmd = [
        ffprobe_path(),
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        path,
    ]
    try:
        result = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
    except FileNotFoundError as exc:
        raise FFToolNotFoundError("ffprobe не найден в PATH") from exc

    if result.returncode != 0:
        raise FFProbeError(
            f"ffprobe завершился с ошибкой (код {result.returncode}): {result.stderr.strip()}"
        )

    try:
        data = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise FFProbeError(f"Не удалось разобрать вывод ffprobe: {exc}") from exc

    streams = data.get("streams", [])
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), None)
    audio_stream = next((s for s in streams if s.get("codec_type") == "audio"), None)

    if video_stream is None:
        raise FFProbeError("В файле не найдена видеодорожка")

    fmt = data.get("format", {})
    duration_str = video_stream.get("duration") or fmt.get("duration")
    if duration_str is None:
        raise FFProbeError("Не удалось определить длительность видео")
    duration = float(duration_str)

    rate_str = video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate") or "0/1"
    fps = _parse_fps(rate_str)
    if not fps and video_stream.get("r_frame_rate"):
        fps = _parse_fps(video_stream["r_frame_rate"])

    width = int(video_stream.get("width") or 0)
    height = int(video_stream.get("height") or 0)

    return VideoInfo(
        duration=duration,
        fps=fps,
        width=width,
        height=height,
        has_audio=audio_stream is not None,
    )
