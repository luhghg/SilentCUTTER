"""Speech-to-text via faster-whisper, with word-level timestamps.

Kept separate from matcher.py so that Word (the tiny data holder used
throughout the profanity pipeline) can be unit tested without pulling in
faster-whisper - only this module needs the actual model.
"""

from __future__ import annotations

import threading
from typing import Callable, Optional

from .cutter import CancelledError
from .matcher import Word

__all__ = ["Word", "TranscriberError", "transcribe", "WHISPER_PROFANITY_PROMPT"]

# Real spoken words essentially never take this long. On long audio, Whisper
# occasionally hallucinates near the end of a stream (a known failure mode -
# it can get stuck and emit a "word" whose timestamp spans minutes). Without
# a check like this, a single such glitch that happens to normalize to a
# profanity root turns into a multi-minute mute in the final video.
MAX_WORD_DURATION = 3.0

# Fed to Whisper as initial_prompt so it doesn't self-censor or quietly drop
# swearing it hears - without this it tends to substitute swear words with
# tamer lookalikes or silently omit them.
WHISPER_PROFANITY_PROMPT = (
    "Блядь, сука, нахуй, пиздец, ёбаный, мудак, гондон, пиздабол, охуенно, заебал."
)


class TranscriberError(RuntimeError):
    """Raised when speech recognition fails."""


def _sanitize_word(text: str, start: float, end: float, total_duration: float) -> Optional[Word]:
    """Validate and clamp one raw Whisper word, or reject it.

    Rejects empty text and words with a nonsensical or implausibly long
    duration (see MAX_WORD_DURATION). Clamps start/end into [0, total_duration]
    when that's known, so a mistimed word can never point past the real end
    of the audio.
    """
    text = text.strip()
    if not text:
        return None
    if end < start or end - start > MAX_WORD_DURATION:
        return None
    if total_duration > 0:
        start = min(start, total_duration)
        end = min(end, total_duration)
    return Word(text=text, start=start, end=end)


def transcribe(
    path: str,
    model_size: str = "small",
    language: str = "ru",
    progress_cb: Optional[Callable[[float], None]] = None,
    on_log: Optional[Callable[[str], None]] = None,
    cancel_event: Optional[threading.Event] = None,
) -> list[Word]:
    """Transcribe ``path`` and return every recognized word with timing.

    Progress is reported as a fraction in [0, 1] of the audio duration
    processed so far, based on the segments faster-whisper yields.
    """
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise TranscriberError(
            "Пакет faster-whisper не установлен. Выполните "
            "pip install -r requirements.txt"
        ) from exc

    if on_log is not None:
        on_log(
            f"Загрузка модели Whisper ({model_size})... Если модель ещё не "
            "скачана, первый запуск может занять несколько минут."
        )

    try:
        # device="cpu" is deliberate, not a placeholder: "auto" makes
        # ctranslate2 probe for a CUDA GPU and try to load cuBLAS/cuDNN even
        # on machines with no NVIDIA card or CUDA toolkit installed, which
        # fails with a cryptic "cublas64_12.dll not found" instead of just
        # running on CPU. This project has no GPU/CUDA install story, so CPU
        # is the only mode that's guaranteed to work on a plain machine.
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
    except Exception as exc:  # noqa: BLE001 - surface as a readable app error
        raise TranscriberError(f"Не удалось загрузить модель Whisper: {exc}") from exc

    try:
        segments, info = model.transcribe(
            path,
            language=language,
            word_timestamps=True,
            vad_filter=True,
            initial_prompt=WHISPER_PROFANITY_PROMPT,
        )
    except Exception as exc:  # noqa: BLE001
        raise TranscriberError(f"Ошибка распознавания речи: {exc}") from exc

    total_duration = info.duration if info and info.duration else 0.0

    words: list[Word] = []
    try:
        for segment in segments:
            if cancel_event is not None and cancel_event.is_set():
                raise CancelledError()

            for w in segment.words or []:
                word = _sanitize_word(w.word, w.start, w.end, total_duration)
                if word is not None:
                    words.append(word)

            if progress_cb is not None and total_duration > 0:
                progress_cb(min(1.0, max(0.0, segment.end / total_duration)))
    except CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise TranscriberError(f"Ошибка распознавания речи: {exc}") from exc

    return words
