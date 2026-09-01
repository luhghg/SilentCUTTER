"""The (single) application window."""

from __future__ import annotations

import threading
from pathlib import Path
from tkinter import filedialog
from typing import Optional

import customtkinter as ctk

from . import censor, cutter, detector, ffprobe, matcher, transcriber

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD

    _DND_AVAILABLE = True
except ImportError:
    _DND_AVAILABLE = False

ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

if _DND_AVAILABLE:
    _BaseWindow = type("_BaseWindow", (ctk.CTk, TkinterDnD.DnDWrapper), {})
else:
    _BaseWindow = ctk.CTk


class App(_BaseWindow):
    def __init__(self) -> None:
        super().__init__()
        if _DND_AVAILABLE:
            try:
                self.TkdndVersion = TkinterDnD._require(self)
            except Exception:
                self._dnd_ready = False
            else:
                self._dnd_ready = True
        else:
            self._dnd_ready = False

        self.title("SilentCutter — вырезание тишины из видео")
        self.geometry("760x720")
        self.minsize(680, 640)

        self.video_path: str | None = None
        self.cancel_event = threading.Event()
        self.current_process = None
        self._process_lock = threading.Lock()
        self._processing = False

        self._build_widgets()
        self._check_ffmpeg()

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------

    def _build_widgets(self) -> None:
        pad = {"padx": 12, "pady": 6}

        # --- file selection ---
        file_frame = ctk.CTkFrame(self)
        file_frame.pack(fill="x", **pad)

        self.path_entry = ctk.CTkEntry(
            file_frame, placeholder_text="Файл не выбран (можно перетащить видео в окно)"
        )
        self.path_entry.pack(side="left", fill="x", expand=True, padx=(10, 6), pady=10)
        self.path_entry.configure(state="disabled")

        self.browse_button = ctk.CTkButton(
            file_frame, text="Выбрать видео", width=140, command=self._on_browse
        )
        self.browse_button.pack(side="right", padx=(6, 10), pady=10)

        if self._dnd_ready:
            self.drop_target_register(DND_FILES)
            self.dnd_bind("<<Drop>>", self._on_drop)

        # --- parameters ---
        params_frame = ctk.CTkFrame(self)
        params_frame.pack(fill="x", **pad)

        self.noise_var = ctk.DoubleVar(value=-40)
        self.min_silence_var = ctk.DoubleVar(value=0.5)
        self.padding_var = ctk.DoubleVar(value=0.15)

        self._build_slider_row(
            params_frame, "Порог тишины, dB", self.noise_var, -60, -20,
            fmt="{:.0f} dB",
        )
        self._build_slider_row(
            params_frame, "Мин. длина тишины, сек", self.min_silence_var, 0.1, 3.0,
            fmt="{:.2f} с",
        )
        self._build_slider_row(
            params_frame, "Padding, сек", self.padding_var, 0.0, 1.0,
            fmt="{:.2f} с",
        )

        self.export_tc_var = ctk.BooleanVar(value=False)
        self.export_tc_check = ctk.CTkCheckBox(
            params_frame, text="Экспортировать таймкоды (.txt)", variable=self.export_tc_var
        )
        self.export_tc_check.pack(anchor="w", padx=14, pady=(4, 10))

        # --- profanity censor ---
        censor_frame = ctk.CTkFrame(self)
        censor_frame.pack(fill="x", **pad)

        self.censor_var = ctk.BooleanVar(value=False)
        self.censor_check = ctk.CTkCheckBox(
            censor_frame,
            text="Заглушать мат",
            variable=self.censor_var,
            command=self._on_censor_toggle,
        )
        self.censor_check.pack(anchor="w", padx=14, pady=(10, 6))

        model_row = ctk.CTkFrame(censor_frame, fg_color="transparent")
        model_row.pack(fill="x", padx=10, pady=4)
        ctk.CTkLabel(model_row, text="Модель Whisper", width=170, anchor="w").pack(side="left")
        self.model_size_var = ctk.StringVar(value="small")
        self.model_size_menu = ctk.CTkOptionMenu(
            model_row,
            values=["tiny", "base", "small", "medium"],
            variable=self.model_size_var,
            state="disabled",
        )
        self.model_size_menu.pack(side="left", padx=10)

        self.lead_var = ctk.DoubleVar(value=matcher.DEFAULT_LEAD)
        self.lead_slider = self._build_slider_row(
            censor_frame,
            "Сколько слышно от слова до обрыва, сек",
            self.lead_var,
            0.0,
            0.2,
            fmt="{:.2f} с",
        )
        self.lead_slider.configure(state="disabled")

        # --- actions ---
        actions_frame = ctk.CTkFrame(self)
        actions_frame.pack(fill="x", **pad)

        self.process_button = ctk.CTkButton(
            actions_frame, text="Обработать", command=self._on_process, height=36
        )
        self.process_button.pack(side="left", padx=(10, 6), pady=10, fill="x", expand=True)

        self.cancel_button = ctk.CTkButton(
            actions_frame,
            text="Отмена",
            command=self._on_cancel,
            height=36,
            fg_color="#8b2c2c",
            hover_color="#6e2222",
            state="disabled",
        )
        self.cancel_button.pack(side="left", padx=(6, 10), pady=10, fill="x", expand=True)

        # --- progress ---
        progress_frame = ctk.CTkFrame(self)
        progress_frame.pack(fill="x", **pad)

        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.set(0)
        self.progress_bar.pack(fill="x", padx=10, pady=(10, 2))

        self.progress_label = ctk.CTkLabel(progress_frame, text="Готов к работе")
        self.progress_label.pack(anchor="w", padx=10, pady=(0, 10))

        # --- log ---
        log_frame = ctk.CTkFrame(self)
        log_frame.pack(fill="both", expand=True, **pad)

        ctk.CTkLabel(log_frame, text="Лог").pack(anchor="w", padx=10, pady=(8, 0))
        self.log_box = ctk.CTkTextbox(log_frame, wrap="word", state="disabled")
        self.log_box.pack(fill="both", expand=True, padx=10, pady=(4, 10))

    def _build_slider_row(self, parent, label, var, lo, hi, fmt):
        row = ctk.CTkFrame(parent, fg_color="transparent")
        row.pack(fill="x", padx=10, pady=4)

        name_label = ctk.CTkLabel(row, text=label, width=170, anchor="w")
        name_label.pack(side="left")

        value_label = ctk.CTkLabel(row, text=fmt.format(var.get()), width=70, anchor="e")
        value_label.pack(side="right")

        def on_change(value):
            value_label.configure(text=fmt.format(float(value)))

        slider = ctk.CTkSlider(row, from_=lo, to=hi, variable=var, command=on_change)
        slider.pack(side="left", fill="x", expand=True, padx=10)
        return slider

    def _on_censor_toggle(self) -> None:
        state = "normal" if self.censor_var.get() else "disabled"
        self.model_size_menu.configure(state=state)
        self.lead_slider.configure(state=state)

    # ------------------------------------------------------------------
    # Startup checks
    # ------------------------------------------------------------------

    def _check_ffmpeg(self) -> None:
        missing = ffprobe.check_tools_available()
        if missing:
            names = " и ".join(missing)
            self._append_log(
                f"Не найден(ы) {names} в PATH. Установите ffmpeg и добавьте его в PATH, "
                "затем перезапустите приложение (см. README.md)."
            )
            self.process_button.configure(state="disabled")
        else:
            self._append_log("ffmpeg и ffprobe найдены. Готов к работе.")
        if not self._dnd_ready:
            self._append_log(
                "Перетаскивание файлов недоступно в этой сборке — используйте кнопку "
                "«Выбрать видео»."
            )

    # ------------------------------------------------------------------
    # File selection
    # ------------------------------------------------------------------

    def _on_browse(self) -> None:
        path = filedialog.askopenfilename(
            title="Выберите видео",
            filetypes=[
                ("Видео", "*.mp4 *.mov *.mkv *.avi *.webm *.m4v"),
                ("Все файлы", "*.*"),
            ],
        )
        if path:
            self._set_video_path(path)

    def _on_drop(self, event) -> None:
        raw = event.data
        path = self._parse_dnd_path(raw)
        if path:
            self._set_video_path(path)

    @staticmethod
    def _parse_dnd_path(raw: str) -> str | None:
        raw = raw.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        return raw or None

    def _set_video_path(self, path: str) -> None:
        self.video_path = path
        self.path_entry.configure(state="normal")
        self.path_entry.delete(0, "end")
        self.path_entry.insert(0, path)
        self.path_entry.configure(state="disabled")
        self._append_log(f"Выбран файл: {path}")

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def _on_process(self) -> None:
        if self._processing:
            return
        if not self.video_path:
            self._append_log("Сначала выберите видеофайл.")
            return
        if not Path(self.video_path).exists():
            self._append_log("Выбранный файл не найден на диске.")
            return

        self.cancel_event = threading.Event()
        self.current_process = None
        self._processing = True
        self.process_button.configure(state="disabled")
        self.browse_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self._set_progress(0.0, "Запуск...")

        params = dict(
            input_path=self.video_path,
            noise_db=self.noise_var.get(),
            min_silence=self.min_silence_var.get(),
            padding=self.padding_var.get(),
            export_tc=self.export_tc_var.get(),
            censor_enabled=self.censor_var.get(),
            model_size=self.model_size_var.get(),
            lead=self.lead_var.get(),
        )
        thread = threading.Thread(target=self._process_worker, kwargs=params, daemon=True)
        thread.start()

    def _on_cancel(self) -> None:
        if not self._processing:
            return
        self.cancel_event.set()
        with self._process_lock:
            if self.current_process is not None:
                try:
                    self.current_process.kill()
                except Exception:
                    pass
        self._append_log("Отмена запрошена...")
        self.cancel_button.configure(state="disabled")

    def _register_process(self, proc) -> None:
        with self._process_lock:
            self.current_process = proc
            if self.cancel_event.is_set():
                try:
                    proc.kill()
                except Exception:
                    pass

    def _process_worker(
        self,
        input_path: str,
        noise_db: float,
        min_silence: float,
        padding: float,
        export_tc: bool,
        censor_enabled: bool,
        model_size: str,
        lead: float,
    ) -> None:
        min_clip = 0.2
        min_gap = 0.1
        temp_cut_path: str | None = None
        try:
            self._append_log("Чтение метаданных (ffprobe)...")
            info = ffprobe.get_video_info(input_path)
            self._append_log(
                f"Длительность: {info.duration:.2f} с, FPS: {info.fps:.3f}, "
                f"разрешение: {info.width}x{info.height}"
            )
            if not info.has_audio:
                raise cutter.CutterError(
                    "В видео нет аудиодорожки — нечего анализировать на тишину"
                )

            if censor_enabled:
                off_detect, w_detect = 0.00, 0.25
                off_cut, w_cut = 0.25, 0.25
                off_trans, w_trans = 0.50, 0.30
                off_censor, w_censor = 0.80, 0.20
            else:
                off_detect, w_detect = 0.00, 0.40
                off_cut, w_cut = 0.40, 0.60

            self._append_log("Анализ тишины...")
            stderr_text = cutter.run_silence_detection(
                input_path,
                noise_db,
                min_silence,
                info.duration,
                on_progress=lambda f: self._set_progress(
                    off_detect + f * w_detect, "Анализ тишины..."
                ),
                on_log=self._append_log,
                register_process=self._register_process,
                cancel_event=self.cancel_event,
            )

            silences = detector.parse_silencedetect_output(stderr_text, info.duration)
            self._append_log(f"Найдено участков тишины: {len(silences)}")

            segments = detector.compute_speech_segments(
                silences, info.duration, padding=padding, min_clip=min_clip, min_gap=min_gap
            )
            if not segments:
                raise cutter.CutterError(
                    "После анализа не осталось речи для сборки — попробуйте снизить "
                    "порог тишины или минимальную длину тишины"
                )

            kept = detector.total_kept_duration(segments)
            self._append_log(
                f"Сегментов речи: {len(segments)}, останется {kept:.2f} с из "
                f"{info.duration:.2f} с"
            )

            final_output_path = self._make_output_path(input_path)
            cut_target = (
                self._make_temp_cut_path(input_path) if censor_enabled else final_output_path
            )
            if censor_enabled:
                temp_cut_path = cut_target

            self._append_log(f"Резка и склейка -> {cut_target}")
            cutter.run_cut(
                input_path,
                cut_target,
                segments,
                info.fps,
                info.has_audio,
                on_progress=lambda f: self._set_progress(off_cut + f * w_cut, "Сборка видео..."),
                on_log=self._append_log,
                register_process=self._register_process,
                cancel_event=self.cancel_event,
            )

            if export_tc:
                tc_path = self._export_timecodes(segments, input_path)
                self._append_log(f"Таймкоды сохранены: {tc_path}")

            censored_words: Optional[int] = None
            if censor_enabled:
                self._append_log(f"Распознавание речи (модель {model_size})...")
                words = transcriber.transcribe(
                    cut_target,
                    model_size=model_size,
                    language="ru",
                    progress_cb=lambda f: self._set_progress(
                        off_trans + f * w_trans, "Распознавание речи..."
                    ),
                    on_log=self._append_log,
                    cancel_event=self.cancel_event,
                )
                self._append_log(f"Распознано слов: {len(words)}")

                roots = matcher.load_word_list(matcher.DEFAULT_PROFANITY_PATH)
                whitelist = matcher.load_word_list(matcher.DEFAULT_WHITELIST_PATH)
                profane_words = matcher.filter_profane_words(words, roots, whitelist)
                censored_words = len(profane_words)
                intervals = matcher.find_profanity(
                    words, roots, whitelist, padding=matcher.DEFAULT_PADDING, lead=lead
                )
                self._append_log(f"Найдено матерных слов: {censored_words}")

                self._set_progress(off_censor, "Заглушение...")
                censor.censor(
                    cut_target,
                    intervals,
                    final_output_path,
                    duration=kept,
                    on_progress=lambda f: self._set_progress(
                        off_censor + f * w_censor, "Заглушение..."
                    ),
                    on_log=self._append_log,
                    register_process=self._register_process,
                    cancel_event=self.cancel_event,
                )
                temp_cut_path = None  # consumed (moved or deleted) by censor() on success

            self._set_progress(1.0, "Готово")
            self._show_stats(info.duration, kept, len(segments), final_output_path, censored_words)
            self._append_log("Готово!")
        except cutter.CancelledError:
            self._append_log("Обработка отменена пользователем.")
            self._set_progress(0.0, "Отменено")
        except (
            ffprobe.FFToolNotFoundError,
            ffprobe.FFProbeError,
            cutter.CutterError,
            transcriber.TranscriberError,
        ) as exc:
            self._append_log(f"Ошибка: {exc}")
            self._set_progress(0.0, "Ошибка")
        except Exception as exc:  # noqa: BLE001 - surface unexpected errors in the log, not a traceback
            self._append_log(f"Непредвиденная ошибка: {exc}")
            self._set_progress(0.0, "Ошибка")
        finally:
            if temp_cut_path is not None:
                Path(temp_cut_path).unlink(missing_ok=True)
            self.after(0, self._on_processing_finished)

    def _on_processing_finished(self) -> None:
        self._processing = False
        self.process_button.configure(state="normal")
        self.browse_button.configure(state="normal")
        self.cancel_button.configure(state="disabled")

    @staticmethod
    def _make_output_path(input_path: str) -> str:
        p = Path(input_path)
        return str(p.with_name(f"{p.stem}_cut.mp4"))

    @staticmethod
    def _make_temp_cut_path(input_path: str) -> str:
        p = Path(input_path)
        return str(p.with_name(f"{p.stem}_tmp_precensor.mp4"))

    @staticmethod
    def _export_timecodes(segments: list[detector.Segment], input_path: str) -> str:
        p = Path(input_path)
        tc_path = p.with_name(f"{p.stem}_timecodes.txt")
        lines = [
            f"{detector.format_timecode(s.start)} - {detector.format_timecode(s.end)}"
            for s in segments
        ]
        tc_path.write_text("\n".join(lines), encoding="utf-8")
        return str(tc_path)

    def _show_stats(
        self,
        original: float,
        kept: float,
        n_segments: int,
        output_path: str,
        censored_words: Optional[int] = None,
    ) -> None:
        cut_amount = original - kept
        cut_pct = (cut_amount / original * 100) if original > 0 else 0.0
        lines = [
            "--- Статистика ---",
            f"Исходная длительность: {original:.2f} с",
            f"Итоговая длительность: {kept:.2f} с",
            f"Вырезано: {cut_amount:.2f} с ({cut_pct:.1f}%)",
            f"Склеек (сегментов речи): {n_segments}",
        ]
        if censored_words is not None:
            lines.append(f"Слов заглушено: {censored_words}")
        lines.append(f"Результат: {output_path}")
        self._append_log("\n".join(lines))

    # ------------------------------------------------------------------
    # Thread-safe UI helpers
    # ------------------------------------------------------------------

    def _append_log(self, message: str) -> None:
        self.after(0, self._append_log_ui, message)

    def _append_log_ui(self, message: str) -> None:
        self.log_box.configure(state="normal")
        self.log_box.insert("end", message + "\n")
        self.log_box.see("end")
        self.log_box.configure(state="disabled")

    def _set_progress(self, fraction: float, status: str | None = None) -> None:
        self.after(0, self._set_progress_ui, fraction, status)

    def _set_progress_ui(self, fraction: float, status: str | None) -> None:
        self.progress_bar.set(max(0.0, min(1.0, fraction)))
        if status is not None:
            self.progress_label.configure(text=f"{status} ({fraction * 100:.0f}%)")
