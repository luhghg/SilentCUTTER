"""Entry point for the packaged (PyInstaller) Windows build.

silence_cutter/main.py uses a relative import (`from .ui import App`), which
only works when run as part of the package (`python -m silence_cutter.main`).
PyInstaller's bootloader runs the entry script as a plain top-level module,
so relative imports there would fail - this script does the equivalent
absolute import instead.
"""

from silence_cutter.ui import App


def main() -> None:
    App().mainloop()


if __name__ == "__main__":
    main()
