"""Entry point: SilentCutter desktop app."""

from __future__ import annotations

import sys


def main() -> None:
    try:
        from .ui import App
    except ImportError as exc:
        print(
            "Не удалось импортировать зависимости UI. Убедитесь, что установлены "
            f"пакеты из requirements.txt (pip install -r requirements.txt).\n{exc}",
            file=sys.stderr,
        )
        sys.exit(1)

    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()
