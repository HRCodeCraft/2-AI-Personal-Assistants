"""Entry point — launches the combined AI Personal Assistants app."""

from __future__ import annotations

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))

from apps.combined import create_combined_app, _THEME, CUSTOM_CSS


def main() -> None:
    port = int(os.getenv("GRADIO_SERVER_PORT", "7860"))
    share = os.getenv("GRADIO_SHARE", "false").lower() == "true"

    app = create_combined_app()
    app.queue()
    app.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=share,
        theme=_THEME,
        css=CUSTOM_CSS,
    )


if __name__ == "__main__":
    main()
