"""Serve LLM2Jev System One API plus the browser test page."""

from __future__ import annotations

import os
from pathlib import Path

import uvicorn
from fastapi.responses import HTMLResponse
from llm2jev.server.transformers_server import create_app

TEST_HTML = Path(__file__).parent / "test.html"

_model_path = os.environ.get("MODEL_PATH", "").strip()
if not _model_path:
    raise SystemExit("MODEL_PATH is required")

_multimodal = os.environ.get("MULTIMODAL", "1").lower() not in ("0", "false", "no")

app = create_app(
    _model_path,
    served_model_name=os.environ.get("SERVED_MODEL_NAME") or None,
    api_key=os.environ.get("LLM2JEV_API_KEY") or None,
    multimodal=_multimodal,
)


@app.get("/test", response_class=HTMLResponse)
async def test_page() -> HTMLResponse:
    return HTMLResponse(TEST_HTML.read_text(encoding="utf-8"))


if __name__ == "__main__":
    uvicorn.run(
        app,
        host=os.environ.get("HOST", "0.0.0.0"),
        port=int(os.environ.get("PORT", "30000")),
    )
