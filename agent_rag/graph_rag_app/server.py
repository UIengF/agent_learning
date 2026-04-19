from __future__ import annotations

import uvicorn

from .api import create_app


def serve_fastapi(
    *,
    index_dir: str = "agent",
    host: str = "127.0.0.1",
    port: int = 8765,
    reload: bool = False,
) -> int:
    app = create_app(default_index_dir=index_dir)
    uvicorn.run(app, host=host, port=port, reload=reload)
    return 0
