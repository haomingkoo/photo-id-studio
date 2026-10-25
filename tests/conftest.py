from __future__ import annotations

import os

# rembg downloads a ~170MB model on first use; the MediaPipe fallback is enough for these tests.
os.environ.setdefault("PHOTO_API_ENABLE_REMBG", "0")
