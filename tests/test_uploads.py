from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from land_usage.uploads import UploadImageError, open_uploaded_image


def test_open_uploaded_image_rejects_invalid_bytes_safely():
    with pytest.raises(UploadImageError):
        open_uploaded_image(BytesIO(b"not an image"))


def test_open_uploaded_image_returns_rgb_image():
    buffer = BytesIO()
    Image.new("RGBA", (12, 12), (20, 40, 60, 255)).save(buffer, format="PNG")
    buffer.seek(0)

    image = open_uploaded_image(buffer)

    assert image.mode == "RGB"
    assert image.size == (12, 12)
