from __future__ import annotations

from typing import BinaryIO

from PIL import Image, UnidentifiedImageError


class UploadImageError(ValueError):
    """Raised when an uploaded file cannot be decoded as an RGB image."""


def open_uploaded_image(uploaded_file: BinaryIO) -> Image.Image:
    try:
        image = Image.open(uploaded_file)
        image.load()
    except (OSError, UnidentifiedImageError) as exc:
        raise UploadImageError("That file is not a valid image. Please upload a PNG or JPEG image.") from exc
    return image.convert("RGB")
