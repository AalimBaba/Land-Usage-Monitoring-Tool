from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from land_usage.validation import validate_land_image


def certificate_image() -> Image.Image:
    img = Image.new("RGB", (600, 420), "white")
    draw = ImageDraw.Draw(img)
    draw.rectangle((24, 24, 576, 396), outline=(190, 150, 60), width=5)
    draw.text((210, 70), "CERTIFICATE", fill=(25, 25, 25))
    draw.text((120, 150), "This certifies that a person completed a course.", fill=(40, 40, 40))
    draw.text((150, 230), "Date: 2026-07-08    Signature: ________", fill=(40, 40, 40))
    for y in range(285, 350, 18):
        draw.line((120, y, 480, y), fill=(80, 80, 80), width=1)
    return img


def text_heavy_screenshot() -> Image.Image:
    img = Image.new("RGB", (640, 420), (248, 248, 248))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 640, 48), fill=(42, 45, 52))
    for idx in range(18):
        y = 72 + idx * 17
        draw.rectangle((42, y, 560 - (idx % 4) * 30, y + 7), fill=(30, 30, 30))
    draw.rectangle((420, 65, 610, 390), outline=(120, 120, 120), width=2)
    return img


def id_card_image() -> Image.Image:
    img = Image.new("RGB", (640, 400), (224, 229, 232))
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((58, 52, 582, 338), radius=18, fill=(242, 244, 238), outline=(65, 90, 110), width=4)
    draw.rectangle((58, 52, 582, 102), fill=(42, 92, 132))
    draw.rectangle((95, 135, 225, 280), fill=(205, 148, 115), outline=(85, 85, 85), width=2)
    draw.ellipse((125, 150, 195, 220), fill=(216, 160, 125))
    draw.rectangle((122, 215, 198, 278), fill=(65, 88, 135))
    for idx, width in enumerate([260, 310, 230, 290, 180, 260]):
        y = 132 + idx * 30
        draw.rectangle((265, y, 265 + width, y + 9), fill=(38, 44, 48))
    draw.rectangle((265, 282, 510, 305), outline=(90, 90, 90), width=2)
    return img


def portrait_image() -> Image.Image:
    img = Image.new("RGB", (360, 360), (80, 105, 130))
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 55, 260, 235), fill=(205, 148, 115))
    draw.ellipse((135, 120, 150, 135), fill=(35, 25, 20))
    draw.ellipse((210, 120, 225, 135), fill=(35, 25, 20))
    draw.arc((150, 150, 210, 205), 20, 160, fill=(120, 55, 55), width=4)
    draw.rectangle((120, 235, 240, 360), fill=(45, 70, 120))
    return img


def blank_image() -> Image.Image:
    return Image.new("RGB", (256, 256), (245, 245, 245))


def satellite_like_image(seed: int = 4) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:, :95] = (63, 130, 78)
    arr[:, 95:150] = (70, 125, 175)
    arr[:120, 150:] = (172, 148, 74)
    arr[120:, 150:] = (139, 132, 118)
    arr[118:138, :] = (118, 118, 112)
    arr[:, 142:158] = (115, 115, 112)
    noise = rng.normal(0, 18, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def aerial_land_image(seed: int = 9) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((256, 256, 3), (85, 140, 78), dtype=np.uint8)
    arr[30:95, 35:120] = (187, 169, 82)
    arr[150:225, 60:220] = (125, 124, 115)
    arr[:, 125:138] = (65, 105, 160)
    arr[105:120, :] = (150, 145, 130)
    noise = rng.normal(0, 16, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def ambiguous_object_graphic() -> Image.Image:
    img = Image.new("RGB", (300, 300), (210, 210, 215))
    draw = ImageDraw.Draw(img)
    draw.ellipse((80, 70, 220, 210), fill=(190, 20, 30))
    draw.rectangle((120, 210, 180, 270), fill=(40, 40, 45))
    return img


def test_certificate_document_is_rejected_before_segmentation():
    result = validate_land_image(certificate_image())

    assert result.image_relevance == "Rejected"
    assert result.segmentation == "Not run"


def test_text_heavy_screenshot_is_rejected():
    result = validate_land_image(text_heavy_screenshot())

    assert result.image_relevance == "Rejected"


def test_id_card_document_is_rejected():
    result = validate_land_image(id_card_image())

    assert result.image_relevance == "Rejected"
    assert result.segmentation == "Not run"


def test_portrait_is_rejected():
    result = validate_land_image(portrait_image())

    assert result.image_relevance == "Rejected"


def test_blank_image_is_rejected():
    result = validate_land_image(blank_image())

    assert result.image_relevance == "Rejected"


def test_valid_satellite_like_image_is_accepted():
    result = validate_land_image(satellite_like_image())

    assert result.image_relevance == "Suitable"
    assert result.segmentation == "Run"


def test_valid_aerial_land_image_is_accepted():
    result = validate_land_image(aerial_land_image())

    assert result.image_relevance == "Suitable"
    assert result.segmentation == "Run"


def test_ambiguous_non_land_graphic_warns_or_rejects():
    result = validate_land_image(ambiguous_object_graphic())

    assert result.image_relevance in {"Uncertain", "Rejected"}
    assert result.segmentation == "Not run"
