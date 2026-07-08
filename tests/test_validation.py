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


def photographed_id_card_image(seed: int = 51) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((260, 520, 3), (214, 218, 210), dtype=np.uint8)
    img = Image.fromarray(arr)
    draw = ImageDraw.Draw(img)
    draw.rounded_rectangle((8, 8, 512, 252), radius=12, fill=(218, 222, 216), outline=(116, 124, 116), width=3)
    draw.rectangle((28, 58, 150, 185), fill=(150, 156, 154), outline=(95, 95, 95), width=2)
    draw.ellipse((62, 76, 116, 134), fill=(126, 128, 126))
    draw.rectangle((52, 135, 126, 184), fill=(70, 82, 92))
    for idx, width in enumerate([255, 215, 300, 285, 190, 245, 225]):
        y = 42 + idx * 24
        draw.rectangle((178, y, 178 + width, y + 7), fill=(45, 48, 46))
    for x in range(420, 500, 7):
        shade = 30 if (x // 7) % 2 else 210
        draw.rectangle((x, 88, x + 3, 218), fill=(shade, shade, shade))
    noise = rng.normal(0, 9, (260, 520, 3))
    return Image.fromarray(np.clip(np.asarray(img).astype(np.int16) + noise, 0, 255).astype(np.uint8))


def portrait_image() -> Image.Image:
    img = Image.new("RGB", (360, 360), (80, 105, 130))
    draw = ImageDraw.Draw(img)
    draw.ellipse((100, 55, 260, 235), fill=(205, 148, 115))
    draw.ellipse((135, 120, 150, 135), fill=(35, 25, 20))
    draw.ellipse((210, 120, 225, 135), fill=(35, 25, 20))
    draw.arc((150, 150, 210, 205), 20, 160, fill=(120, 55, 55), width=4)
    draw.rectangle((120, 235, 240, 360), fill=(45, 70, 120))
    return img


def indoor_room_image(seed: int = 42) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((256, 256, 3), (184, 165, 140), dtype=np.uint8)
    arr[:90, :] = (206, 190, 168)
    arr[90:180, :135] = (176, 158, 132)
    arr[90:180, 135:] = (130, 98, 76)
    arr[180:, :] = (118, 92, 70)
    arr[120:176, 150:235] = (84, 62, 48)
    arr[132:168, 25:115] = (222, 222, 210)
    arr[176:184, :] = (70, 58, 50)
    arr[:, 132:138] = (92, 76, 64)
    noise = rng.normal(0, 7, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


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


def dense_urban_aerial_image(seed: int = 12) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((256, 256, 3), (118, 122, 118), dtype=np.uint8)
    for y in range(10, 246, 28):
        arr[y : y + 5, :] = (72, 75, 78)
    for x in range(14, 246, 31):
        arr[:, x : x + 5] = (70, 74, 77)
    for y in range(18, 230, 32):
        for x in range(22, 230, 35):
            colour = rng.choice([(165, 100, 78), (154, 145, 125), (92, 112, 85), (176, 168, 92)])
            arr[y : y + 15, x : x + 20] = colour
    arr[165:205, 20:75] = (64, 132, 82)
    arr[30:62, 185:240] = (55, 112, 165)
    noise = rng.normal(0, 17, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def rural_farmland_aerial_image(seed: int = 15) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:90, :130] = (85, 146, 75)
    arr[:90, 130:] = (192, 174, 88)
    arr[90:175, :110] = (128, 104, 67)
    arr[90:175, 110:] = (96, 156, 86)
    arr[175:, :] = (178, 158, 76)
    arr[:, 122:132] = (112, 110, 102)
    arr[86:96, :] = (105, 103, 98)
    noise = rng.normal(0, 14, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def forest_satellite_image(seed: int = 19) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.full((256, 256, 3), (42, 112, 62), dtype=np.int16)
    canopy = rng.normal(0, 28, base.shape)
    arr = np.clip(base + canopy, 0, 255).astype(np.uint8)
    arr[90:110, :] = (92, 98, 88)
    arr[:, 170:180] = (82, 90, 86)
    return Image.fromarray(arr)


def coastline_aerial_image(seed: int = 22) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.full((256, 256, 3), (58, 123, 178), dtype=np.uint8)
    arr[:, :112] = (78, 142, 82)
    arr[120:, :150] = (190, 172, 95)
    for offset in range(0, 256):
        x = 105 + int(12 * np.sin(offset / 18))
        arr[offset, max(0, x - 3) : min(256, x + 5)] = (222, 206, 142)
    noise = rng.normal(0, 16, arr.shape)
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


def city_map_image() -> Image.Image:
    img = Image.new("RGB", (360, 360), (235, 232, 218))
    draw = ImageDraw.Draw(img)
    for x in range(20, 350, 35):
        draw.line((x, 0, x + 30, 360), fill=(235, 160, 80), width=3)
    for y in range(25, 350, 42):
        draw.line((0, y, 360, y + 20), fill=(95, 115, 180), width=3)
    draw.rectangle((40, 220, 125, 300), fill=(100, 165, 100))
    return img


def ground_landscape_photo(seed: int = 31) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = np.zeros((256, 256, 3), dtype=np.uint8)
    arr[:105, :] = (110, 170, 220)
    arr[105:150, :] = (70, 140, 72)
    arr[150:, :] = (120, 92, 55)
    arr[80:130, 40:95] = (42, 105, 48)
    noise = rng.normal(0, 12, arr.shape)
    return Image.fromarray(np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8))


def screenshot_with_satellite_image(seed: int = 36) -> Image.Image:
    img = Image.new("RGB", (520, 360), (245, 246, 247))
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, 520, 48), fill=(45, 48, 55))
    draw.rectangle((20, 14, 245, 32), fill=(225, 225, 225))
    sat = dense_urban_aerial_image(seed).resize((430, 260), Image.Resampling.BILINEAR)
    img.paste(sat, (45, 78))
    draw.rectangle((45, 78, 475, 338), outline=(40, 40, 40), width=2)
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


def test_photographed_id_card_is_rejected():
    result = validate_land_image(photographed_id_card_image())

    assert result.image_relevance == "Rejected"
    assert result.segmentation == "Not run"


def test_portrait_is_rejected():
    result = validate_land_image(portrait_image())

    assert result.image_relevance == "Rejected"


def test_indoor_room_photo_is_not_suitable():
    result = validate_land_image(indoor_room_image())

    assert result.image_relevance in {"Rejected", "Uncertain"}
    assert result.segmentation == "Not run"


def test_blank_image_is_rejected():
    result = validate_land_image(blank_image())

    assert result.image_relevance == "Rejected"


def test_valid_satellite_like_image_is_accepted():
    result = validate_land_image(satellite_like_image())

    assert result.image_relevance == "Suitable"
    assert result.segmentation == "Run"


def test_dense_urban_aerial_image_is_accepted():
    result = validate_land_image(dense_urban_aerial_image())

    assert result.image_relevance == "Suitable"
    assert result.segmentation == "Run"


def test_rural_farmland_aerial_image_is_accepted():
    result = validate_land_image(rural_farmland_aerial_image())

    assert result.image_relevance == "Suitable"


def test_forest_satellite_image_is_accepted():
    result = validate_land_image(forest_satellite_image())

    assert result.image_relevance == "Suitable"


def test_water_coastline_aerial_image_is_accepted():
    result = validate_land_image(coastline_aerial_image())

    assert result.image_relevance == "Suitable"


def test_valid_aerial_land_image_is_accepted():
    result = validate_land_image(aerial_land_image())

    assert result.image_relevance == "Suitable"
    assert result.segmentation == "Run"


def test_ambiguous_non_land_graphic_warns_or_rejects():
    result = validate_land_image(ambiguous_object_graphic())

    assert result.image_relevance in {"Uncertain", "Rejected"}
    assert result.segmentation == "Not run"


def test_city_map_is_uncertain():
    result = validate_land_image(city_map_image())

    assert result.image_relevance == "Uncertain"


def test_ground_level_landscape_is_uncertain():
    result = validate_land_image(ground_landscape_photo())

    assert result.image_relevance == "Uncertain"


def test_screenshot_containing_satellite_image_is_uncertain():
    result = validate_land_image(screenshot_with_satellite_image())

    assert result.image_relevance == "Uncertain"
