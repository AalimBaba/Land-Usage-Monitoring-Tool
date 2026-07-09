from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

from PIL import Image

from land_usage.uploads import UploadImageError, open_uploaded_image
from land_usage.validation import validate_land_image


SAMPLES_ROOT = Path("tests") / "validation_samples"
VALID_DIR = SAMPLES_ROOT / "valid"
INVALID_DIR = SAMPLES_ROOT / "invalid"
REPORT_PATH = Path("validation_report.json")
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    category: str
    expected: str
    actual: str
    segmentation: str
    passed: bool
    reason: str
    source: str


def _load_builtin_samples() -> list[tuple[str, str, str, Callable[[], Image.Image]]]:
    from tests.test_validation import (
        aerial_land_image,
        blank_image,
        certificate_image,
        coastline_aerial_image,
        dense_urban_aerial_image,
        forest_satellite_image,
        id_card_image,
        indoor_room_image,
        photographed_id_card_image,
        portrait_image,
        rural_farmland_aerial_image,
        satellite_like_image,
        text_heavy_screenshot,
    )

    return [
        ("aerial_city", "valid", "Suitable", dense_urban_aerial_image),
        ("satellite_urban", "valid", "Suitable", satellite_like_image),
        ("farmland_rural", "valid", "Suitable", rural_farmland_aerial_image),
        ("forest_vegetation", "valid", "Suitable", forest_satellite_image),
        ("water_coastline", "valid", "Suitable", coastline_aerial_image),
        ("mixed_land_cover", "valid", "Suitable", aerial_land_image),
        ("id_card", "invalid", "Not Suitable", id_card_image),
        ("photographed_id_card", "invalid", "Not Suitable", photographed_id_card_image),
        ("certificate", "invalid", "Not Suitable", certificate_image),
        ("document_screenshot", "invalid", "Not Suitable", text_heavy_screenshot),
        ("indoor_room", "invalid", "Not Suitable", indoor_room_image),
        ("portrait_selfie", "invalid", "Not Suitable", portrait_image),
        ("blank_simple_graphic", "invalid", "Not Suitable", blank_image),
    ]


def expected_matches(expected: str, actual: str) -> bool:
    if expected == "Suitable":
        return actual == "Suitable"
    if expected == "Rejected":
        return actual == "Rejected"
    if expected == "Uncertain":
        return actual == "Uncertain"
    if expected == "Not Suitable":
        return actual != "Suitable"
    return False


def run_case(name: str, category: str, expected: str, image: Image.Image, source: str) -> BenchmarkCase:
    result = validate_land_image(image)
    return BenchmarkCase(
        name=name,
        category=category,
        expected=expected,
        actual=result.image_relevance,
        segmentation=result.segmentation,
        passed=expected_matches(expected, result.image_relevance),
        reason=" ".join(result.reasons),
        source=source,
    )


def iter_local_images(samples_root: Path) -> list[tuple[str, str, str, Path]]:
    cases: list[tuple[str, str, str, Path]] = []
    for category, expected in (("valid", "Suitable"), ("invalid", "Not Suitable")):
        folder = samples_root / category
        if not folder.exists():
            continue
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                cases.append((path.stem, category, expected, path))
    return cases


def run_benchmark(samples_root: Path = SAMPLES_ROOT, include_builtin: bool = True) -> dict:
    cases: list[BenchmarkCase] = []

    if include_builtin:
        for name, category, expected, factory in _load_builtin_samples():
            cases.append(run_case(name, category, expected, factory(), "built-in synthetic"))

    for name, category, expected, path in iter_local_images(samples_root):
        try:
            with path.open("rb") as handle:
                image = open_uploaded_image(handle)
            cases.append(run_case(name, category, expected, image, str(path)))
        except UploadImageError as exc:
            cases.append(
                BenchmarkCase(
                    name=name,
                    category=category,
                    expected=expected,
                    actual="Invalid image",
                    segmentation="Not run",
                    passed=category == "invalid",
                    reason=str(exc),
                    source=str(path),
                )
            )

    total = len(cases)
    passed = sum(case.passed for case in cases)
    false_accepts = [case for case in cases if case.expected != "Suitable" and case.actual == "Suitable"]
    false_rejects = [case for case in cases if case.expected == "Suitable" and case.actual != "Suitable"]

    return {
        "summary": {
            "total_cases": total,
            "passed": passed,
            "failed": total - passed,
            "validation_accuracy": round(passed / total, 4) if total else None,
            "false_accepts": len(false_accepts),
            "false_rejects": len(false_rejects),
            "note": "Validation accuracy is input-relevance gate accuracy, not U-Net segmentation accuracy.",
        },
        "false_accepts": [asdict(case) for case in false_accepts],
        "false_rejects": [asdict(case) for case in false_rejects],
        "cases": [asdict(case) for case in cases],
    }


def print_table(report: dict) -> None:
    rows = report["cases"]
    headers = ("Name", "Category", "Expected", "Actual", "Segmentation", "Pass")
    table_rows = [
        [row["name"], row["category"], row["expected"], row["actual"], row["segmentation"], "yes" if row["passed"] else "no"]
        for row in rows
    ]
    widths = [
        max(len(str(value)) for value in [headers[idx]] + [row[idx] for row in table_rows])
        for idx in range(len(headers))
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for values in table_rows:
        print(" | ".join(str(value).ljust(widths[idx]) for idx, value in enumerate(values)))
    print()
    summary = report["summary"]
    print(f"Validation accuracy: {summary['validation_accuracy']}")
    print(f"False accepts: {summary['false_accepts']}")
    print(f"False rejects: {summary['false_rejects']}")
    print("This is validator accuracy only, not segmentation/model accuracy.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the land-image input validation benchmark.")
    parser.add_argument("--samples-root", type=Path, default=SAMPLES_ROOT)
    parser.add_argument("--output", type=Path, default=REPORT_PATH)
    parser.add_argument("--no-builtins", action="store_true", help="Only use images from tests/validation_samples.")
    args = parser.parse_args()

    report = run_benchmark(samples_root=args.samples_root, include_builtin=not args.no_builtins)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print_table(report)
    print(f"Saved report to {args.output}")


if __name__ == "__main__":
    main()
