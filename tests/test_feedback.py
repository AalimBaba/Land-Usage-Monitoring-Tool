from __future__ import annotations

import json

from PIL import Image

from land_usage.feedback import append_feedback, feedback_record, records_to_jsonl


def test_feedback_record_excludes_raw_image_and_can_be_written(tmp_path):
    image = Image.new("RGB", (32, 32), (80, 120, 90))
    record = feedback_record(
        uploaded_filename="sample.png",
        image=image,
        validation_status="Suitable",
        segmentation_status="Run",
        dominant_class="Vegetation",
        pixel_distribution={"Vegetation": 100.0},
        mean_softmax_confidence=72.5,
        feedback_category="Correct",
        user_note="Looks fine",
        expected_input_type="Satellite / aerial land image",
    )

    assert "image_hash" in record
    assert "raw_image" not in record

    path = tmp_path / "feedback_log.jsonl"
    assert append_feedback(record, path=path)

    rows = path.read_text(encoding="utf-8").splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["uploaded_filename"] == "sample.png"
    assert records_to_jsonl([record])
