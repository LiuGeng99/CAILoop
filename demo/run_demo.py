"""Run CAILoop on the bundled synthetic cases."""

from __future__ import annotations

import json
from pathlib import Path

from cailoop.aggregate import call_malignant, case_malignant_score, fine_grained_prediction
from cailoop.confidence import display_confidence
from cailoop.taxonomy import CLASS_NAMES


def main() -> None:
    path = Path(__file__).with_name("demo_cases.json")
    payload = json.loads(path.read_text())
    print("case_id            cohort        score   call        fine_grained          display_conf")
    for case in payload["cases"]:
        probabilities = [image["probabilities"] for image in case["images"]]
        score = case_malignant_score(probabilities, case["aggregation"])
        called = "malignant" if call_malignant(score) else "non-malignant"
        image_index, class_id, confidence = fine_grained_prediction(probabilities)
        shown = display_confidence(confidence)
        print(
            f"{case['case_id']:<18} {case['cohort']:<13} {score:7.4f}  {called:<13} "
            f"{CLASS_NAMES[class_id]:<12} img {image_index}   {shown:.4f}"
        )


if __name__ == "__main__":
    main()
