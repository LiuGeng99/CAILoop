import json
from pathlib import Path

import numpy as np

from cailoop.aggregate import case_malignant_score, fine_grained_prediction, image_malignant_probability
from cailoop.confidence import display_confidence


ROOT = Path(__file__).resolve().parents[1]


def test_display_confidence_matches_the_paper_formula():
    probability = 0.91
    expected = 1.0 - (1.0 - probability) ** (1.0 / 3.0)
    assert display_confidence(probability) == expected


def test_internal_mean_and_prospective_top3_on_the_demo():
    payload = json.loads((ROOT / "demo" / "demo_cases.json").read_text())
    by_id = {case["case_id"]: case for case in payload["cases"]}

    internal = [image["probabilities"] for image in by_id["internal_demo"]["images"]]
    image_scores = image_malignant_probability(internal)
    assert np.allclose(image_scores, [0.10, 0.20, 0.80, 0.91])
    assert case_malignant_score(internal, "mean") == float(np.mean(image_scores))

    prospective = [image["probabilities"] for image in by_id["prospective_demo"]["images"]]
    prospective_scores = image_malignant_probability(prospective)
    top3 = np.sort(prospective_scores)[-3:].mean()
    assert case_malignant_score(prospective, "top3_mean") == float(top3)
    assert case_malignant_score(prospective, "mean") == float(prospective_scores.mean())


def test_fine_grained_uses_the_highest_confidence_image():
    payload = json.loads((ROOT / "demo" / "demo_cases.json").read_text())
    internal = next(case for case in payload["cases"] if case["case_id"] == "internal_demo")
    probabilities = [image["probabilities"] for image in internal["images"]]
    image_index, class_id, confidence = fine_grained_prediction(probabilities)
    assert (image_index, class_id, confidence) == (3, 0, 0.91)
