"""Pins the source-derived target inventory used by every factorial arm."""

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TARGET_DIR = ROOT / "config" / "frozen_targets"

EXPECTED = {
    "ar-iclib": {
        "ACCEL_singleSample", "acquire_window", "transform", "featurize",
        "classify", "record_stats", "warmup_sensor", "train", "recognize",
        "sqrt16",
    },
    "sobel-iclib": {"sobel_filtering"},
    "stringsearch-iclib": {"init_search", "strsearch"},
    "fft-iclib": {"fft_radix2", "exp", "sin", "cos", "cexp"},
    "lqi-iclib": {"calculateDeliveryRatio"},
    "bc-iclib": {"btbl_bitcnt"},
}


def test_all_six_benchmarks_have_source_derived_targets():
    for app, expected in EXPECTED.items():
        data = json.loads((TARGET_DIR / f"{app}.json").read_text())
        selected = {
            name for name, choice in data["target_functions"].items()
            if choice.lower() == "approximate"
        }
        assert data["app"] == app
        assert selected == expected
        assert data["selection_policy"]
        assert data["code_summary"]


def test_stringsearch_no_longer_excludes_the_proven_reuse_target():
    data = json.loads((TARGET_DIR / "stringsearch-iclib.json").read_text())
    assert data["target_functions"]["init_search"] == "approximate"
    assert "T25" not in data["code_summary"]


def test_campaign_never_asks_an_llm_to_freeze_evaluation_targets():
    campaign = (ROOT / "scripts" / "run_ablation_campaign.sh").read_text()
    assert 'TARGETS_DIR="${TARGETS_DIR:-config/frozen_targets}"' in campaign
    assert "--write_target_functions" not in campaign
