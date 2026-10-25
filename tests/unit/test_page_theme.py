"""Portfolio theme of the results page: self-hosted font, strict CSP, and headline figures."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from app import create_app, headline_stats

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
FONTS = ROOT / "app" / "static" / "fonts"


def published(name: str) -> dict:
    return json.loads((ROOT / "reports" / name).read_text(encoding="utf-8"))


def test_font_is_self_hosted_with_its_license() -> None:
    client = create_app().test_client()
    response = client.get("/static/fonts/cinzel-latin.woff2")
    assert response.status_code == 200
    assert response.mimetype == "font/woff2"
    assert response.data[:4] == b"wOF2"
    assert "SIL Open Font License" in (FONTS / "OFL.txt").read_text(encoding="utf-8")


def test_page_loads_nothing_from_other_origins() -> None:
    response = create_app().test_client().get("/")
    policy = response.headers["Content-Security-Policy"]
    assert "font-src 'self'" in policy and "default-src 'none'" in policy
    assert "script-src" not in policy
    page = response.get_data(as_text=True)
    assert "fonts.googleapis.com" not in page and "<script" not in page
    assert "/static/fonts/cinzel-latin.woff2" in page


def test_navigation_targets_every_section() -> None:
    page = create_app().test_client().get("/").get_data(as_text=True)
    for section in ("classify", "benchmark", "external", "hardware", "replay"):
        assert f'href="#{section}"' in page and f'id="{section}"' in page


def test_headline_figures_come_from_the_reports() -> None:
    results = published("benchmark-results.json")
    external = published("external-validation.json")
    subset = published("external-qpu-subset.json")
    stats = headline_stats(results, external, subset)
    values = [stat["value"] for stat in stats]
    best = max(
        m["metrics"]["accuracy"] for m in results["models"] if m["kernel_source"] == "classical"
    )
    assert values[0] == f"{best:.3f}"
    billed = (
        results["runtime"]["qpu"]["charged_seconds"] + subset["runtime"]["qpu"]["charged_seconds"]
    )
    assert values[-1] == f"{billed:.0f} s"
    page = create_app().test_client().get("/").get_data(as_text=True)
    assert all(f"<b>{value}</b>" in page for value in values)


def test_headline_figures_follow_changed_reports() -> None:
    results = copy.deepcopy(published("benchmark-results.json"))
    for model in results["models"]:
        if model["name"] == "RBF SVM":
            model["metrics"]["accuracy"] = 0.999
    assert headline_stats(results, None, None)[0]["value"] == "0.999"


def test_headline_figures_skip_missing_reports() -> None:
    assert headline_stats(None, None, None) == []
    results = copy.deepcopy(published("benchmark-results.json"))
    results.pop("runtime")
    labels = [stat["label"] for stat in headline_stats(results, None, None)]
    assert not any("billed" in label for label in labels)
