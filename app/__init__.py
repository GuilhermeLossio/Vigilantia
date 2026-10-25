"""Read-only Flask service that publishes recorded Vigilantia benchmark results."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, jsonify, render_template

from app import results as results_module

DEFAULT_RESULTS_PATH = Path(__file__).resolve().parents[1] / "reports" / "benchmark-results.json"
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
STATUS_CODES = {None: 200, "results_unavailable": 503, "results_invalid": 500}


def create_app(results_path: str | Path | None = None) -> Flask:
    app = Flask(__name__)
    path = Path(results_path or os.environ.get("VIGILANTIA_RESULTS_PATH") or DEFAULT_RESULTS_PATH)

    def read_results() -> tuple[dict | None, str | None, str | None]:
        """Return (results, error_code, detail). The small file is re-read per request."""
        try:
            return results_module.load(path), None, None
        except results_module.ResultsUnavailable as error:
            return None, "results_unavailable", str(error)
        except results_module.ResultsInvalid as error:
            app.logger.error("Invalid benchmark results: %s", error)
            return None, "results_invalid", str(error)

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/v1/benchmarks")
    def benchmarks():
        results, error, detail = read_results()
        if error:
            return jsonify(error=error, detail=detail), STATUS_CODES[error]
        return jsonify(results)

    @app.get("/")
    def index():
        results, error, detail = read_results()
        page = render_template(
            "index.html",
            results=results,
            error=error,
            detail=detail,
            metrics=results_module.METRICS,
            sources=results_module.KERNEL_SOURCES,
        )
        return page, STATUS_CODES[error]

    @app.after_request
    def add_security_headers(response):
        response.headers.update(SECURITY_HEADERS)
        return response

    return app
