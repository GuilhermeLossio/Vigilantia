"""Flask service: recorded benchmark results, replayed quantum predictions, the external
validation report, and live classical inference for new hosts. It never fetches URLs, trains models, or submits jobs."""

from __future__ import annotations

import os
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_file

from app import external as external_module
from app import replay as replay_module
from app import results as results_module

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = ROOT / "reports" / "benchmark-results.json"
DEFAULT_REPLAY_PATH = ROOT / "reports" / "benchmark-replay.json"
DEFAULT_EXTERNAL_PATH = ROOT / "reports" / "external-validation.json"
DEFAULT_CHART_PATH = ROOT / "docs" / "benchmark-results.svg"
MAX_REQUEST_BYTES = 2048
SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; img-src 'self'; "
        "base-uri 'none'; frame-ancestors 'none'; form-action 'self'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
}
STATUS_CODES = {None: 200, "results_unavailable": 503, "results_invalid": 500}
REPLAY_LABELS = {
    "simulation": "Simulation",
    "qpu": "QPU, uncorrected",
    "qpu_calibrated": "QPU, calibrated",
}


def _default_predictor():
    """Load the served classical model lazily; heavy imports stay out of the results path."""
    from src.inference.predictor import Predictor

    return Predictor.from_artifacts()


def create_app(
    results_path: str | Path | None = None,
    replay_path: str | Path | None = None,
    chart_path: str | Path | None = None,
    predictor_factory=None,
    external_path: str | Path | None = None,
) -> Flask:
    app = Flask(__name__)

    @app.template_filter("duration")
    def duration(seconds: float | None) -> str:
        if seconds is None:
            return "n/a"
        return f"{seconds * 1000:.2f} ms" if seconds < 1 else f"{seconds:.1f} s"

    app.config["MAX_CONTENT_LENGTH"] = MAX_REQUEST_BYTES
    env = os.environ.get
    results_file = Path(results_path or env("VIGILANTIA_RESULTS_PATH") or DEFAULT_RESULTS_PATH)
    replay_file = Path(replay_path or env("VIGILANTIA_REPLAY_PATH") or DEFAULT_REPLAY_PATH)
    external_file = Path(external_path or env("VIGILANTIA_EXTERNAL_PATH") or DEFAULT_EXTERNAL_PATH)
    chart_file = Path(chart_path or env("VIGILANTIA_CHART_PATH") or DEFAULT_CHART_PATH)
    factory = predictor_factory or _default_predictor
    state: dict = {}

    def read(loader, path: Path) -> tuple[dict | None, str | None, str | None]:
        """Return (data, error_code, detail). Small files are re-read on each request."""
        try:
            return loader(path), None, None
        except results_module.ResultsUnavailable as error:
            return None, "results_unavailable", str(error)
        except results_module.ResultsInvalid as error:
            app.logger.error("Invalid published file %s: %s", path.name, error)
            return None, "results_invalid", str(error)

    def predictor():
        """Cache the predictor after the first successful load; report failures as None."""
        if "predictor" not in state:
            try:
                state["predictor"] = factory()
            except Exception as error:  # noqa: BLE001 - any artifact problem means unavailable
                app.logger.error("Model unavailable: %s", error)
                return None, str(error)
        return state["predictor"], None

    def classify(value: object) -> tuple[dict, int]:
        from src.inference.predictor import PredictionError

        model, problem = predictor()
        if model is None:
            return {"error": "model_unavailable", "detail": problem}, 503
        try:
            return model.predict(value), 200
        except PredictionError as error:
            return {"error": "invalid_input", "detail": str(error)}, 400

    @app.get("/health")
    def health():
        return jsonify(status="ok")

    @app.get("/v1/benchmarks")
    def benchmarks():
        data, error, detail = read(results_module.load, results_file)
        if error:
            return jsonify(error=error, detail=detail), STATUS_CODES[error]
        return jsonify(data)

    @app.get("/v1/benchmarks/replay")
    def replay():
        data, error, detail = read(replay_module.load, replay_file)
        if error:
            return jsonify(error=error, detail=detail), STATUS_CODES[error]
        return jsonify(data)

    @app.get("/v1/external-validation")
    def external_validation():
        data, error, detail = read(external_module.load, external_file)
        if error:
            return jsonify(error=error, detail=detail), STATUS_CODES[error]
        return jsonify(data)

    @app.get("/benchmark.svg")
    def chart():
        if not chart_file.is_file():
            abort(404)
        return send_file(chart_file, mimetype="image/svg+xml", max_age=300)

    @app.post("/v1/phishing/predict")
    def predict():
        body = request.get_json(silent=True)
        if not isinstance(body, dict) or not isinstance(body.get("host"), str):
            return jsonify(
                error="invalid_input", detail='Send JSON like {"host": "example.com"}.'
            ), 400
        payload, status = classify(body["host"])
        return jsonify(payload), status

    @app.get("/")
    def index():
        results, error, detail = read(results_module.load, results_file)
        replay_data, replay_error, _ = read(replay_module.load, replay_file)
        external, external_error, external_detail = read(external_module.load, external_file)
        query = request.args.get("host", "").strip()
        prediction, prediction_status = (None, 200)
        if query:
            prediction, prediction_status = classify(query[:MAX_REQUEST_BYTES])
        page = render_template(
            "index.html",
            results=results,
            error=error,
            detail=detail,
            metrics=results_module.METRICS,
            sources=results_module.KERNEL_SOURCES,
            replay=replay_data,
            replay_error=replay_error,
            replay_labels=REPLAY_LABELS,
            chart=chart_file.is_file(),
            external=external,
            external_error=external_error,
            external_detail=external_detail,
            external_steps=external_module.STEPS,
            external_sources=external_module.SOURCES,
            query=query,
            prediction=prediction,
            prediction_status=prediction_status,
        )
        return page, STATUS_CODES[error]

    @app.errorhandler(413)
    def too_large(_):
        return jsonify(error="invalid_input", detail="Request body is too large."), 413

    @app.after_request
    def add_security_headers(response):
        response.headers.update(SECURITY_HEADERS)
        return response

    return app
