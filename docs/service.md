# Results and Inference Service

**MVP routes implemented.** The Flask service in `app/` reads the [published results, replay, external-validation, and hardware-subset files](../reports/README.md), validates them on every request, and reports them as pending until they exist. Live prediction loads the trained classical model on first use; without the model artifacts it answers `503 model_unavailable` while the rest of the service keeps working. Request bodies are limited to 2 KB, and the content security policy allows no scripts.

The page follows the visual style of the [author's portfolio](https://guilhermelossio.github.io/GuilhermeLossio/): a forest-green frame, a parchment page, brass rules, and Cinzel headings over Georgia text. Its header shows headline figures computed from the validated reports and links to each section. Cinzel is self-hosted from `app/static/fonts/` under the SIL Open Font License (`OFL.txt` alongside the font), so the page loads nothing from other origins: the policy adds only `font-src 'self'`. The page declares a light color scheme, so it and its charts keep the parchment theme when the operating system uses dark mode.

The service never trains models, fetches URLs, or submits quantum jobs. It runs locally; AWS deployment is a later phase described in the [roadmap](roadmap.md#deployment-strategy).

## API

| Method | Route | Behavior |
| --- | --- | --- |
| `GET` | `/health` | Application liveness |
| `POST` | `/v1/phishing/predict` | Implemented: `{"host": "..."}` returns the RBF SVM prediction, decision margin, features, out-of-range features, and model version; `400` for invalid input, `503` without model artifacts |
| `POST` | `/v1/exposure/analyze` | Later phase: consistency and reference-match checks on controlled samples |
| `GET` | `/v1/incidents` | Later phase: demonstration incidents and review states |
| `GET` | `/v1/benchmarks` | Implemented: return validated recorded results, `503` while pending, `500` for an invalid file |
| `GET` | `/v1/benchmarks/replay` | Implemented: recorded quantum-kernel SVM test predictions per source, labeled replay, with source runs |
| `GET` | `/v1/external-validation` | Implemented: the validated external-validation report, `503 results_unavailable` while it is missing, `500 results_invalid` for an invalid file |
| `GET` | `/v1/external-validation/qpu` | Implemented: the validated 40-row external subset report from `ibm_fez`, `503 results_unavailable` while it is missing, `500 results_invalid` for an invalid file |
| `GET` | `/benchmark.svg` | Implemented: the generated benchmark chart |
| `GET` | `/comparison.svg` | Implemented: the generated comparison chart (`docs/model-comparison.svg`), `404` when it is missing |
| `GET` | `/` | Implemented: host form, results page with chart and sources, the replay table, the external validation, and the hardware subset with the comparison chart |

The features are computed from the host alone; the page is never fetched. Invalid input and unavailable artifacts are distinguishable from a classification result. Saved quantum benchmark predictions belong to recorded benchmark results and must identify their sample and run as replay. The external-validation section labels the quantum-kernel SVM row as exact simulation, not hardware; the hardware-subset section labels each row by its test kernel source and shows the run, the sample size, the intervals, and a reading computed from the report, which says that differences of a row or two are not evidence that hardware improves the model. Training and hardware benchmark jobs do not run inside prediction requests. The later exposure module will return evidence observations, not authentication verdicts.

`app/external.py` and `app/external_qpu.py` validate the external report and the hardware-subset report with the standard library only, because the service image ships neither pandas nor the training code. Unit tests check that each accepts the published report and rejects the same broken variants as its counterpart in `src/evaluation/`.

## Run the Flask service

The Flask service needs the pinned dependencies in `requirements-app.txt` (the complete `requirements.txt` includes them). Until the results, replay, external-validation, and hardware-subset files exist, the page and the JSON routes report them as pending. Live prediction needs the trained models in `artifacts/benchmark/`, produced by `scripts/train_benchmark.py` or [notebook 03](../notebooks/03-training-and-simulated-evaluation.ipynb).

```powershell
python -m pip install -r requirements-app.txt
python -m flask --app "app:create_app()" run --port 8000
```

Open `http://127.0.0.1:8000/`. Without `--port`, Flask serves on its default `http://127.0.0.1:5000/`; the README and the [reproduction guide](reproducing.md#5-run-the-service) use the same command and port 8000.

### Run in Docker

Build and run the container, which serves on port 8000 with Gunicorn as a non-root user:

```powershell
docker build -t vigilantia .
docker run --rm -p 8000:8000 -v "${PWD}/artifacts/benchmark:/srv/vigilantia/artifacts/benchmark:ro" vigilantia
```

The image contains the code, the fitted scaler, the published results, the replay, the external validation and its hardware subset, and both charts, but not the trained models; the read-only mount provides them. The recorded QPU evidence, in both evidence directories, is excluded from the build context. Without the mount, prediction answers `503`. Open `http://127.0.0.1:8000/`. The image has a health check on `/health`.

### Configuration

`VIGILANTIA_RESULTS_PATH`, `VIGILANTIA_REPLAY_PATH`, `VIGILANTIA_EXTERNAL_PATH`, `VIGILANTIA_EXTERNAL_QPU_PATH`, `VIGILANTIA_CHART_PATH`, and `VIGILANTIA_COMPARISON_CHART_PATH` override the file locations. They are read from the process environment, not from `.env`. After a new evaluation, refresh the replay with `scripts/publish_replay.py` and the charts with `scripts/render_results_svg.py` and `scripts/render_comparison_svg.py`.

## Classify new hosts

Classify hosts or URLs with the served classical model from the command line. Only the host is used, and nothing is fetched or visited:

```powershell
python scripts/predict_host.py example.com https://login-check.web.app/
```

Each result includes the prediction, the SVM decision margin, the four features, any feature outside the training range, and a model version derived from the artifacts. The served model is the RBF SVM, named the primary nonlinear baseline before any test result existed; `--model logistic_regression` is also available. The margin is not a probability. The quantum-kernel SVM is not served live, because each prediction would need QPU kernel evaluations.

The page form and `POST /v1/phishing/predict` return the same result as JSON.

## Security and data handling

- Use appropriately licensed public data for phishing research and fictitious records for the exposure laboratory.
- Keep credentials, personal records, and real breach samples out of the repository, logs, and demonstration video.
- Validate input size and schema, protect analysis routes before public exposure, and define artifact retention.
- Do not fetch arbitrary user-provided URLs in the MVP.
- Publish no host or URL from either dataset; published files and replay identify samples by row index and hashed domain group.
- Keep operational secrets outside source code and use task roles for AWS access.
- Present correlations as investigation leads. Require human review before actions affecting customers or accounts.

Unit tests cover target mapping, domain isolation, training-only preprocessing, feature ordering, invalid input, and missing model artifacts; see [tests/README.md](../tests/README.md).

See the [architecture](../Architecture.md), the [results contracts](../reports/README.md), and the [documentation index](README.md).
