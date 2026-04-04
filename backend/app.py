from __future__ import annotations

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from backend.model import get_dataset, predict_text, train_model

app = Flask(__name__, static_folder="../frontend/dist", static_url_path="")
CORS(app)


@app.get("/api/health")
def health() -> tuple[dict[str, str], int]:
    return {"status": "ok"}, 200


@app.get("/api/metrics")
def metrics() -> tuple[dict[str, object], int]:
    _, model_metrics = train_model()
    return jsonify(model_metrics), 200


@app.get("/api/examples")
def examples() -> tuple[dict[str, object], int]:
    dataset = get_dataset().sample(min(8, len(get_dataset())), random_state=7)
    return jsonify({"examples": dataset.to_dict(orient="records")}), 200


@app.post("/api/predict")
def predict() -> tuple[dict[str, object], int]:
    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Text is required."}), 400

    result = predict_text(text)
    return jsonify(result), 200


@app.get("/")
def index() -> object:
    dist_dir = app.static_folder
    if dist_dir:
        return send_from_directory(dist_dir, "index.html")
    return {"message": "Frontend build not found. Run the React app build first."}, 200


@app.errorhandler(404)
def not_found(_: object) -> tuple[dict[str, str], int]:
    return {"error": "Not found"}, 404


if __name__ == "__main__":
    app.run(debug=True)
