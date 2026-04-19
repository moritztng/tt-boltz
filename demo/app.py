"""Flask front-end for the tt-boltz streaming predictor.

The prediction pipeline itself lives in :mod:`tt_boltz.predictor` — this
file is just an HTTP adapter around it.
"""

import json
import re
import threading
import time

from flask import Flask, Response, jsonify, render_template, request

from tt_boltz.predictor import MAX_LEN, MIN_LEN, predict_structure, validate_sequence

app = Flask(__name__)

MAX_REQUEST_SIZE = 50 * 1024                      # 50 KiB — only a sequence
MIN_PREDICTION_INTERVAL = 5                       # seconds per client IP

_last_prediction: dict[str, float] = {}
_rate_limit_lock = threading.Lock()
_active_prediction = threading.Lock()             # one inference at a time


def _clean(raw: object) -> str:
    """Strip anything that isn't a letter and upper-case the rest."""
    if not isinstance(raw, str):
        return ""
    return re.sub(r"[^A-Za-z]", "", raw).upper()


def _parse_request(require_length: bool = True):
    """Parse and clean a sequence from a JSON POST body.

    Returns ``(sequence, body, None)`` on success or
    ``(None, None, (payload, status))`` on failure so the caller can forward
    the error response unchanged.
    """
    if request.content_length and request.content_length > MAX_REQUEST_SIZE:
        return None, None, ({"type": "error", "message": "Request too large"}, 400)

    try:
        data = request.get_json(force=False, silent=False)
    except Exception:
        return None, None, ({"type": "error", "message": "Invalid request format"}, 400)

    if not isinstance(data, dict) or "sequence" not in data:
        return None, None, ({"type": "error", "message": "Missing sequence"}, 400)

    seq = _clean(data.get("sequence"))
    if require_length:
        if not seq:
            return None, None, ({"type": "error", "message": "Sequence is empty"}, 400)
        if len(seq) < MIN_LEN:
            return None, None, ({"type": "error", "message": f"Sequence too short (minimum {MIN_LEN} residues)"}, 400)
        if len(seq) > MAX_LEN:
            return None, None, ({"type": "error", "message": f"Sequence too long (maximum {MAX_LEN} residues)"}, 400)

    ok, err = validate_sequence(seq)
    if not ok:
        return None, None, ({"type": "error", "message": err}, 400)

    return seq, data, None


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Stream an NDJSON sequence of prediction events for one sequence."""
    sequence, body, err = _parse_request()
    if err is not None:
        payload, status = err
        return jsonify(payload), status

    client_ip = request.remote_addr or "unknown"
    now = time.time()
    with _rate_limit_lock:
        last = _last_prediction.get(client_ip, 0.0)
        if now - last < MIN_PREDICTION_INTERVAL:
            wait = int(MIN_PREDICTION_INTERVAL - (now - last)) + 1
            return jsonify({
                "type": "error",
                "message": f"Please wait {wait} seconds before submitting again",
            }), 429
        _last_prediction[client_ip] = now

    if not _active_prediction.acquire(blocking=False):
        return jsonify({
            "type": "error",
            "message": "Another prediction is in progress. Please wait.",
        }), 503

    use_msa = bool(body.get("use_msa_server", True))

    def stream():
        try:
            for event in predict_structure(
                sequence=sequence,
                use_msa_server=use_msa,
                accelerator="tenstorrent",
            ):
                yield json.dumps(event) + "\n"
        except Exception:
            # Never leak internal details to the browser.
            yield json.dumps({"type": "error", "message": "Prediction failed. Please try again."}) + "\n"
        finally:
            _active_prediction.release()

    return Response(
        stream(),
        mimetype="application/x-ndjson",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.route("/validate", methods=["POST"])
def validate():
    """Validate a sequence without running prediction."""
    sequence, _body, err = _parse_request(require_length=False)
    if err is not None:
        payload, _ = err
        return jsonify({"valid": False, "error": payload.get("message", "Invalid sequence")})
    return jsonify({"valid": True, "error": None, "length": len(sequence)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
