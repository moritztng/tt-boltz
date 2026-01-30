"""Flask application for TT-Boltz protein structure prediction demo."""

import json
import re
import threading
import time
from flask import Flask, request, Response, render_template, jsonify

from tt_boltz.predictor import predict_structure, validate_sequence

app = Flask(__name__)

# Rate limiting: track last prediction time per IP
_last_prediction = {}
_prediction_lock = threading.Lock()
_active_prediction = threading.Lock()  # Only one prediction at a time

# Constants
MAX_REQUEST_SIZE = 50 * 1024  # 50KB max request size
MIN_PREDICTION_INTERVAL = 5  # seconds between predictions per IP
MAX_SEQUENCE_LENGTH = 1024
MIN_SEQUENCE_LENGTH = 10


@app.route('/')
def index():
    """Serve the main page."""
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    """
    Stream protein structure prediction progress.
    
    Expects JSON body: {"sequence": "ACDEFG..."}
    Returns: NDJSON stream of progress events
    """
    # Check request size
    if request.content_length and request.content_length > MAX_REQUEST_SIZE:
        return jsonify({"type": "error", "message": "Request too large"}), 400
    
    # Rate limiting per IP
    client_ip = request.remote_addr or "unknown"
    current_time = time.time()
    
    with _prediction_lock:
        last_time = _last_prediction.get(client_ip, 0)
        if current_time - last_time < MIN_PREDICTION_INTERVAL:
            wait_time = int(MIN_PREDICTION_INTERVAL - (current_time - last_time)) + 1
            return jsonify({
                "type": "error", 
                "message": f"Please wait {wait_time} seconds before submitting again"
            }), 429
        _last_prediction[client_ip] = current_time
    
    # Parse JSON safely
    try:
        data = request.get_json(force=False, silent=False)
    except Exception:
        return jsonify({"type": "error", "message": "Invalid request format"}), 400
    
    if not data or not isinstance(data, dict):
        return jsonify({"type": "error", "message": "Invalid request format"}), 400
    
    if 'sequence' not in data:
        return jsonify({"type": "error", "message": "Missing sequence"}), 400
    
    # Validate sequence is a string
    raw_sequence = data.get('sequence')
    if not isinstance(raw_sequence, str):
        return jsonify({"type": "error", "message": "Sequence must be a string"}), 400
    
    # Clean and normalize sequence: remove all whitespace and non-alpha characters
    sequence = re.sub(r'[^A-Za-z]', '', raw_sequence).upper()
    
    # Length checks before full validation
    if len(sequence) == 0:
        return jsonify({"type": "error", "message": "Sequence is empty"}), 400
    
    if len(sequence) < MIN_SEQUENCE_LENGTH:
        return jsonify({"type": "error", "message": f"Sequence too short (minimum {MIN_SEQUENCE_LENGTH} residues)"}), 400
    
    if len(sequence) > MAX_SEQUENCE_LENGTH:
        return jsonify({"type": "error", "message": f"Sequence too long (maximum {MAX_SEQUENCE_LENGTH} residues)"}), 400
    
    # Full validation
    is_valid, error_msg = validate_sequence(sequence)
    if not is_valid:
        return jsonify({"type": "error", "message": error_msg}), 400
    
    # Get optional parameters with strict validation
    use_msa = bool(data.get('use_msa_server', True))
    accelerator = 'tenstorrent'  # Fixed for demo - don't allow user override
    
    # Try to acquire prediction lock (only one prediction at a time)
    if not _active_prediction.acquire(blocking=False):
        return jsonify({
            "type": "error", 
            "message": "Another prediction is in progress. Please wait."
        }), 503
    
    def generate():
        """Generator that yields NDJSON events."""
        try:
            for event in predict_structure(
                sequence=sequence,
                use_msa_server=use_msa,
                accelerator=accelerator,
            ):
                yield json.dumps(event) + '\n'
        except Exception:
            # Don't expose internal error details
            yield json.dumps({"type": "error", "message": "Prediction failed. Please try again."}) + '\n'
        finally:
            _active_prediction.release()
    
    return Response(
        generate(),
        mimetype='application/x-ndjson',
        headers={
            'Cache-Control': 'no-cache',
            'X-Accel-Buffering': 'no',  # Disable nginx buffering
        }
    )


@app.route('/validate', methods=['POST'])
def validate():
    """Validate a sequence without running prediction."""
    # Check request size
    if request.content_length and request.content_length > MAX_REQUEST_SIZE:
        return jsonify({"valid": False, "error": "Request too large"})
    
    try:
        data = request.get_json(force=False, silent=False)
    except Exception:
        return jsonify({"valid": False, "error": "Invalid request format"})
    
    if not data or not isinstance(data, dict) or 'sequence' not in data:
        return jsonify({"valid": False, "error": "Missing sequence"})
    
    raw_sequence = data.get('sequence')
    if not isinstance(raw_sequence, str):
        return jsonify({"valid": False, "error": "Sequence must be a string"})
    
    # Clean and normalize
    sequence = re.sub(r'[^A-Za-z]', '', raw_sequence).upper()
    
    is_valid, error_msg = validate_sequence(sequence)
    
    return jsonify({
        "valid": is_valid,
        "error": error_msg if not is_valid else None,
        "length": len(sequence) if is_valid else None
    })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)

