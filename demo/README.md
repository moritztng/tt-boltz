# TT-Boltz Web Demo

A web interface for protein structure prediction using TT-Boltz.

## Running the Demo

1. Make sure TT-Boltz is installed:
   ```bash
   cd /path/to/tt-boltz
   pip install -e .
   ```

2. Start the Flask server:
   ```bash
   cd demo
   python app.py
   ```

3. Open your browser at: http://localhost:5000

## Features

- Enter amino acid sequences (10-2000 residues)
- Real-time progress updates during prediction
- Interactive 3D structure visualization with Mol*
- Confidence score display (pLDDT, pTM, confidence score)
- Download predicted structure as CIF file

## Example Sequences

The demo includes example sequences:
- **Lysozyme** (129 residues)
- **Ubiquitin** (76 residues)
- **Insulin A chain** (21 residues)

## API Endpoints

### POST /predict
Submit a sequence for structure prediction. Returns a streaming NDJSON response with progress updates and final results.

**Request:**
```json
{"sequence": "ACDEFGHIK..."}
```

**Response (NDJSON stream):**
```json
{"type": "progress", "stage": "msa", "message": "Generating MSA...", "step": 0, "total": 1}
{"type": "progress", "stage": "diffusion", "message": "Diffusion step 45/200", "step": 45, "total": 200}
{"type": "complete", "cif": "...", "confidence": {"confidence_score": 0.85, ...}}
```

### POST /validate
Validate a sequence without running prediction.

**Request:**
```json
{"sequence": "ACDEFGHIK..."}
```

**Response:**
```json
{"valid": true, "length": 100}
```

