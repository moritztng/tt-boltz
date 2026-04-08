#!/bin/bash
set -e

# Robust paths relative to this script's location
TESTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TT_BOLTZ_DIR="$(dirname "$TESTS_DIR")"
WORKSPACE="$(dirname "$TT_BOLTZ_DIR")"
TT_METAL_DIR="$WORKSPACE/tt-metal"
LOGFILE="$TT_BOLTZ_DIR/test_run.log"

log() {
    echo -e "$1" | tee -a "$LOGFILE"
}

setup_env() {
    cd "$TT_BOLTZ_DIR"
    source env/bin/activate
    export PYTHONPATH="$TT_METAL_DIR"
    
    # Ensure sfpi runtime is linked for TT hardware
    mkdir -p env/lib/python3.10/site-packages/ttnn/runtime
    ln -sf "$TT_METAL_DIR/runtime/sfpi" env/lib/python3.10/site-packages/ttnn/runtime/sfpi
}

build_stack() {
    log "=== Build Started: $(date) ==="
    
    log "--> Updating tt-metal"
    cd "$TT_METAL_DIR"
    git pull origin main >> "$LOGFILE" 2>&1 || true
    log "--> Building tt-metal"
    ./build_metal.sh >> "$LOGFILE" 2>&1

    log "--> Updating tt-boltz"
    cd "$TT_BOLTZ_DIR"
    git pull origin main >> "$LOGFILE" 2>&1 || true
}

test_correctness() {
    log "=== Correctness Test Started: $(date) ==="
    setup_env
    cd "$TT_BOLTZ_DIR"
    local SEED=48
    
    log "--> Predict (Normal): hemoglobin"
    tt-boltz predict examples/hemoglobin.yaml --use_msa_server --override --seed $SEED >> "$LOGFILE" 2>&1
    log "--> Eval (Normal):"
    python tests/test_structure.py hemoglobin | tee -a "$LOGFILE"

    log "--> Predict (Fast): hemoglobin"
    tt-boltz predict examples/hemoglobin.yaml --use_msa_server --override --fast --seed $SEED >> "$LOGFILE" 2>&1
    log "--> Eval (Fast):"
    python tests/test_structure.py hemoglobin | tee -a "$LOGFILE"
}

test_memory() {
    local MAX_LEN=${1:-1536}
    log "=== Memory Test Started: $(date) ==="
    setup_env
    cd "$TT_BOLTZ_DIR"
    local INPUT_DIR="$TT_BOLTZ_DIR/memory_inputs"
    local SEED=1337

    log "--> Generating random inputs up to seq len $MAX_LEN (step 32)"
    rm -rf "$INPUT_DIR"
    mkdir -p "$INPUT_DIR"
    python tests/generate_random_protein_sweep.py --out-dir "$INPUT_DIR" --max-len $MAX_LEN --step 32 >> "$LOGFILE" 2>&1

    log "--> Running Memory Test (Normal mode)"
    tt-boltz predict "$INPUT_DIR/inputs" --override --recycling_steps 0 --sampling_steps 10 --diffusion_samples 5 --seed $SEED --debug --log >> "$LOGFILE" 2>&1 || log "Normal mode encountered an error/OOM!"

    log "--> Running Memory Test (Fast mode)"
    tt-boltz predict "$INPUT_DIR/inputs" --override --recycling_steps 0 --sampling_steps 10 --diffusion_samples 5 --seed $SEED --fast --debug --log >> "$LOGFILE" 2>&1 || log "Fast mode encountered an error/OOM!"
}

print_usage() {
    echo "Usage: $0 {all|build|correctness|memory} [max_len]"
    echo "  all         : Run build, correctness, and memory tests"
    echo "  build       : Pull latest and build tt-metal"
    echo "  correctness : Run hemoglobin accuracy tests"
    echo "  memory      : Run memory limit sweep tests (default max_len: 1536)"
}

COMMAND=${1:-all}

case "$COMMAND" in
    all)
        echo "Starting test suite: ALL" > "$LOGFILE"
        build_stack
        test_correctness
        test_memory "$2"
        ;;
    build)
        echo "Starting test suite: BUILD" > "$LOGFILE"
        build_stack
        ;;
    correctness)
        echo "Starting test suite: CORRECTNESS" > "$LOGFILE"
        test_correctness
        ;;
    memory)
        echo "Starting test suite: MEMORY" > "$LOGFILE"
        test_memory "$2"
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

log "=== Done: $(date) ==="