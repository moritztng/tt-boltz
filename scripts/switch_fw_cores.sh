#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  ./switch_fw_cores.sh --disable-cores <true|false> [--yes]

Description:
  Uses a dedicated virtualenv to install/update firmware tools,
  downloads the latest firmware bundle, applies disable-count policy,
  and flashes the card.

Flags:
  --disable-cores true|false
                            true => disable 2 columns
                            false => disable 0 columns
  --yes                     Skip confirmation prompt
  -h, --help                Show this help

Examples:
  ./switch_fw_cores.sh --disable-cores true
  ./switch_fw_cores.sh --disable-cores false --yes
EOF
}

DISABLE_CORES_BOOL=""
OUTPUT_DIR="${HOME}/.cache/tt-fw-switch"
TOOLS_VENV="${OUTPUT_DIR}/tools-venv"
SKIP_CONFIRM=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --disable-cores)
            DISABLE_CORES_BOOL="${2:-}"
            shift 2
            ;;
        --yes)
            SKIP_CONFIRM=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown argument: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${DISABLE_CORES_BOOL}" ]]; then
    echo "Missing required flag: --disable-cores <true|false>" >&2
    usage
    exit 1
fi

if [[ "${DISABLE_CORES_BOOL}" != "true" && "${DISABLE_CORES_BOOL}" != "false" ]]; then
    echo "--disable-cores must be either 'true' or 'false'" >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "python3 is required but not found in PATH." >&2
    exit 1
fi

if ! command -v protoc >/dev/null 2>&1; then
    echo "protoc is required. Install it first (for example: sudo apt install protobuf-compiler)." >&2
    exit 1
fi

if [[ "${DISABLE_CORES_BOOL}" == "true" ]]; then
    DISABLE_COUNT=2
    DISABLE_MODE="on"
else
    DISABLE_COUNT=0
    DISABLE_MODE="off"
fi

mkdir -p "${OUTPUT_DIR}"

echo "==> Preparing firmware tools virtualenv..."
if [[ ! -x "${TOOLS_VENV}/bin/python" ]]; then
    python3 -m venv "${TOOLS_VENV}"
fi

TOOLS_PYTHON="${TOOLS_VENV}/bin/python"
TT_FLASH_BIN="${TOOLS_VENV}/bin/tt-flash"
TT_UPDATE_BIN="${TOOLS_VENV}/bin/tt-update-tensix-disable-count"

echo "==> Installing/updating firmware tools in ${TOOLS_VENV}..."
"${TOOLS_PYTHON}" -m pip install --upgrade pip
"${TOOLS_PYTHON}" -m pip install --upgrade tt-flash tt-update-tensix-disable-count

if [[ ! -x "${TT_FLASH_BIN}" || ! -x "${TT_UPDATE_BIN}" ]]; then
    echo "Firmware tools were not installed correctly in ${TOOLS_VENV}." >&2
    exit 1
fi

echo "==> Resolving latest firmware bundle..."
BUNDLE_OUTPUT="$("${TOOLS_PYTHON}" <<'PY'
from tt_flash.download import download_fwbundle
path = download_fwbundle("latest", no_tty=True)
print(f"__BUNDLE_PATH__={path}")
PY
)"

LATEST_BUNDLE=""
while IFS= read -r line; do
    echo "${line}"
    if [[ "${line}" == __BUNDLE_PATH__=* ]]; then
        LATEST_BUNDLE="${line#__BUNDLE_PATH__=}"
    fi
done <<< "${BUNDLE_OUTPUT}"

if [[ -z "${LATEST_BUNDLE}" ]]; then
    echo "Failed to resolve latest bundle path." >&2
    exit 1
fi

if [[ ! -f "${LATEST_BUNDLE}" ]]; then
    echo "Latest bundle file not found: ${LATEST_BUNDLE}" >&2
    exit 1
fi

LATEST_BASENAME="$(basename "${LATEST_BUNDLE}")"
LATEST_VERSION="${LATEST_BASENAME#fw_pack-}"
LATEST_VERSION="${LATEST_VERSION%.fwbundle}"
MODIFIED_BUNDLE="${OUTPUT_DIR}/fw_pack-${LATEST_VERSION}-disable-count-${DISABLE_COUNT}.fwbundle"

echo "==> Building modified bundle..."
"${TT_UPDATE_BIN}" \
    --input "${LATEST_BUNDLE}" \
    --output "${MODIFIED_BUNDLE}" \
    --disable-count "${DISABLE_COUNT}"

echo
echo "Planned action:"
echo "  Latest bundle:  ${LATEST_BUNDLE}"
echo "  Output bundle:  ${MODIFIED_BUNDLE}"
echo "  Disable mode:   ${DISABLE_MODE}"
echo "  Disable count:  ${DISABLE_COUNT}"
echo

if [[ "${SKIP_CONFIRM}" -ne 1 ]]; then
    read -r -p "Proceed with flash now? [y/N] " reply
    case "${reply}" in
        y|Y|yes|YES)
            ;;
        *)
            echo "Aborted before flashing."
            exit 0
            ;;
    esac
fi

echo "==> Flashing firmware..."
"${TT_FLASH_BIN}" flash "${MODIFIED_BUNDLE}" --force

echo "==> Post-flash check (tt-smi)..."
tt-smi -s --snapshot_no_tty

echo
echo "Flash flow completed."
echo "IMPORTANT: Reboot the host now to fully apply firmware changes."
