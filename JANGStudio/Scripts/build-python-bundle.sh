#!/bin/bash
# JANG Studio — assembles a hermetic Python 3.11 runtime for the .app bundle.
# Runs once per clean build; idempotent when build/python/bin/python3 already exists.
#
# What it does:
#   1. Downloads python-build-standalone (Astral) aarch64-apple-darwin tarball
#   2. Extracts to build/python/
#   3. Builds + pip-installs the local jang-tools wheel with [mlx] extras,
#      then mlx-vlm --no-deps (skips cv2/pyarrow/datasets), then transformers stack
#   4. Strips tests/docs/caches and unused stdlib (idlelib, tkinter, ensurepip)
#   5. Ad-hoc codesigns every .dylib / .so (real signing happens in codesign-runtime.sh)
#   6. Runs a smoke test: python3 -m jang_tools --version
#   7. Fails the build if total size exceeds 450 MB
#
# Output: build/python/ (relative to JANGStudio/) containing a self-contained
# interpreter + site-packages with jang, mlx, mlx-lm, mlx-vlm, numpy, safetensors,
# tqdm, transformers, tokenizers, sentencepiece, huggingface_hub.
#
# Usage: cd JANGStudio && Scripts/build-python-bundle.sh
# Created by Jinho Jang (eric@jangq.ai)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JANG_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_ROOT="$JANG_ROOT/JANGStudio/build/python"
STANDALONE_URL="https://github.com/astral-sh/python-build-standalone/releases/download/20241016/cpython-3.11.10+20241016-aarch64-apple-darwin-install_only.tar.gz"
WHEEL_DIR="$JANG_ROOT/jang-tools/dist"
MAX_MB=450

repair_stdlib_layout() {
    if [ -d "$BUILD_ROOT/lib/python3.11 2" ]; then
        echo "[bundle] repairing duplicate stdlib directory"
        rsync -a "$BUILD_ROOT/lib/python3.11 2/" "$BUILD_ROOT/lib/python3.11/"
        rm -rf "$BUILD_ROOT/lib/python3.11 2"
    fi
    if [ ! -f "$BUILD_ROOT/lib/python3.11/encodings/__init__.py" ]; then
        echo "[bundle] FAIL — Python stdlib is missing encodings package" >&2
        exit 1
    fi
}

refresh_local_jang_tools() {
    local src="$JANG_ROOT/jang-tools/jang_tools"
    local dst="$BUILD_ROOT/lib/python3.11/site-packages/jang_tools"
    if [ -d "$src" ] && [ -d "$dst" ]; then
        echo "[bundle] refreshing jang_tools package from local source"
        rsync -a --delete "$src/" "$dst/"
    fi
}

# Idempotency: if the bundle already exists and smoke-tests OK, skip.
if [ -x "$BUILD_ROOT/bin/python3" ]; then
    if "$BUILD_ROOT/bin/python3" -m jang_tools --version >/dev/null 2>&1; then
        refresh_local_jang_tools
        echo "[bundle] already assembled at $BUILD_ROOT (refreshed local jang_tools)"
        exit 0
    fi
    echo "[bundle] existing bundle is broken; rebuilding"
fi

echo "[bundle] cleaning $BUILD_ROOT"
rm -rf "$BUILD_ROOT"
mkdir -p "$BUILD_ROOT"

echo "[bundle] downloading python-build-standalone (~35 MB)"
curl -fsSL "$STANDALONE_URL" | tar xz -C "$BUILD_ROOT" --strip-components=1
repair_stdlib_layout

echo "[bundle] building jang-tools wheel"
(cd "$JANG_ROOT/jang-tools" && "$BUILD_ROOT/bin/python3" -m pip install --quiet build \
    && "$BUILD_ROOT/bin/python3" -m build --wheel 1>/dev/null)
WHEEL=$(ls -t "$WHEEL_DIR"/jang-*.whl | head -1)
if [ -z "$WHEEL" ]; then
    echo "[bundle] FAIL — no jang-*.whl found under $WHEEL_DIR" >&2
    exit 1
fi
echo "[bundle] wheel: $WHEEL"

echo "[bundle] pip installing jang[mlx] (primary stack)"
"$BUILD_ROOT/bin/python3" -m pip install --quiet --no-compile --disable-pip-version-check \
    --target "$BUILD_ROOT/lib/python3.11/site-packages" \
    "${WHEEL}[mlx]"
refresh_local_jang_tools

echo "[bundle] pip installing mlx-vlm --no-deps (skips cv2/pyarrow/datasets — inference-only deps)"
"$BUILD_ROOT/bin/python3" -m pip install --quiet --no-compile --disable-pip-version-check \
    --target "$BUILD_ROOT/lib/python3.11/site-packages" \
    --no-deps "mlx-vlm>=0.6.3"

echo "[bundle] pip installing transformers/tokenizers/sentencepiece (needed by jang_tools tokenizer detection)"
"$BUILD_ROOT/bin/python3" -m pip install --quiet --no-compile --disable-pip-version-check \
    --target "$BUILD_ROOT/lib/python3.11/site-packages" \
    "transformers>=4.40" "tokenizers>=0.19" "sentencepiece>=0.2"

echo "[bundle] stripping tests, docs, caches, unused stdlib"
find "$BUILD_ROOT" -type d \( -name "__pycache__" -o -name "tests" -o -name "test" \) -prune -exec rm -rf {} + 2>/dev/null || true
find "$BUILD_ROOT" -name "*.pyc" -delete
rm -rf \
    "$BUILD_ROOT/lib/python3.11/idlelib" \
    "$BUILD_ROOT/lib/python3.11/tkinter" \
    "$BUILD_ROOT/lib/python3.11/ensurepip" \
    "$BUILD_ROOT/share/man" \
    2>/dev/null || true
repair_stdlib_layout

echo "[bundle] ad-hoc codesigning dylibs (real signing in codesign-runtime.sh)"
find "$BUILD_ROOT" -type f \( -name "*.dylib" -o -name "*.so" \) -print0 \
    | xargs -0 -n1 codesign --force --sign - 2>/dev/null || true

BYTES=$(du -sk "$BUILD_ROOT" | awk '{print $1 * 1024}')
MB=$(( BYTES / 1024 / 1024 ))
echo "[bundle] total size: ${MB} MB (max ${MAX_MB})"
if [ "$MB" -gt "$MAX_MB" ]; then
    echo "[bundle] FAIL — bundle exceeds ${MAX_MB} MB cap" >&2
    exit 1
fi

echo "[bundle] smoke test: python3 -m jang_tools --version"
"$BUILD_ROOT/bin/python3" -m jang_tools --version

echo "[bundle] OK → $BUILD_ROOT (${MB} MB)"
