#!/bin/bash
# one-time setup: fetch the cutlass headers and the nvidia/dvlt checkpoint, then
# build the bf16 weight blob the binary loads. run from the repo root: ./setup.sh
# if the checkpoint repo is gated, the script points you at it to download by hand.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

CUTLASS_DIR="ext/cutlass"
MODEL_DIR="model"
SAFETENSORS="$MODEL_DIR/model.safetensors"
WEIGHTS="$MODEL_DIR/weights.dvlt"
HF_URL="https://huggingface.co/nvidia/dvlt/resolve/main/model.safetensors"

# green status lines; plain when stdout isn't a terminal.
if [ -t 1 ]; then G=$'\033[32m'; Z=$'\033[0m'; else G=""; Z=""; fi
say() { echo "${G}$*${Z}"; }

# is the repo on a rotational disk (hdd)? linux-only, fails quiet.
on_hdd() {
    command -v lsblk >/dev/null 2>&1 || return 1
    local src
    src=$(df --output=source "$ROOT" 2>/dev/null | tail -1) || return 1
    [ -b "$src" ] || return 1
    [ "$(lsblk -ndo ROTA "$src" 2>/dev/null | head -1)" = "1" ]
}

# the hand-made logo (assets/logo.txt, same file the cli reads), green on a terminal.
if [ -f "$ROOT/assets/logo.txt" ]; then
    printf '%s' "$G"; cat "$ROOT/assets/logo.txt"; printf 'github.com/yassa9%s\n' "$Z"
fi
echo

# PHASE 1: cutlass (header-only; needed to build ./build/dvlt). a sparse, blobless clone
#    pulls only include/ + the one fused-attention example (~25 MB), not the 165 MB tree.
if [ ! -d "$CUTLASS_DIR/include" ]; then
    say "==> fetching cutlass headers"
    command -v git >/dev/null || { echo "error: git not found" >&2; exit 1; }
    tmp=$(mktemp -d); trap 'rm -rf "$tmp"' EXIT
    git clone --depth 1 --filter=blob:none --no-checkout \
        https://github.com/NVIDIA/cutlass.git "$tmp/cutlass"
    git -C "$tmp/cutlass" sparse-checkout set include examples/41_fused_multi_head_attention
    git -C "$tmp/cutlass" checkout
    mkdir -p "$CUTLASS_DIR/examples"
    cp -r "$tmp/cutlass/include" "$CUTLASS_DIR/"
    cp -r "$tmp/cutlass/examples/41_fused_multi_head_attention" "$CUTLASS_DIR/examples/"
    rm -rf "$tmp"; trap - EXIT
else
    say "==> cutlass already present, skipping"
fi

# PHASE 2: checkpoint. the nvidia/dvlt weights are NVIDIA's, released under the NVIDIA
#    License (non-commercial, research/evaluation). we pull them straight from
#    nvidia's hugging face repo; if that repo is gated and the pull fails, we just
#    point you there to grab it by hand. nothing is accepted on your behalf.
if [ ! -f "$WEIGHTS" ]; then
    if [ ! -f "$SAFETENSORS" ]; then
        command -v wget >/dev/null || { echo "error: wget not found" >&2; exit 1; }
        mkdir -p "$MODEL_DIR"
        say "==> downloading model.safetensors (~468 MB) from $HF_URL"
        if wget -c -O "$SAFETENSORS.part" "$HF_URL"; then
            mv "$SAFETENSORS.part" "$SAFETENSORS"
        else
            rm -f "$SAFETENSORS.part"
            {
                echo
                echo "could not download the checkpoint automatically"
                echo "(the nvidia/dvlt repo may be gated or require a login)."
                echo "download it by hand from:"
                echo "    https://huggingface.co/nvidia/dvlt"
                echo "place model.safetensors at:"
                echo "    $SAFETENSORS"
                echo "then re-run ./setup.sh"
            } >&2
            exit 1
        fi
    fi
    say "==> converting safetensors -> $WEIGHTS (bf16) via build/convert"
    make convert >/dev/null
    ./build/convert
else
    say "==> $WEIGHTS already present, skipping"
fi

# PHASE 3: the binary
say "==> building ./build/dvlt"
make dvlt

echo
say "==> setup done. run:"
echo "./build/dvlt <image_dir>/"

# loading the 253 MB weight blob is i/o-bound; an ssd helps. only nag on hdd.
if on_hdd; then
    echo
    echo "note: this repo is on a rotational disk (hdd). for faster model loading,"
    echo "      copy the weights to an ssd and point -w at them:"
    echo "      ./build/dvlt -w /path/on/ssd/weights.dvlt <image_dir>/"
fi
