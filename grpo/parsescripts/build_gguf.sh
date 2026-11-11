#!/bin/bash
# Build the two shipping GGUFs from the HF checkpoints.
#
# Two files are needed, not one, because GGUF/llama.cpp cannot toggle a LoRA adapter on and
# off at runtime the way the HF path does:
#   verifier  = runs/sft_v2_4b/merged          (frozen SFT, adapter OFF) -> abstain score
#   generator = SFT base + GRPO step80 merged  (adapter ON)              -> answer text
# The generator's merge is a separate CPU step; run parsescripts/merge_grpo.py first and pass
# its output dir as GEN_HF below.
#
# Quant: Q4_K_M for BOTH. Chosen over Q5_K_M and over an imatrix Q4: the ~0.04 generator f1
# gap that motivated escalating was a GATE SCORER BUG (strict scoring on the quantised side vs
# evaluate.py's rule-free raw_f1 on the fp32 side), not quantisation. Rescored symmetrically on
# full dev, plain Q4 drops -0.0036 strict (95% CI [-0.0176,+0.0106]) and -0.0002 rule-free, both
# PASS; see parsescripts/rescore_gguf_gate.py and runs/gguf/rescore_q4km_dev.txt. Q5_K_M would
# cost ~1.23x the file size (~+1.1GB of bundle for the pair) for no measured quality gain, so it
# and the imatrix build were retired. Verified in the shipped files: general.file_type = 15
# (Q4_K_M), general.quantization_version = 2, arch qwen3, 36 blocks, 398 tensors.
#
# VERSION MISMATCH, deliberate and must stay tested: the CONVERTER is llama.cpp (this script's
# $LLAMA, commit 3de52ac at time of build) while the RUNTIME is llama-cpp-python 0.3.36. They are
# different llama.cpp versions, so every GGUF produced here MUST be test-loaded through 0.3.36
# before being trusted (the final block does this).
#
# Env: ggufconv ONLY (python=3.11 + cmake). Never install the converter's requirements into
# localpaperreader -- doing so clobbered transformers/torch there once.
# Needs llama-quantize built: see the CPU-only cmake build in the project's quantise job script
# (GGML_CUDA=OFF, LLAMA_BUILD_SERVER/TESTS/EXAMPLES=OFF, LLAMA_OPENSSL=OFF, -j4).
#
# Runtime ~10-20 min total, CPU only, no GPU. Peak disk: the f16 intermediate is ~8GB per model,
# so write intermediates to node-local /tmp, NOT scratch (scratch's per-user NFS quota killed
# the 15GB generator merge once).
#
#   bash parsescripts/build_gguf.sh                       # both, default paths
#   GEN_HF=/tmp/grpo_v2_4b_step80_merged bash parsescripts/build_gguf.sh
set -euo pipefail

LLAMA=${LLAMA:-/scratch_hive/kw1525/llama.cpp}
ENVDIR=${ENVDIR:-/scratch_hive/kw1525/micromamba-root/envs/ggufconv}
OUT=${OUT:-runs/gguf}
TMP=${TMP:-/tmp/ggufbuild.$$}

VER_HF=${VER_HF:-runs/sft_v2_4b/merged}
GEN_HF=${GEN_HF:-/tmp/grpo_v2_4b_step80_merged}

CONVERT="$LLAMA/convert_hf_to_gguf.py"
QUANT="$LLAMA/build/bin/llama-quantize"
PY="$ENVDIR/bin/python"

for p in "$CONVERT" "$QUANT" "$PY"; do
  [ -e "$p" ] || { echo "MISSING: $p" >&2; exit 1; }
done
[ -d "$VER_HF" ] || { echo "MISSING verifier HF dir: $VER_HF" >&2; exit 1; }
[ -d "$GEN_HF" ] || { echo "MISSING generator HF dir: $GEN_HF -- run parsescripts/merge_grpo.py first" >&2; exit 1; }

mkdir -p "$OUT" "$TMP"
trap 'rm -rf "$TMP"' EXIT        # the f16 intermediates are ~8GB each; never leave them behind

build_one () {
  local name=$1 src=$2
  local f16="$TMP/$name-f16.gguf" final="$OUT/$name-q4km.gguf"
  echo "=== $name: HF -> f16 GGUF ==="
  "$PY" "$CONVERT" "$src" --outfile "$f16" --outtype f16
  echo "=== $name: f16 -> Q4_K_M ==="
  "$QUANT" "$f16" "$final" Q4_K_M
  rm -f "$f16"                   # free the intermediate before the next model
  ls -l "$final"
}

build_one verifier  "$VER_HF"
build_one generator "$GEN_HF"

# Test-load through the RUNTIME (not the converter) and assert the quant type that shipped.
# A GGUF that converts cleanly can still fail to load on 0.3.36 -- that is the whole point.
echo "=== test-load through llama-cpp-python 0.3.36 (the runtime) ==="
PYTHONPATH=. /scratch_hive/kw1525/micromamba-root/envs/localpaperreader/bin/python - "$OUT" <<'PYEOF'
import struct, sys, pathlib
out = pathlib.Path(sys.argv[1])
for f in sorted(out.glob("*-q4km.gguf")):
    with open(f, "rb") as fh:
        assert fh.read(4) == b"GGUF", f"{f.name}: not a GGUF"
    from llama_cpp import Llama
    llm = Llama(model_path=str(f), n_ctx=256, n_gpu_layers=0, verbose=False)
    ids = llm.tokenize(b"The capital of France is", add_bos=True, special=True)
    print(f"  {f.name}: loads OK, n_vocab {llm.n_vocab()}, tokenised {len(ids)} ids")
    del llm
print("all GGUFs load through the runtime")
PYEOF
echo
echo "built in $OUT -- now gate them before shipping:"
echo "  verifier : parsescripts/gguf_verifier_gate.py   (rho >= 0.95, AUC drop <= 0.02)"
echo "  generator: parsescripts/gguf_generator_gate.py  (mean drop <= 0.03, CI upper <= 0.05)"
echo "  tiers    : parsescripts/gguf_score_dev.py then recalibrate_tiers.py (Q4 shifts the scale)"
