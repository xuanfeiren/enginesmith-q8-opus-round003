#!/usr/bin/env bash
# Builds target/engine. Build time uses Python, NKI and neuronx-cc (to compile the kernels to NEFFs); the
# executable links only libnrt and the C/C++ runtime, with every NEFF embedded.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$(pwd)
NV=${NEURON_VENV:-/opt/aws_neuronx_venv_pytorch_inference_vllm_0_24_0_1_1_0}
NRT=/opt/aws/neuron
export PATH=$NV/bin:$NRT/bin:/usr/bin:/bin
export NEURON_PLATFORM_TARGET_OVERRIDE=trn2
export PYTHONPATH=$ROOT/kernels
# NKI's documented route compiles a kernel by calling it, which also runs it once: point the runtime at the run's
# experiment core (ENGINESMITH_NEURON_CORE + 2), never at the cores the engine and the reference use.
export NEURON_RT_VISIBLE_CORES=${NEURON_RT_VISIBLE_CORES:-$(( ${ENGINESMITH_NEURON_CORE:-0} + 2 ))}
mkdir -p target/neff target/neffcache target/gen
rm -f target/neff/*.neff   # only the current NEFF list gets embedded

# 1. NKI kernels -> NEFFs (cached in target/neffcache by source and shape)
"$NV/bin/python" kernels/build_neffs.py target/neff target/neffcache "${NEFF_JOBS:-32}"

# 2. embed the NEFFs
S=target/gen/neffs.S
INC=target/gen/neffs.inc
: > $S
: > $INC
echo '.section .rodata' >> $S
names=()
for f in target/neff/*.neff; do
  n=$(basename "$f" .neff)
  names+=("$n")
  printf '.balign 64\n.global neff_%s_start\nneff_%s_start:\n.incbin "%s"\n.global neff_%s_end\nneff_%s_end:\n' "$n" "$n" "$ROOT/$f" "$n" "$n" >> $S
  printf 'extern "C" const char neff_%s_start[], neff_%s_end[];\n' "$n" "$n" >> $INC
done
echo '.section .note.GNU-stack,"",@progbits' >> $S
echo 'static const NeffBlob NEFFS[] = {' >> $INC
for n in "${names[@]}"; do printf '  {"%s", neff_%s_start, neff_%s_end},\n' "$n" "$n" "$n" >> $INC; done
echo '};' >> $INC

# 3. the native executable
g++ -O2 -std=c++17 -I$NRT/include -Itarget/gen -Isrc src/engine.cpp $S -o target/engine.tmp \
    -L$NRT/lib -lnrt -lpthread -Wl,-rpath,$NRT/lib
mv target/engine.tmp target/engine
echo "built target/engine ($(du -h target/engine | cut -f1))"
