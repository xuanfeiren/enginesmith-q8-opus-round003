#!/bin/bash
# dev only: profile one NEFF on the experiment core: cap.sh X.neff OUTDIR  -> OUTDIR/sum.txt, OUTDIR/prof.json
set -e
N=$1; O=$2; mkdir -p $O
export NEURON_RT_VISIBLE_CORES=${XCORE:-34}
cd $O
neuron-explorer capture -n $N -s prof.ntff --num-exec 3 > cap.log 2>&1
neuron-explorer view -n $N -s prof_exec_3.ntff --output-format summary-text > sum.txt 2>&1
[ -n "$JSON" ] && neuron-explorer view -n $N -s prof_exec_3.ntff --output-format json > json.log 2>&1 || true
ls $O
