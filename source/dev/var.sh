#!/bin/bash
# dev: compile + profile one kc.py spec with env overrides: var.sh TAG MODULE KERNEL SPEC [VAR=val ...]
TAG=$1; MOD=$2; K=$3; SP=$4; shift 4
E=/home/ubuntu/es-dev2/tasks/q8-opus/runs/q8-opus-20261103-223918/work/003/engine
source $E/dev/env.sh
L=$E/../brief/skill/scripts/gpu_lock.py
env NEURON_RT_VISIBLE_CORES=35 "$@" timeout 1800 python $E/dev/kc.py $MOD $K $SP /tmp/q8x/v_$TAG.neff > /tmp/q8x/v_$TAG.log 2>&1 || { tail -5 /tmp/q8x/v_$TAG.log; exit 1; }
JSON=${JSON:-} timeout 900 python $L -- bash $E/dev/cap.sh /tmp/q8x/v_$TAG.neff /tmp/q8x/pv_$TAG > /dev/null 2>&1
echo "$TAG $(grep -E '^ *total_time' /tmp/q8x/pv_$TAG/sum.txt) $(grep -E '^ *hbm_read_bytes' /tmp/q8x/pv_$TAG/sum.txt)"
