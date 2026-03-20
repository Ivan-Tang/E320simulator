#!/bin/bash
# Convenience wrapper for torchrun DDP training.
#
# Usage:
#   bash scripts/launch_ddp.sh <nproc_per_node> <module> [args...]
#
# Examples:
#   bash scripts/launch_ddp.sh 2 src.train --task edge \
#       --clusters /storage/.../sim_clusters_train.parquet \
#       --model interaction_net --epochs 100 \
#       --gradient-accumulation-steps 8 \
#       --checkpoint /storage/.../runs/inet_ddp
#
#   bash scripts/launch_ddp.sh 2 src.train --task embedder \
#       --clusters /storage/.../sim_clusters_train.parquet \
#       --epochs 30 --gradient-accumulation-steps 4 \
#       --checkpoint /storage/.../runs/embedder_ddp

NPROC=${1:?"Usage: launch_ddp.sh <nproc_per_node> <module> [args...]"}
shift
MODULE=${1:?"Usage: launch_ddp.sh <nproc_per_node> <module> [args...]"}
shift

exec torchrun --standalone --nproc_per_node="$NPROC" -m "$MODULE" "$@"
