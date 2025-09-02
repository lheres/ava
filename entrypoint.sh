#!/bin/sh
# This script sets performance-critical environment variables based on the CPU
# and then executes the main application command.

# 1. Get the number of physical CPU cores
CORES=$(lscpu -p=Core,Socket | grep -v '^#' | sort -u | wc -l)
echo "INFO: Found ${CORES} physical cores. Setting OMP_NUM_THREADS."

# 2. Set environment variables for optimal performance on Intel CPUs
export OMP_NUM_THREADS=${CORES}
export KMP_AFFINITY=granularity=fine,compact,1,0
export KMP_BLOCKTIME=1

# 3. Execute the command passed to this script (the Dockerfile's CMD)
exec "$@"
