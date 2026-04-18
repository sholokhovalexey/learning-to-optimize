#!/bin/bash
set -euo pipefail

# Adopted from here: https://github.com/VLSIDA/lithosim/blob/main/get_benchmarks.sh

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$SCRIPT_DIR/benchmarks"

mkdir -p  "$BENCH_DIR/iccad2013"

# --- ICCAD 2013: Mask optimization .glp layouts (32nm M1) ---
echo ""
echo "=== ICCAD 2013 Mask Optimization Benchmarks ==="
BASE_URL="https://raw.githubusercontent.com/OpenOPC/OpenILT/main/benchmark"
for i in $(seq 1 10); do
    F="$BENCH_DIR/iccad2013/M1_test${i}.glp"
    if [ ! -f "$F" ]; then
        echo "Downloading M1_test${i}.glp..."
        curl -sL -o "$F" "$BASE_URL/ICCAD2013/M1_test${i}.glp"
    fi
done

GCD="$BENCH_DIR/iccad2013/gcd_45nm.gds"
if [ ! -f "$GCD" ]; then
    echo "Downloading gcd_45nm.gds..."
    curl -sL -o "$GCD" "$BASE_URL/gcd_45nm.gds"
fi
echo "Done."

echo ""
echo "Benchmarks ready in $BENCH_DIR/"
echo ""
echo "ICCAD 2013 (GLP + GDS):"
ls "$BENCH_DIR/iccad2013/"*.glp "$BENCH_DIR/iccad2013/"*.gds 2>/dev/null | while read f; do echo "  $f"; done