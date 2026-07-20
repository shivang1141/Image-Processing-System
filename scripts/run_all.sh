#!/usr/bin/env bash
# Build, test, and benchmark in one go. Results land in results/.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
RESULTS="$ROOT/results"

cmake -S "$ROOT" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release
cmake --build "$BUILD" -j"$(nproc)"

echo; echo "=== Tests ==="
ctest --test-dir "$BUILD" --output-on-failure

mkdir -p "$RESULTS"
APP="$BUILD/imgproc"

echo; echo "=== Parameter comparison ==="
"$APP" compare -o "$RESULTS/compare" --csv "$RESULTS/compare_synthetic.csv" | tee "$RESULTS/compare_synthetic.txt"
if [[ -f "$ROOT/data/building.jpg" ]]; then
    "$APP" compare "$ROOT/data/building.jpg" -o "$RESULTS/compare" --csv "$RESULTS/compare_building.csv" \
        | tee "$RESULTS/compare_building.txt"
fi

echo; echo "=== Generating sample images ==="
"$APP" generate "$ROOT/samples" --count 12 --sizes 640x480,1280x720,1920x1080

echo; echo "=== Batch processing ==="
"$APP" batch "$ROOT/samples" -o "$ROOT/output" --csv "$RESULTS/batch.csv" | tee "$RESULTS/batch.txt"

echo; echo "=== Size benchmark (multi-threaded) ==="
"$APP" bench --iterations 50 --csv "$RESULTS/bench_mt.csv" | tee "$RESULTS/bench_mt.txt"

echo; echo "=== Size benchmark (single-threaded) ==="
"$APP" bench --iterations 50 --threads 1 --csv "$RESULTS/bench_st.csv" | tee "$RESULTS/bench_st.txt"

echo; echo "Done. Results in $RESULTS, processed images in $ROOT/output"
