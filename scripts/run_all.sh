#!/usr/bin/env bash
# Build, test, compare parameters, and benchmark in one go. Results land in results/.
#   OPENCV_DIR=/path/to/opencv/lib/cmake/opencv4 ./scripts/run_all.sh   # non-system OpenCV
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="$ROOT/build"
RESULTS="$ROOT/results"

CMAKE_ARGS=(-DCMAKE_BUILD_TYPE=Release)
if [[ -n "${OPENCV_DIR:-}" ]]; then
    CMAKE_ARGS+=(-DOpenCV_DIR="$OPENCV_DIR")
fi

cmake -S "$ROOT" -B "$BUILD" "${CMAKE_ARGS[@]}"
cmake --build "$BUILD" -j"$(nproc)"

mkdir -p "$RESULTS"
APP="$BUILD/imgproc"

echo; echo "=== Upgrade 3: tests (GoogleTest + CLI checks) ==="
ctest --test-dir "$BUILD" --output-on-failure -j"$(nproc)" | tee "$RESULTS/tests.txt"
"$BUILD/imgproc_tests" > "$RESULTS/gtest.txt"

echo; echo "=== Upgrade 1: parameter comparison ==="
"$APP" compare -o "$RESULTS/compare" --csv "$RESULTS/compare_synthetic.csv" | tee "$RESULTS/compare_synthetic.txt"
if [[ -f "$ROOT/data/building.jpg" ]]; then
    "$APP" compare "$ROOT/data/building.jpg" -o "$RESULTS/compare" --csv "$RESULTS/compare_building.csv" \
        | tee "$RESULTS/compare_building.txt"
fi

echo; echo "=== Generating sample images ==="
"$APP" generate "$ROOT/samples" --count 12 --sizes 640x480,1280x720,1920x1080

echo; echo "=== Batch processing (sequential, then all cores) ==="
"$APP" batch "$ROOT/samples" -o "$ROOT/output" --csv "$RESULTS/batch.csv" | tee "$RESULTS/batch.txt"
"$APP" batch "$ROOT/samples" -o "$ROOT/output" --workers 0 | tee "$RESULTS/batch_parallel.txt"

echo; echo "=== Upgrade 2: size benchmark (multi-threaded OpenCV) ==="
"$APP" bench --iterations 50 --csv "$RESULTS/bench_mt.csv" | tee "$RESULTS/bench_mt.txt"

echo; echo "=== Size benchmark (single-threaded OpenCV) ==="
"$APP" bench --iterations 50 --threads 1 --csv "$RESULTS/bench_st.csv" | tee "$RESULTS/bench_st.txt"

echo; echo "=== Upgrade 2: sequential vs parallel images (small / medium / large) ==="
"$APP" scale --sizes small,medium,large --images 24 --repeats 3 --csv "$RESULTS/scale.csv" \
    | tee "$RESULTS/scale.txt"

echo; echo "Done. Results in $RESULTS, processed images in $ROOT/output"
