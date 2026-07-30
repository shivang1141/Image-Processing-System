# Automated validation (GoogleTest)

The project is verified by **104 GoogleTest tests** plus **8 end-to-end CLI checks**, all run through CTest.

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
ctest --test-dir build --output-on-failure -j$(nproc)   # everything
./build/imgproc_tests --gtest_filter='InvalidImages*'    # one group
```

GoogleTest comes from the system package (`sudo apt install libgtest-dev`) when it's installed. Otherwise CMake downloads v1.14.0 automatically.

## Results

| | |
|---|---|
| GoogleTest | **104 / 104 passed** in 28 suites (3.1 s) |
| CLI checks | **8 / 8 passed** |
| CTest total | **112 / 112 passed** (4.1 s with `-j12`) |
| Environment | WSL2 Ubuntu 24.04, g++ 13.3, OpenCV 4.6.0, GoogleTest 1.14.0 |

`./scripts/run_all.sh` writes the full logs to `results/tests.txt` (CTest) and `results/gtest.txt` (every GoogleTest case).

## What is tested

| File | Area |
|------|------|
| [`tests/test_validation.cpp`](../tests/test_validation.cpp) | Missing files, invalid images, invalid parameters, unwritable output, image sizes, output properties |
| [`tests/test_processor.cpp`](../tests/test_processor.cpp) | Grayscale, blur, Canny, load/save, single-file pipeline |
| [`tests/test_batch.cpp`](../tests/test_batch.cpp) | Folder processing, parallel vs sequential, reports, CSV, benchmarks |
| [`tests/test_compare.cpp`](../tests/test_compare.cpp) | Edge metrics, parameter grid, how parameters change the edges |
| [`tests/test_parse.cpp`](../tests/test_parse.cpp) | Command-line sizes, numbers and lists |
| [`CMakeLists.txt`](../CMakeLists.txt) (`cli.*`) | The real `imgproc` binary exits non-zero with a readable message |

### Missing files

| Case | Expected | Result |
|------|----------|--------|
| `loadImage` on a path that doesn't exist | `runtime_error` "File not found" | ✅ |
| `loadImage` on a directory named `x.png` | `runtime_error` "Not a regular file" | ✅ |
| `processFile` on a missing file | throws, **writes no output files** | ✅ |
| `processFolder` on a missing folder, or on a file | `runtime_error` "Not a directory" | ✅ |
| CLI `process definitely_missing.png` | exit 1, "File not found" | ✅ |
| CLI `batch definitely_missing_dir` | exit 1, "Not a directory" | ✅ |

### Invalid images

Each case is a file with an image extension that can't be decoded. Every case is tested two ways: `loadImage` must throw "Failed to decode", and `batch` must record the file as a failure and still process the good image next to it.

| File | Contents | Result |
|------|----------|--------|
| `empty.png` | 0 bytes | ✅ |
| `garbage.jpg` | Plain text | ✅ |
| `notes.bmp` | Text file renamed | ✅ |
| `header.png` | Only the 8-byte PNG signature | ✅ |
| `truncated.png` | First half of a valid PNG | ✅ |
| `.txt`, `.csv`, no extension | Not images | ✅ ignored by `batch` (neither processed nor reported as failed) |

Other invalid input: 16-bit and float images passed to Canny are rejected with `invalid_argument` instead of crashing inside OpenCV ✅.

### Invalid parameters and unwritable output

| Case | Result |
|------|--------|
| Negative thresholds, negative blur, aperture not 3/5/7 → `invalid_argument` | ✅ |
| Aperture 4 is rejected *before* it reaches OpenCV (which would throw `cv::Exception`) | ✅ |
| `processFolder` with bad parameters fails once, up front, instead of once per file | ✅ |
| Saving into a missing folder, or with an unknown extension → `runtime_error` | ✅ |
| Output folder path that is actually a file → throws | ✅ |
| CLI: `--iterations ten`, `--sizes 640x`, `--aperture 4`, `--low -5`, `--bogus`, unknown command → exit 1 with a message naming the problem | ✅ |

### Different image sizes and expected output properties

Two parameterized tests run at **9 sizes**: 1×1, 2×2, 3×7, 17×13, 1×500, 500×1, 640×480, 1920×1080 and 3840×2160. That's 18 cases, including tiny and odd sizes and single-row and single-column images.

For each size the tests check:

- Grayscale output is `CV_8UC1` and the same size as the input.
- Every gray pixel lies between that pixel's smallest and largest BGR channel (±1 for rounding). This holds because grayscale is a weighted average of the channels.
- The edge map is `CV_8UC1`, the same size as the input, **strictly binary** (0 or 255 only), and never "all edges".
- The `_gray.png` and `_edges.png` files written to disk decode back to the same size, with one channel, and the edges are still binary. ✅ all 18

Other output checks:

- **Edges sit on real intensity changes.** Every edge pixel found on a white square lies within 2 px of the square's true border.
- **Deterministic.** The same input always gives the same edges.
- **Colour or gray input gives the same edges.**
- **Grayscale weights.** Pure blue → 29, green → 150, red → 76 (OpenCV's 0.114 / 0.587 / 0.299).

### Parameter behaviour (Upgrade 1)

These check properties of Canny that must always hold, not just example outputs:

- Raising the **high** threshold never adds edges, and neither does raising the **low** threshold. This follows from Canny's hysteresis step.
- On a noisy image, no blur gives more than twice the edge pixels of a 5×5 blur, and a 9×9 blur gives fewer still.
- Blur reduces fragmentation: there are fewer separate edge segments.

### Concurrency (Upgrade 2)

- **Parallel output is byte-for-byte identical to sequential.** 12 images processed with 1 worker and with 4 workers give identical `_gray.png` and `_edges.png` files, the same result order and the same failure list.
- More workers than images works. All 8 failures are collected when 4 workers process 8 broken files.
- Results come back sorted by file name whatever the worker count.
- `photo.png` and `photo.bmp` in the same folder produce `photo_png_*` and `photo_bmp_*` and no longer overwrite each other. Without this, two threads could also write the same output file at the same time.
- The scaling benchmark runs every mode and restores OpenCV's global thread setting afterwards.

## Problems found and fixed while adding validation

| Problem | Fix |
|---------|-----|
| `--iterations ten` printed just `Error: stoi`; `--iterations 5x` was silently accepted as 5 | Strict parsing with messages such as `Invalid --iterations: 'ten' (expected an integer)` |
| Bad Canny settings (e.g. aperture 4) only failed deep inside OpenCV with a `cv::Exception` | `validateParams` rejects them up front with a clear message; `--aperture` is now a CLI option |
| A missing file and a corrupt file gave the same "Failed to load" message | Separate "File not found", "Not a regular file" and "Failed to decode" messages |
| `a.png` and `a.jpg` in one folder overwrote each other's outputs | Output names include the extension when stems collide |
| The first image in a batch paid OpenCV's one-time start-up cost | Untimed warm-up before the batch timer starts |
| `--sizes 640x480,` (trailing comma) was accepted | Rejected as an empty list entry |

## Not covered

- **Truncated JPEGs.** libjpeg can return a partly decoded image with only a warning, so a cut-off `.jpg` may "succeed" with a grey bottom area. Only truncated PNGs are tested.
- **Permission errors** (read-only folders). These behave differently under WSL's `/mnt/c` and on native Linux, so they aren't tested.
- **Timing values.** The tests check that timings are present and ordered (min ≤ mean ≤ max), not how fast anything runs. Speed is measured by the benchmarks in [EXPERIMENTS.md](EXPERIMENTS.md).
