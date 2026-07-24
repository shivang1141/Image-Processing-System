#pragma once

#include "processor.hpp"

#include <opencv2/core.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace imgproc {

// ---------- Folder (batch) processing ----------

struct BatchReport {
    std::vector<ProcessResult> results;  // in sorted file order, whatever the worker count
    std::vector<std::string> failures;   // "path: reason"
    int workers = 1;
    double wallMs = 0.0;                 // elapsed time for the whole folder

    StageTimings average() const;
    double imagesPerSecond() const;
};

// True for file extensions OpenCV can typically decode (.png, .jpg, .bmp, ...).
bool isSupportedImage(const std::string& path);

// Number of worker threads to use when the caller asks for `requested` (0 = one per CPU core).
int resolveWorkers(int requested);

// Processes every supported image in inputDir (non-recursive, sorted by name).
// With workers > 1 independent images are processed concurrently on a pool of
// std::threads (0 = one per CPU core). Files that fail to load are recorded in
// BatchReport::failures and skipped. Files sharing a stem (a.png, a.jpg) get
// distinct output names (a_png_*, a_jpg_*) so no output is overwritten.
BatchReport processFolder(const std::string& inputDir,
                          const std::string& outputDir,
                          const EdgeParams& params = {},
                          int workers = 1);

void printBatchReport(std::ostream& os, const BatchReport& report);
void writeBatchCsv(const std::string& path, const BatchReport& report);

// ---------- Image-size benchmark ----------

struct Stats {
    double meanMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
};

Stats computeStats(const std::vector<double>& samples);

struct SizeBenchmarkResult {
    int width = 0;
    int height = 0;
    int iterations = 0;
    Stats grayscale;
    Stats edges;
};

// Builds a deterministic synthetic BGR test image (gradient + shapes + noise).
cv::Mat makeSyntheticImage(int width, int height, unsigned seed = 42);

// Benchmarks grayscale + Canny at each requested size. If `source` is non-empty
// it is resized to each size; otherwise a synthetic image is used.
// One warm-up run is performed per size and excluded from the stats.
std::vector<SizeBenchmarkResult> benchmarkSizes(const std::vector<cv::Size>& sizes,
                                                int iterations,
                                                const cv::Mat& source = cv::Mat(),
                                                const EdgeParams& params = {});

void printSizeBenchmark(std::ostream& os, const std::vector<SizeBenchmarkResult>& results);
void writeSizeBenchmarkCsv(const std::string& path, const std::vector<SizeBenchmarkResult>& results);

// ---------- Sequential vs parallel scaling benchmark ----------

struct ScalingRun {
    std::string mode;      // "sequential", "sequential + OpenCV threads", "parallel"
    int workers = 1;       // images processed at the same time
    int opencvThreads = 1; // threads OpenCV may use inside each image (0 = OpenCV default)
    Stats wall;            // wall time for the whole set of images
    double imagesPerSecond = 0.0;
    double speedup = 1.0;  // relative to the plain sequential run
};

struct ScalingResult {
    int width = 0;
    int height = 0;
    int images = 0;
    std::vector<ScalingRun> runs;
};

// For each size, writes `imagesPerSize` images to a temporary folder and runs the
// full load -> gray -> Canny -> save pipeline over them with:
//   1. one image at a time, OpenCV single-threaded      (sequential baseline)
//   2. one image at a time, OpenCV's own threading      (library-level parallelism)
//   3. N images at a time, OpenCV single-threaded, for each N in workerCounts > 1
// Each mode gets one untimed warm-up run and `repeats` timed runs.
// OpenCV's global thread setting is restored afterwards.
std::vector<ScalingResult> benchmarkScaling(const std::vector<cv::Size>& sizes,
                                            int imagesPerSize,
                                            const std::vector<int>& workerCounts,
                                            int repeats,
                                            const cv::Mat& source = cv::Mat(),
                                            const EdgeParams& params = {});

void printScaling(std::ostream& os, const std::vector<ScalingResult>& results);
void writeScalingCsv(const std::string& path, const std::vector<ScalingResult>& results);

} // namespace imgproc
