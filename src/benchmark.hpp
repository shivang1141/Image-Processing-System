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
// BatchReport::failures and skipped.
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

} // namespace imgproc
