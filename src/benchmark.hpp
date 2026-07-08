#pragma once

#include "processor.hpp"

#include <opencv2/core.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace imgproc {

// ---------- Folder (batch) processing ----------

struct BatchReport {
    std::vector<ProcessResult> results;
    std::vector<std::string> failures; // "path: reason"

    StageTimings average() const;
};

// True for file extensions OpenCV can typically decode (.png, .jpg, .bmp, ...).
bool isSupportedImage(const std::string& path);

// Processes every supported image in inputDir (non-recursive, sorted by name).
// Files that fail to load are recorded in BatchReport::failures and skipped.
BatchReport processFolder(const std::string& inputDir,
                          const std::string& outputDir,
                          const EdgeParams& params = {});

void printBatchReport(std::ostream& os, const BatchReport& report);
void writeBatchCsv(const std::string& path, const BatchReport& report);

// ---------- Image-size benchmark ----------

struct Stats {
    double meanMs = 0.0;
    double minMs = 0.0;
    double maxMs = 0.0;
};

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
