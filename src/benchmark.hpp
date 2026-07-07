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

} // namespace imgproc
