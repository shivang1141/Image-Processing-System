#pragma once

#include "processor.hpp"

#include <opencv2/core.hpp>

#include <iosfwd>
#include <string>
#include <vector>

namespace imgproc {

// Simple measurements of a binary edge map.
struct EdgeMetrics {
    int edgePixels = 0;     // number of non-zero pixels
    double densityPct = 0.0; // edgePixels as a percentage of all pixels
    int segments = 0;       // 8-connected edge fragments (lower = longer, cleaner outlines)
};

EdgeMetrics measureEdges(const cv::Mat& edges);

// One parameter combination and what it produced.
struct ParamTrial {
    EdgeParams params;
    EdgeMetrics metrics;
    double meanMs = 0.0; // blur + Canny, averaged over the repeats
    cv::Mat edges;
};

// Every blur x low x high combination, in that order. Combinations with
// low > high are skipped (Canny would just swap them).
std::vector<EdgeParams> buildParamGrid(const std::vector<int>& blurs,
                                       const std::vector<double>& lows,
                                       const std::vector<double>& highs,
                                       int apertureSize = 3);

// Runs detectEdges on `gray` for every parameter set; timings are averaged over `repeats` runs.
std::vector<ParamTrial> compareParams(const cv::Mat& gray,
                                      const std::vector<EdgeParams>& grid,
                                      int repeats = 5);

// Contact sheet for one blur value: rows are low thresholds, columns are high
// thresholds, each cell shows the edge map plus its edge density and segment count.
cv::Mat makeContactSheet(const std::vector<ParamTrial>& trials,
                         int blurKernel,
                         const std::vector<double>& lows,
                         const std::vector<double>& highs,
                         int thumbWidth = 320);

// Original -> grayscale -> blurred -> edges, side by side with labels.
cv::Mat makeStageStrip(const cv::Mat& bgr, const EdgeParams& params, int thumbWidth = 320);

void printComparison(std::ostream& os, const std::vector<ParamTrial>& trials);
void writeComparisonCsv(const std::string& path, const std::vector<ParamTrial>& trials);

} // namespace imgproc
