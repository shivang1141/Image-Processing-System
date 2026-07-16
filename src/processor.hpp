#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace imgproc {

// Parameters for Canny edge detection.
struct EdgeParams {
    double lowThreshold = 50.0;
    double highThreshold = 150.0;
    int blurKernel = 5;   // Gaussian blur kernel size before Canny (0 disables blur)
    int apertureSize = 3; // Sobel aperture used by Canny (3, 5 or 7)
};

// Throws std::invalid_argument if the parameters cannot be used by Canny
// (negative thresholds or blur kernel, aperture not 3/5/7).
void validateParams(const EdgeParams& params);

// Wall-clock time spent in each stage of the pipeline, in milliseconds.
struct StageTimings {
    double loadMs = 0.0;
    double grayscaleMs = 0.0;
    double edgeMs = 0.0;
    double saveMs = 0.0;

    double processingMs() const { return grayscaleMs + edgeMs; }
    double totalMs() const { return loadMs + grayscaleMs + edgeMs + saveMs; }
};

// Loads an image from disk. Throws std::runtime_error if it cannot be read.
cv::Mat loadImage(const std::string& path);

// Converts a BGR/BGRA image to single-channel grayscale.
// Already-grayscale input is returned as a copy.
cv::Mat toGrayscale(const cv::Mat& image);

// Gaussian blur with a k x k kernel (even k is rounded up to the next odd size).
// k == 0 returns the input unchanged.
cv::Mat blurImage(const cv::Mat& gray, int kernel);

// Runs Canny edge detection on an 8-bit grayscale image and returns a binary edge map.
// Blurs first when params.blurKernel > 0.
cv::Mat detectEdges(const cv::Mat& gray, const EdgeParams& params = {});

// Saves an image to disk. Throws std::runtime_error on failure.
void saveImage(const std::string& path, const cv::Mat& image);

// Result of running the full pipeline on one image file.
struct ProcessResult {
    std::string inputPath;
    int width = 0;
    int height = 0;
    StageTimings timings;
};

// Load -> grayscale -> edges -> save (grayscale and edge outputs) for one file.
// Outputs are written to outputDir as <outputStem>_gray.png and <outputStem>_edges.png;
// an empty outputStem means the input file's stem.
ProcessResult processFile(const std::string& inputPath,
                          const std::string& outputDir,
                          const EdgeParams& params = {},
                          const std::string& outputStem = "");

} // namespace imgproc
