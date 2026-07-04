#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace imgproc {

// Parameters for Canny edge detection.
struct EdgeParams {
    double lowThreshold = 50.0;
    double highThreshold = 150.0;
    int blurKernel = 5;   // Gaussian blur kernel size before Canny (0 disables blur)
    int apertureSize = 3; // Sobel aperture used by Canny
};

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

// Runs Canny edge detection on a grayscale image and returns a binary edge map.
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
// Outputs are written to outputDir as <stem>_gray.png and <stem>_edges.png.
ProcessResult processFile(const std::string& inputPath,
                          const std::string& outputDir,
                          const EdgeParams& params = {});

} // namespace imgproc
