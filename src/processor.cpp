#include "processor.hpp"
#include "timer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace imgproc {

cv::Mat loadImage(const std::string& path) {
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to load image: " + path);
    }
    return image;
}

cv::Mat toGrayscale(const cv::Mat& image) {
    if (image.empty()) {
        throw std::invalid_argument("toGrayscale: empty image");
    }

    cv::Mat gray;
    switch (image.channels()) {
        case 1: gray = image.clone(); break;
        case 3: cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY); break;
        case 4: cv::cvtColor(image, gray, cv::COLOR_BGRA2GRAY); break;
        default:
            throw std::invalid_argument("toGrayscale: unsupported channel count");
    }
    return gray;
}

cv::Mat detectEdges(const cv::Mat& gray, const EdgeParams& params) {
    if (gray.empty() || gray.channels() != 1) {
        throw std::invalid_argument("detectEdges: expected a non-empty single-channel image");
    }

    cv::Mat source = gray;
    if (params.blurKernel > 0) {
        // Kernel must be odd; round up if needed.
        int k = params.blurKernel | 1;
        cv::GaussianBlur(gray, source, cv::Size(k, k), 0);
    }

    cv::Mat edges;
    cv::Canny(source, edges, params.lowThreshold, params.highThreshold, params.apertureSize);
    return edges;
}

void saveImage(const std::string& path, const cv::Mat& image) {
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

ProcessResult processFile(const std::string& inputPath,
                          const std::string& outputDir,
                          const EdgeParams& params) {
    ProcessResult result;
    result.inputPath = inputPath;

    Timer timer;
    cv::Mat image = loadImage(inputPath);
    result.timings.loadMs = timer.elapsedMs();
    result.width = image.cols;
    result.height = image.rows;

    timer.reset();
    cv::Mat gray = toGrayscale(image);
    result.timings.grayscaleMs = timer.elapsedMs();

    timer.reset();
    cv::Mat edges = detectEdges(gray, params);
    result.timings.edgeMs = timer.elapsedMs();

    fs::create_directories(outputDir);
    const std::string stem = fs::path(inputPath).stem().string();
    const fs::path out(outputDir);

    timer.reset();
    saveImage((out / (stem + "_gray.png")).string(), gray);
    saveImage((out / (stem + "_edges.png")).string(), edges);
    result.timings.saveMs = timer.elapsedMs();

    return result;
}

} // namespace imgproc
