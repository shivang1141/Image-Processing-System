#include "processor.hpp"
#include "timer.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

namespace imgproc {

void validateParams(const EdgeParams& params) {
    if (params.lowThreshold < 0.0 || params.highThreshold < 0.0) {
        throw std::invalid_argument("Canny thresholds must be >= 0");
    }
    if (params.blurKernel < 0) {
        throw std::invalid_argument("Blur kernel must be >= 0 (0 disables blur)");
    }
    if (params.apertureSize != 3 && params.apertureSize != 5 && params.apertureSize != 7) {
        throw std::invalid_argument("Canny aperture must be 3, 5 or 7");
    }
}

cv::Mat loadImage(const std::string& path) {
    if (!fs::exists(path)) {
        throw std::runtime_error("File not found: " + path);
    }
    if (!fs::is_regular_file(path)) {
        throw std::runtime_error("Not a regular file: " + path);
    }
    cv::Mat image = cv::imread(path, cv::IMREAD_COLOR);
    if (image.empty()) {
        throw std::runtime_error("Failed to decode image (corrupt or unsupported format): " + path);
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

cv::Mat blurImage(const cv::Mat& gray, int kernel) {
    if (kernel < 0) {
        throw std::invalid_argument("blurImage: kernel must be >= 0");
    }
    if (kernel == 0) {
        return gray;
    }
    // Kernel must be odd; round up if needed.
    const int k = kernel | 1;
    cv::Mat blurred;
    cv::GaussianBlur(gray, blurred, cv::Size(k, k), 0);
    return blurred;
}

cv::Mat detectEdges(const cv::Mat& gray, const EdgeParams& params) {
    if (gray.empty() || gray.channels() != 1 || gray.depth() != CV_8U) {
        throw std::invalid_argument("detectEdges: expected a non-empty 8-bit single-channel image");
    }
    validateParams(params);

    cv::Mat edges;
    cv::Canny(blurImage(gray, params.blurKernel), edges,
              params.lowThreshold, params.highThreshold, params.apertureSize);
    return edges;
}

void saveImage(const std::string& path, const cv::Mat& image) {
    bool ok = false;
    try {
        ok = cv::imwrite(path, image);
    } catch (const cv::Exception& e) {
        throw std::runtime_error("Failed to save image: " + path + " (" + e.what() + ")");
    }
    if (!ok) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

ProcessResult processFile(const std::string& inputPath,
                          const std::string& outputDir,
                          const EdgeParams& params,
                          const std::string& outputStem) {
    validateParams(params);

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
    const std::string stem = outputStem.empty() ? fs::path(inputPath).stem().string() : outputStem;
    const fs::path out(outputDir);

    timer.reset();
    saveImage((out / (stem + "_gray.png")).string(), gray);
    saveImage((out / (stem + "_edges.png")).string(), edges);
    result.timings.saveMs = timer.elapsedMs();

    return result;
}

} // namespace imgproc
