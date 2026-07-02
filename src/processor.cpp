#include "processor.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <stdexcept>

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

void saveImage(const std::string& path, const cv::Mat& image) {
    if (!cv::imwrite(path, image)) {
        throw std::runtime_error("Failed to save image: " + path);
    }
}

} // namespace imgproc
