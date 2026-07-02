#pragma once

#include <opencv2/core.hpp>

#include <string>

namespace imgproc {

// Loads an image from disk. Throws std::runtime_error if it cannot be read.
cv::Mat loadImage(const std::string& path);

// Converts a BGR/BGRA image to single-channel grayscale.
// Already-grayscale input is returned as a copy.
cv::Mat toGrayscale(const cv::Mat& image);

// Saves an image to disk. Throws std::runtime_error on failure.
void saveImage(const std::string& path, const cv::Mat& image);

} // namespace imgproc
