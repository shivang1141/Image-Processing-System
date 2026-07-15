#pragma once

#include <opencv2/core.hpp>

#include <string>
#include <vector>

namespace imgproc {

// Named size presets used by --sizes: small, medium, large.
cv::Size sizePreset(const std::string& name);

// Parses "640x480,1920x1080" or presets ("small,medium,large"), or a mix.
// Throws std::invalid_argument with a readable message on bad input.
std::vector<cv::Size> parseSizes(const std::string& text);

// Strict whole-string number parsing. `what` names the value in error messages.
int parseInt(const std::string& text, const std::string& what);
double parseDouble(const std::string& text, const std::string& what);

// Comma-separated lists, e.g. "1,2,4,8" or "25,50,100".
std::vector<int> parseIntList(const std::string& text, const std::string& what);
std::vector<double> parseDoubleList(const std::string& text, const std::string& what);

} // namespace imgproc
