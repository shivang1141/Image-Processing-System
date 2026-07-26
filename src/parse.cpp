#include "parse.hpp"

#include <algorithm>
#include <cctype>
#include <sstream>
#include <stdexcept>

namespace imgproc {

namespace {

std::vector<std::string> splitList(const std::string& text, const std::string& what) {
    if (!text.empty() && text.back() == ',') {
        throw std::invalid_argument("Empty entry in " + what + " list: '" + text + "'");
    }
    std::vector<std::string> items;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) {
            throw std::invalid_argument("Empty entry in " + what + " list: '" + text + "'");
        }
        items.push_back(item);
    }
    if (items.empty()) {
        throw std::invalid_argument("Empty " + what + " list");
    }
    return items;
}

std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return s;
}

} // namespace

cv::Size sizePreset(const std::string& name) {
    const std::string n = lower(name);
    if (n == "small") return {640, 480};
    if (n == "medium") return {1920, 1080};
    if (n == "large") return {3840, 2160};
    throw std::invalid_argument("Unknown size preset '" + name + "' (use small, medium, large or WxH)");
}

int parseInt(const std::string& text, const std::string& what) {
    size_t used = 0;
    int value = 0;
    try {
        value = std::stoi(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid " + what + ": '" + text + "' (expected an integer)");
    }
    if (used != text.size()) {
        throw std::invalid_argument("Invalid " + what + ": '" + text + "' (expected an integer)");
    }
    return value;
}

double parseDouble(const std::string& text, const std::string& what) {
    size_t used = 0;
    double value = 0.0;
    try {
        value = std::stod(text, &used);
    } catch (const std::exception&) {
        throw std::invalid_argument("Invalid " + what + ": '" + text + "' (expected a number)");
    }
    if (used != text.size()) {
        throw std::invalid_argument("Invalid " + what + ": '" + text + "' (expected a number)");
    }
    return value;
}

std::vector<cv::Size> parseSizes(const std::string& text) {
    std::vector<cv::Size> sizes;
    for (const auto& item : splitList(text, "size")) {
        const auto x = item.find_first_of("xX");
        if (x == std::string::npos) {
            sizes.push_back(sizePreset(item));
            continue;
        }
        const int w = parseInt(item.substr(0, x), "width in '" + item + "'");
        const int h = parseInt(item.substr(x + 1), "height in '" + item + "'");
        if (w <= 0 || h <= 0) {
            throw std::invalid_argument("Size must be positive: " + item);
        }
        sizes.emplace_back(w, h);
    }
    return sizes;
}

std::vector<int> parseIntList(const std::string& text, const std::string& what) {
    std::vector<int> values;
    for (const auto& item : splitList(text, what)) {
        values.push_back(parseInt(item, what));
    }
    return values;
}

std::vector<double> parseDoubleList(const std::string& text, const std::string& what) {
    std::vector<double> values;
    for (const auto& item : splitList(text, what)) {
        values.push_back(parseDouble(item, what));
    }
    return values;
}

} // namespace imgproc
