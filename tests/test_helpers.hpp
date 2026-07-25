#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>

namespace imgproc::test {

namespace fs = std::filesystem;

// Fresh, empty scratch directory under the system temp folder, removed on destruction.
// Named after the running test so parallel ctest runs do not collide.
class TempDir {
public:
    TempDir() {
        const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
        std::string name = std::string(info->test_suite_name()) + "_" + info->name();
        for (char& c : name) {
            if (c == '/') c = '_';
        }
        path_ = fs::temp_directory_path() / ("imgproc_test_" + name);
        fs::remove_all(path_);
        fs::create_directories(path_);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(path_, ec);
    }
    TempDir(const TempDir&) = delete;
    TempDir& operator=(const TempDir&) = delete;

    const fs::path& path() const { return path_; }
    fs::path operator/(const std::string& name) const { return path_ / name; }

private:
    fs::path path_;
};

// Black image with a white filled square in the middle.
inline cv::Mat squareImage(int size = 100) {
    cv::Mat img(size, size, CV_8UC3, cv::Scalar::all(0));
    cv::rectangle(img, cv::Point(size / 4, size / 4), cv::Point(3 * size / 4, 3 * size / 4),
                  cv::Scalar::all(255), cv::FILLED);
    return img;
}

inline void writeBytes(const fs::path& path, const std::string& bytes) {
    std::ofstream out(path, std::ios::binary);
    out << bytes;
}

inline std::string readBytes(const fs::path& path) {
    std::ifstream in(path, std::ios::binary);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

// True if every pixel is 0 or 255.
inline bool isBinary(const cv::Mat& m) {
    return cv::countNonZero((m != 0) & (m != 255)) == 0;
}

} // namespace imgproc::test
