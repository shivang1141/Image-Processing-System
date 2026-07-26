// Automated validation: missing files, invalid images, many image sizes,
// invalid parameters, and the properties every output must have.
#include "benchmark.hpp"
#include "processor.hpp"
#include "test_helpers.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

#include <string>
#include <vector>

using namespace imgproc;
using namespace imgproc::test;

// ---------- Missing files ----------

TEST(MissingFiles, LoadThrowsWithFileNotFound) {
    try {
        loadImage("definitely_not_here_12345.png");
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("File not found"), std::string::npos) << e.what();
    }
}

TEST(MissingFiles, ProcessFileThrowsAndWritesNothing) {
    TempDir dir;
    EXPECT_THROW(processFile((dir / "missing.png").string(), (dir / "out").string()), std::runtime_error);
    EXPECT_FALSE(fs::exists(dir / "out" / "missing_gray.png"));
    EXPECT_FALSE(fs::exists(dir / "out" / "missing_edges.png"));
}

TEST(MissingFiles, ProcessFolderRejectsMissingDirectory) {
    EXPECT_THROW(processFolder("no_such_dir_12345", "out"), std::runtime_error);
}

TEST(MissingFiles, ProcessFolderRejectsAFileGivenAsFolder) {
    TempDir dir;
    saveImage((dir / "a.png").string(), squareImage());
    EXPECT_THROW(processFolder((dir / "a.png").string(), (dir / "out").string()), std::runtime_error);
}

TEST(MissingFiles, LoadRejectsADirectory) {
    TempDir dir;
    fs::create_directories(dir / "looks_like_image.png");
    try {
        loadImage((dir / "looks_like_image.png").string());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Not a regular file"), std::string::npos) << e.what();
    }
}

// ---------- Invalid images ----------

// Each case is a file with an image extension whose contents cannot be decoded.
struct InvalidImageCase {
    std::string name;
    std::string fileName;
    std::string bytes;
};

class InvalidImages : public ::testing::TestWithParam<InvalidImageCase> {};

TEST_P(InvalidImages, LoadThrowsDecodeError) {
    TempDir dir;
    const fs::path path = dir / GetParam().fileName;
    writeBytes(path, GetParam().bytes);
    try {
        loadImage(path.string());
        FAIL() << "expected std::runtime_error";
    } catch (const std::runtime_error& e) {
        EXPECT_NE(std::string(e.what()).find("Failed to decode"), std::string::npos) << e.what();
    }
}

TEST_P(InvalidImages, BatchRecordsFailureAndContinues) {
    TempDir dir;
    writeBytes(dir / GetParam().fileName, GetParam().bytes);
    saveImage((dir / "good.png").string(), squareImage());

    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string());
    ASSERT_EQ(report.results.size(), 1u);
    EXPECT_EQ(fs::path(report.results[0].inputPath).filename(), "good.png");
    ASSERT_EQ(report.failures.size(), 1u);
    EXPECT_NE(report.failures[0].find(GetParam().fileName), std::string::npos);
}

std::string pngHeaderOnly() {
    std::vector<uchar> png;
    cv::imencode(".png", squareImage(), png);
    return std::string(png.begin(), png.begin() + 8); // just the 8-byte signature
}

std::string truncatedPng() {
    std::vector<uchar> png;
    cv::imencode(".png", squareImage(), png);
    return std::string(png.begin(), png.begin() + static_cast<long>(png.size() / 2));
}

INSTANTIATE_TEST_SUITE_P(
    Files, InvalidImages,
    ::testing::Values(
        InvalidImageCase{"EmptyFile", "empty.png", ""},
        InvalidImageCase{"GarbageBytes", "garbage.jpg", "this is not an image at all"},
        InvalidImageCase{"TextRenamedToBmp", "notes.bmp", "hello\nworld\n"},
        InvalidImageCase{"PngSignatureOnly", "header.png", pngHeaderOnly()},
        InvalidImageCase{"TruncatedPng", "truncated.png", truncatedPng()}),
    [](const auto& info) { return info.param.name; });

TEST(UnsupportedFiles, AreIgnoredByBatch) {
    TempDir dir;
    writeBytes(dir / "notes.txt", "not an image");
    writeBytes(dir / "data.csv", "a,b\n1,2\n");
    writeBytes(dir / "noext", "x");
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string());
    EXPECT_TRUE(report.results.empty());
    EXPECT_TRUE(report.failures.empty());
}

// ---------- Invalid input data and parameters ----------

TEST(InvalidInput, CannyRejects16BitImages) {
    EXPECT_THROW(detectEdges(cv::Mat(10, 10, CV_16UC1, cv::Scalar(1000))), std::invalid_argument);
}

TEST(InvalidInput, CannyRejectsFloatImages) {
    EXPECT_THROW(detectEdges(cv::Mat(10, 10, CV_32FC1, cv::Scalar(0.5))), std::invalid_argument);
}

TEST(InvalidParams, ValidateAcceptsDefaults) {
    EXPECT_NO_THROW(validateParams(EdgeParams{}));
}

TEST(InvalidParams, RejectsNegativeThresholds) {
    EdgeParams p;
    p.lowThreshold = -1;
    EXPECT_THROW(validateParams(p), std::invalid_argument);
    p = {};
    p.highThreshold = -1;
    EXPECT_THROW(validateParams(p), std::invalid_argument);
}

TEST(InvalidParams, RejectsNegativeBlur) {
    EdgeParams p;
    p.blurKernel = -3;
    EXPECT_THROW(validateParams(p), std::invalid_argument);
}

TEST(InvalidParams, RejectsBadApertures) {
    for (int aperture : {0, 1, 2, 4, 6, 8, 9}) {
        EdgeParams p;
        p.apertureSize = aperture;
        EXPECT_THROW(validateParams(p), std::invalid_argument) << "aperture " << aperture;
    }
    for (int aperture : {3, 5, 7}) {
        EdgeParams p;
        p.apertureSize = aperture;
        EXPECT_NO_THROW(validateParams(p)) << "aperture " << aperture;
    }
}

TEST(InvalidParams, DetectEdgesValidatesBeforeRunning) {
    EdgeParams p;
    p.apertureSize = 4; // would be a cv::Exception inside OpenCV without our check
    EXPECT_THROW(detectEdges(toGrayscale(squareImage()), p), std::invalid_argument);
}

TEST(InvalidParams, ProcessFolderFailsFastOnBadParams) {
    TempDir dir;
    saveImage((dir / "a.png").string(), squareImage());
    EdgeParams p;
    p.lowThreshold = -10;
    EXPECT_THROW(processFolder(dir.path().string(), (dir / "out").string(), p), std::invalid_argument);
}

// ---------- Output that cannot be written ----------

TEST(Unwritable, SaveToMissingFolderThrows) {
    TempDir dir;
    EXPECT_THROW(saveImage((dir / "no" / "such" / "dir" / "a.png").string(), squareImage()),
                 std::runtime_error);
}

TEST(Unwritable, SaveWithUnknownExtensionThrows) {
    TempDir dir;
    EXPECT_THROW(saveImage((dir / "a.unknownext").string(), squareImage()), std::runtime_error);
}

TEST(Unwritable, OutputDirThatIsAFileThrows) {
    TempDir dir;
    saveImage((dir / "in.png").string(), squareImage());
    writeBytes(dir / "out", "I am a file, not a folder");
    EXPECT_THROW(processFile((dir / "in.png").string(), (dir / "out").string()), std::exception);
}

// ---------- Different image sizes + expected output properties ----------

class ImageSizes : public ::testing::TestWithParam<cv::Size> {};

TEST_P(ImageSizes, PipelineOutputHasExpectedProperties) {
    const cv::Size size = GetParam();
    const cv::Mat image = makeSyntheticImage(size.width, size.height);
    ASSERT_EQ(image.size(), size);

    const cv::Mat gray = toGrayscale(image);
    EXPECT_EQ(gray.size(), size);
    EXPECT_EQ(gray.type(), CV_8UC1);

    // Grayscale of a BGR pixel is a weighted average, so it lies between the
    // channel min and max (allowing 1 for rounding).
    cv::Mat channels[3];
    cv::split(image, channels);
    cv::Mat lo = cv::min(cv::min(channels[0], channels[1]), channels[2]);
    cv::Mat hi = cv::max(cv::max(channels[0], channels[1]), channels[2]);
    cv::Mat loMinus, hiPlus;
    cv::subtract(lo, cv::Scalar(1), loMinus);
    cv::add(hi, cv::Scalar(1), hiPlus);
    EXPECT_EQ(cv::countNonZero(gray < loMinus), 0);
    EXPECT_EQ(cv::countNonZero(gray > hiPlus), 0);

    const cv::Mat edges = detectEdges(gray);
    EXPECT_EQ(edges.size(), size);
    EXPECT_EQ(edges.type(), CV_8UC1);
    EXPECT_TRUE(isBinary(edges));
    EXPECT_LT(cv::countNonZero(edges), static_cast<int>(edges.total())); // never "everything is an edge"
}

TEST_P(ImageSizes, SavedOutputsDecodeWithSameSize) {
    const cv::Size size = GetParam();
    TempDir dir;
    const fs::path input = dir / "in.png";
    saveImage(input.string(), makeSyntheticImage(size.width, size.height));

    const ProcessResult r = processFile(input.string(), (dir / "out").string());
    EXPECT_EQ(cv::Size(r.width, r.height), size);

    const cv::Mat gray = cv::imread((dir / "out" / "in_gray.png").string(), cv::IMREAD_UNCHANGED);
    const cv::Mat edges = cv::imread((dir / "out" / "in_edges.png").string(), cv::IMREAD_UNCHANGED);
    ASSERT_FALSE(gray.empty());
    ASSERT_FALSE(edges.empty());
    EXPECT_EQ(gray.size(), size);
    EXPECT_EQ(edges.size(), size);
    EXPECT_EQ(gray.channels(), 1);
    EXPECT_EQ(edges.channels(), 1);
    EXPECT_TRUE(isBinary(edges));
}

INSTANTIATE_TEST_SUITE_P(
    Sizes, ImageSizes,
    ::testing::Values(cv::Size(1, 1), cv::Size(2, 2), cv::Size(3, 7), cv::Size(17, 13),
                      cv::Size(1, 500), cv::Size(500, 1), cv::Size(640, 480),
                      cv::Size(1920, 1080), cv::Size(3840, 2160)),
    [](const auto& info) {
        return std::to_string(info.param.width) + "x" + std::to_string(info.param.height);
    });

TEST(ExpectedOutput, EdgesAreDeterministic) {
    const cv::Mat gray = toGrayscale(makeSyntheticImage(320, 240, 3));
    EXPECT_EQ(cv::norm(detectEdges(gray), detectEdges(gray), cv::NORM_INF), 0.0);
}

TEST(ExpectedOutput, EdgesSitOnIntensityChanges) {
    // Every edge pixel of the square must be within 2px of the true boundary.
    const cv::Mat edges = detectEdges(toGrayscale(squareImage(100)));
    for (int y = 0; y < edges.rows; ++y) {
        for (int x = 0; x < edges.cols; ++x) {
            if (edges.at<uchar>(y, x) == 0) continue;
            const int dx = std::min(std::abs(x - 25), std::abs(x - 75));
            const int dy = std::min(std::abs(y - 25), std::abs(y - 75));
            const bool nearVertical = dx <= 2 && y >= 23 && y <= 77;
            const bool nearHorizontal = dy <= 2 && x >= 23 && x <= 77;
            EXPECT_TRUE(nearVertical || nearHorizontal) << "edge at (" << x << "," << y << ")";
        }
    }
}

TEST(ExpectedOutput, ColorAndGrayInputsGiveSameEdges) {
    const cv::Mat color = makeSyntheticImage(200, 150, 9);
    const cv::Mat gray = toGrayscale(color);
    EXPECT_EQ(cv::norm(detectEdges(toGrayscale(color)), detectEdges(toGrayscale(gray)), cv::NORM_INF), 0.0);
}
