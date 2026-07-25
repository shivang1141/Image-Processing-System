// Core pipeline stages: grayscale, blur, Canny, load/save, single-file pipeline.
#include "processor.hpp"
#include "test_helpers.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

using namespace imgproc;
using namespace imgproc::test;

// ---------- Grayscale ----------

TEST(Grayscale, OutputIsSingleChannelSameSize) {
    const cv::Mat gray = toGrayscale(cv::Mat(48, 64, CV_8UC3, cv::Scalar(10, 20, 30)));
    EXPECT_EQ(gray.channels(), 1);
    EXPECT_EQ(gray.size(), cv::Size(64, 48));
    EXPECT_EQ(gray.type(), CV_8UC1);
}

TEST(Grayscale, UsesLuminanceWeights) {
    // Pure white/black stay white/black; pure blue uses the 0.114 weight, red 0.299, green 0.587.
    EXPECT_EQ(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(255))).at<uchar>(0, 0), 255);
    EXPECT_EQ(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(0))).at<uchar>(0, 0), 0);
    EXPECT_NEAR(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar(255, 0, 0))).at<uchar>(0, 0), 29, 1);
    EXPECT_NEAR(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar(0, 255, 0))).at<uchar>(0, 0), 150, 1);
    EXPECT_NEAR(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar(0, 0, 255))).at<uchar>(0, 0), 76, 1);
}

TEST(Grayscale, HandlesBgraInput) {
    const cv::Mat gray = toGrayscale(cv::Mat(10, 10, CV_8UC4, cv::Scalar(0, 0, 0, 255)));
    EXPECT_EQ(gray.channels(), 1);
    EXPECT_EQ(gray.at<uchar>(5, 5), 0);
}

TEST(Grayscale, GrayInputIsCopiedNotAliased) {
    const cv::Mat gray(10, 10, CV_8UC1, cv::Scalar(77));
    const cv::Mat out = toGrayscale(gray);
    EXPECT_EQ(out.channels(), 1);
    EXPECT_EQ(out.at<uchar>(5, 5), 77);
    EXPECT_NE(out.data, gray.data);
}

TEST(Grayscale, RejectsEmptyImage) {
    EXPECT_THROW(toGrayscale(cv::Mat()), std::invalid_argument);
}

TEST(Grayscale, RejectsTwoChannelImage) {
    EXPECT_THROW(toGrayscale(cv::Mat(4, 4, CV_8UC2, cv::Scalar::all(0))), std::invalid_argument);
}

// ---------- Blur ----------

TEST(Blur, ZeroKernelReturnsInputUnchanged) {
    const cv::Mat gray = toGrayscale(squareImage());
    EXPECT_EQ(cv::norm(blurImage(gray, 0), gray, cv::NORM_INF), 0.0);
}

TEST(Blur, SmoothsSharpEdges) {
    const cv::Mat gray = toGrayscale(squareImage());
    const cv::Mat blurred = blurImage(gray, 5);
    EXPECT_EQ(blurred.size(), gray.size());
    // A pixel just outside the square picks up some brightness from inside it.
    EXPECT_GT(blurred.at<uchar>(50, 24), 0);
    EXPECT_LT(blurred.at<uchar>(50, 25), 255);
}

TEST(Blur, EvenKernelIsRoundedUpToOdd) {
    const cv::Mat gray = toGrayscale(squareImage());
    EXPECT_EQ(cv::norm(blurImage(gray, 4), blurImage(gray, 5), cv::NORM_INF), 0.0);
}

TEST(Blur, RejectsNegativeKernel) {
    EXPECT_THROW(blurImage(toGrayscale(squareImage()), -1), std::invalid_argument);
}

// ---------- Canny ----------

TEST(Canny, UniformImageHasNoEdges) {
    EXPECT_EQ(cv::countNonZero(detectEdges(cv::Mat(100, 100, CV_8UC1, cv::Scalar(128)))), 0);
}

TEST(Canny, FindsOnlyTheSquareBoundary) {
    const cv::Mat edges = detectEdges(toGrayscale(squareImage()));
    EXPECT_EQ(edges.type(), CV_8UC1);
    EXPECT_GT(cv::countNonZero(edges), 100); // perimeter of a 50px square is ~200px
    EXPECT_EQ(edges.at<uchar>(50, 50), 0);   // interior
    EXPECT_EQ(edges.at<uchar>(5, 5), 0);     // far corner
    EXPECT_TRUE(isBinary(edges));
}

TEST(Canny, WorksWithBlurDisabled) {
    EdgeParams p;
    p.blurKernel = 0;
    EXPECT_GT(cv::countNonZero(detectEdges(toGrayscale(squareImage()), p)), 100);
}

TEST(Canny, RejectsMultiChannelInput) {
    EXPECT_THROW(detectEdges(squareImage()), std::invalid_argument);
}

TEST(Canny, RejectsEmptyInput) {
    EXPECT_THROW(detectEdges(cv::Mat()), std::invalid_argument);
}

// ---------- Load / save / single-file pipeline ----------

TEST(LoadSave, RoundTripsAPng) {
    TempDir dir;
    const cv::Mat original = squareImage(40);
    saveImage((dir / "a.png").string(), original);
    const cv::Mat loaded = loadImage((dir / "a.png").string());
    EXPECT_EQ(cv::norm(loaded, original, cv::NORM_INF), 0.0); // PNG is lossless
}

TEST(ProcessFile, WritesGrayAndEdgeOutputs) {
    TempDir dir;
    const fs::path input = dir / "square.png";
    saveImage(input.string(), squareImage());

    const ProcessResult r = processFile(input.string(), (dir / "out").string());
    EXPECT_EQ(r.width, 100);
    EXPECT_EQ(r.height, 100);
    EXPECT_GE(r.timings.grayscaleMs, 0.0);
    EXPECT_GE(r.timings.edgeMs, 0.0);
    EXPECT_GE(r.timings.totalMs(), r.timings.processingMs());
    ASSERT_TRUE(fs::exists(dir / "out" / "square_gray.png"));
    ASSERT_TRUE(fs::exists(dir / "out" / "square_edges.png"));

    const cv::Mat edges = cv::imread((dir / "out" / "square_edges.png").string(), cv::IMREAD_UNCHANGED);
    EXPECT_GT(cv::countNonZero(edges), 100);
}

TEST(ProcessFile, UsesCustomOutputStem) {
    TempDir dir;
    saveImage((dir / "a.png").string(), squareImage());
    processFile((dir / "a.png").string(), (dir / "out").string(), {}, "custom");
    EXPECT_TRUE(fs::exists(dir / "out" / "custom_gray.png"));
    EXPECT_TRUE(fs::exists(dir / "out" / "custom_edges.png"));
}
