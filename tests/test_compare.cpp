// Parameter comparison: edge metrics, parameter grids, and how parameters change the edges.
#include "benchmark.hpp"
#include "compare.hpp"
#include "processor.hpp"
#include "test_helpers.hpp"

#include <opencv2/imgproc.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <vector>

using namespace imgproc;
using namespace imgproc::test;

namespace {

// Flat gray image with a square and strong Gaussian noise: blur matters a lot here.
cv::Mat noisySquare(int size = 200, double sigma = 25.0) {
    cv::Mat img(size, size, CV_8UC1, cv::Scalar(100));
    cv::rectangle(img, {size / 4, size / 4}, {3 * size / 4, 3 * size / 4}, cv::Scalar(180), cv::FILLED);
    cv::Mat noise(img.size(), CV_16SC1);
    cv::RNG rng(1);
    rng.fill(noise, cv::RNG::NORMAL, 0, sigma);
    cv::Mat out;
    cv::add(img, noise, out, cv::noArray(), CV_8UC1);
    return out;
}

int edgesWith(const cv::Mat& gray, int blur, double low, double high) {
    EdgeParams p;
    p.blurKernel = blur;
    p.lowThreshold = low;
    p.highThreshold = high;
    return cv::countNonZero(detectEdges(gray, p));
}

} // namespace

// ---------- Metrics ----------

TEST(EdgeMetrics, CountsPixelsDensityAndSegments) {
    cv::Mat edges(10, 10, CV_8UC1, cv::Scalar(0));
    edges.row(2).colRange(0, 5) = 255; // one horizontal segment of 5px
    edges.at<uchar>(8, 8) = 255;       // a separate single pixel
    const EdgeMetrics m = measureEdges(edges);
    EXPECT_EQ(m.edgePixels, 6);
    EXPECT_DOUBLE_EQ(m.densityPct, 6.0);
    EXPECT_EQ(m.segments, 2);
}

TEST(EdgeMetrics, DiagonalNeighboursAreOneSegment) {
    cv::Mat edges(5, 5, CV_8UC1, cv::Scalar(0));
    for (int i = 0; i < 5; ++i) edges.at<uchar>(i, i) = 255;
    EXPECT_EQ(measureEdges(edges).segments, 1);
}

TEST(EdgeMetrics, EmptyMapHasNoSegments) {
    const EdgeMetrics m = measureEdges(cv::Mat(10, 10, CV_8UC1, cv::Scalar(0)));
    EXPECT_EQ(m.edgePixels, 0);
    EXPECT_EQ(m.segments, 0);
}

TEST(EdgeMetrics, RejectsInvalidInput) {
    EXPECT_THROW(measureEdges(cv::Mat()), std::invalid_argument);
    EXPECT_THROW(measureEdges(cv::Mat(4, 4, CV_8UC3)), std::invalid_argument);
}

// ---------- Grid ----------

TEST(ParamGrid, BuildsEveryValidCombinationInOrder) {
    const auto grid = buildParamGrid({0, 5}, {25, 100}, {50, 150});
    // low 100 > high 50 is skipped for each blur: 2 blurs x 3 combinations.
    ASSERT_EQ(grid.size(), 6u);
    EXPECT_EQ(grid[0].blurKernel, 0);
    EXPECT_EQ(grid[0].lowThreshold, 25);
    EXPECT_EQ(grid[0].highThreshold, 50);
    EXPECT_EQ(grid[2].lowThreshold, 100);
    EXPECT_EQ(grid[2].highThreshold, 150);
    EXPECT_EQ(grid[3].blurKernel, 5);
    for (const auto& p : grid) {
        EXPECT_LE(p.lowThreshold, p.highThreshold);
    }
}

TEST(ParamGrid, RejectsInvalidValues) {
    EXPECT_THROW(buildParamGrid({-1}, {50}, {150}), std::invalid_argument);
    EXPECT_THROW(buildParamGrid({0}, {-5}, {150}), std::invalid_argument);
    EXPECT_THROW(buildParamGrid({0}, {50}, {150}, 4), std::invalid_argument);
}

// ---------- How parameters affect the edges ----------

TEST(ParameterEffects, HigherHighThresholdNeverAddsEdges) {
    // Canny keeps a weak pixel only if it connects to a strong one; raising the
    // high threshold removes strong seeds, so the edge set can only shrink.
    const cv::Mat gray = toGrayscale(makeSyntheticImage(320, 240));
    int previous = edgesWith(gray, 5, 30, 60);
    for (double high : {100.0, 150.0, 200.0, 300.0}) {
        const int now = edgesWith(gray, 5, 30, high);
        EXPECT_LE(now, previous) << "high " << high;
        previous = now;
    }
}

TEST(ParameterEffects, HigherLowThresholdNeverAddsEdges) {
    const cv::Mat gray = toGrayscale(makeSyntheticImage(320, 240));
    int previous = edgesWith(gray, 5, 10, 200);
    for (double low : {30.0, 60.0, 100.0, 200.0}) {
        const int now = edgesWith(gray, 5, low, 200);
        EXPECT_LE(now, previous) << "low " << low;
        previous = now;
    }
}

TEST(ParameterEffects, BlurSuppressesNoiseEdges) {
    const cv::Mat gray = noisySquare();
    const int noBlur = edgesWith(gray, 0, 50, 150);
    const int blur5 = edgesWith(gray, 5, 50, 150);
    const int blur9 = edgesWith(gray, 9, 50, 150);
    EXPECT_GT(noBlur, 2 * blur5) << "noise should dominate without blur";
    EXPECT_LT(blur9, blur5);
}

TEST(ParameterEffects, BlurReducesFragmentation) {
    const cv::Mat gray = noisySquare();
    EdgeParams noBlur;
    noBlur.blurKernel = 0;
    EdgeParams blur9;
    blur9.blurKernel = 9;
    EXPECT_GT(measureEdges(detectEdges(gray, noBlur)).segments,
              measureEdges(detectEdges(gray, blur9)).segments);
}

// ---------- Comparison run and outputs ----------

TEST(CompareParams, ProducesOneTrialPerParameterSet) {
    const cv::Mat gray = toGrayscale(makeSyntheticImage(160, 120));
    const auto grid = buildParamGrid({0, 5}, {50}, {100, 150});
    const auto trials = compareParams(gray, grid, 2);
    ASSERT_EQ(trials.size(), grid.size());
    for (size_t i = 0; i < trials.size(); ++i) {
        EXPECT_EQ(trials[i].params.blurKernel, grid[i].blurKernel);
        EXPECT_EQ(trials[i].edges.size(), gray.size());
        EXPECT_EQ(trials[i].metrics.edgePixels, cv::countNonZero(trials[i].edges));
        EXPECT_GE(trials[i].meanMs, 0.0);
    }
    EXPECT_THROW(compareParams(gray, grid, 0), std::invalid_argument);
    EXPECT_TRUE(compareParams(gray, {}, 1).empty());
}

TEST(CompareParams, ContactSheetHasOneCellPerCombination) {
    const cv::Mat gray = toGrayscale(makeSyntheticImage(160, 120));
    const std::vector<double> lows = {25, 100}, highs = {50, 150, 250};
    const auto trials = compareParams(gray, buildParamGrid({3}, lows, highs), 1);
    const cv::Mat sheet = makeContactSheet(trials, 3, lows, highs, 100);
    EXPECT_EQ(sheet.type(), CV_8UC3);
    EXPECT_GT(sheet.cols, 3 * 100);
    EXPECT_GT(sheet.rows, 2 * 75);
    EXPECT_THROW(makeContactSheet(trials, 7, lows, highs), std::invalid_argument); // no blur-7 trials
}

TEST(CompareParams, StageStripShowsFourPanels) {
    const cv::Mat strip = makeStageStrip(makeSyntheticImage(200, 100), EdgeParams{}, 100);
    EXPECT_EQ(strip.type(), CV_8UC3);
    EXPECT_GT(strip.cols, 4 * 100);
    EXPECT_GT(strip.rows, 50);
}

TEST(CompareParams, PrintAndCsvListEveryTrial) {
    TempDir dir;
    const cv::Mat gray = toGrayscale(makeSyntheticImage(96, 64));
    const auto trials = compareParams(gray, buildParamGrid({0, 5}, {50}, {150}), 1);

    std::ostringstream os;
    printComparison(os, trials);
    EXPECT_NE(os.str().find("no blur"), std::string::npos);
    EXPECT_NE(os.str().find("blur 5x5"), std::string::npos);

    writeComparisonCsv((dir / "c.csv").string(), trials);
    std::ifstream in(dir / "c.csv");
    int lines = 0;
    for (std::string line; std::getline(in, line);) ++lines;
    EXPECT_EQ(lines, 3);
}
