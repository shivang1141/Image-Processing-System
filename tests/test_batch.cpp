// Folder processing (sequential and parallel), reports, CSV, and benchmarks.
#include "benchmark.hpp"
#include "processor.hpp"
#include "test_helpers.hpp"

#include <opencv2/core/utility.hpp>
#include <opencv2/imgcodecs.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

using namespace imgproc;
using namespace imgproc::test;

namespace {

int countLines(const fs::path& path) {
    std::ifstream in(path);
    int lines = 0;
    for (std::string line; std::getline(in, line);) ++lines;
    return lines;
}

void writeSamples(const TempDir& dir, int count, int width = 160, int height = 120) {
    for (int i = 0; i < count; ++i) {
        saveImage((dir / ("img" + std::to_string(i) + ".png")).string(),
                  makeSyntheticImage(width, height, static_cast<unsigned>(i + 1)));
    }
}

} // namespace

TEST(SupportedImage, RecognisesExtensionsCaseInsensitively) {
    EXPECT_TRUE(isSupportedImage("a.png"));
    EXPECT_TRUE(isSupportedImage("a.JPG"));
    EXPECT_TRUE(isSupportedImage("dir/b.jpeg"));
    EXPECT_TRUE(isSupportedImage("c.TiFf"));
    EXPECT_FALSE(isSupportedImage("a.txt"));
    EXPECT_FALSE(isSupportedImage("noext"));
    EXPECT_FALSE(isSupportedImage("png")); // a name, not an extension
}

TEST(ProcessFolder, AveragesTimingsAndCountsFailures) {
    TempDir dir;
    writeSamples(dir, 3);
    writeBytes(dir / "notes.txt", "not an image"); // ignored: unsupported extension
    writeBytes(dir / "broken.jpg", "garbage");     // supported extension, fails to load

    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string());
    ASSERT_EQ(report.results.size(), 3u);
    EXPECT_EQ(report.failures.size(), 1u);
    EXPECT_GT(report.wallMs, 0.0);
    EXPECT_GT(report.imagesPerSecond(), 0.0);

    double sumGray = 0.0;
    for (const auto& r : report.results) sumGray += r.timings.grayscaleMs;
    EXPECT_NEAR(report.average().grayscaleMs, sumGray / 3.0, 1e-9);
}

TEST(ProcessFolder, ResultsAreSortedByName) {
    TempDir dir;
    for (const char* name : {"c.png", "a.png", "b.png"}) {
        saveImage((dir / name).string(), squareImage(32));
    }
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string(), {}, 3);
    ASSERT_EQ(report.results.size(), 3u);
    EXPECT_EQ(fs::path(report.results[0].inputPath).filename(), "a.png");
    EXPECT_EQ(fs::path(report.results[1].inputPath).filename(), "b.png");
    EXPECT_EQ(fs::path(report.results[2].inputPath).filename(), "c.png");
}

TEST(ProcessFolder, EmptyFolderGivesEmptyReport) {
    TempDir dir;
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string());
    EXPECT_TRUE(report.results.empty());
    EXPECT_TRUE(report.failures.empty());
    EXPECT_EQ(report.average().totalMs(), 0.0);
    EXPECT_EQ(report.imagesPerSecond(), 0.0);
}

TEST(ProcessFolder, SameStemDifferentExtensionDoesNotOverwrite) {
    TempDir dir;
    saveImage((dir / "photo.png").string(), squareImage(40));
    saveImage((dir / "photo.bmp").string(), squareImage(60));
    saveImage((dir / "other.png").string(), squareImage(20));

    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string());
    ASSERT_EQ(report.results.size(), 3u);
    EXPECT_TRUE(fs::exists(dir / "out" / "photo_png_edges.png"));
    EXPECT_TRUE(fs::exists(dir / "out" / "photo_bmp_edges.png"));
    EXPECT_TRUE(fs::exists(dir / "out" / "other_edges.png")); // unique stems keep their plain name
    EXPECT_EQ(cv::imread((dir / "out" / "photo_png_gray.png").string()).cols, 40);
    EXPECT_EQ(cv::imread((dir / "out" / "photo_bmp_gray.png").string()).cols, 60);
}

TEST(ProcessFolder, CsvHasHeaderAndOneRowPerImage) {
    TempDir dir;
    writeSamples(dir, 3);
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string());
    writeBatchCsv((dir / "report.csv").string(), report);
    EXPECT_EQ(countLines(dir / "report.csv"), 4);
}

TEST(ProcessFolder, CsvToUnwritablePathThrows) {
    TempDir dir;
    EXPECT_THROW(writeBatchCsv((dir / "missing" / "report.csv").string(), BatchReport{}), std::runtime_error);
}

TEST(ProcessFolder, PrintedReportShowsAveragesWallTimeAndFailures) {
    TempDir dir;
    writeSamples(dir, 2);
    writeBytes(dir / "broken.png", "garbage");
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string(), {}, 2);
    std::ostringstream os;
    printBatchReport(os, report);
    const std::string text = os.str();
    EXPECT_NE(text.find("AVERAGE (2 images)"), std::string::npos);
    EXPECT_NE(text.find("2 worker(s)"), std::string::npos);
    EXPECT_NE(text.find("Failed (1)"), std::string::npos);
    EXPECT_NE(text.find("broken.png"), std::string::npos);
}

// ---------- Parallel processing ----------

TEST(Workers, ResolveZeroMeansOnePerCore) {
    EXPECT_EQ(resolveWorkers(0), static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
    EXPECT_EQ(resolveWorkers(3), 3);
    EXPECT_THROW(resolveWorkers(-1), std::invalid_argument);
}

TEST(Parallel, GivesIdenticalOutputsToSequential) {
    TempDir dir;
    writeSamples(dir, 12, 320, 240);
    writeBytes(dir / "broken.png", "garbage");

    const BatchReport seq = processFolder(dir.path().string(), (dir / "seq").string(), {}, 1);
    const BatchReport par = processFolder(dir.path().string(), (dir / "par").string(), {}, 4);

    EXPECT_EQ(seq.workers, 1);
    EXPECT_EQ(par.workers, 4);
    ASSERT_EQ(seq.results.size(), par.results.size());
    ASSERT_EQ(seq.failures, par.failures);
    for (size_t i = 0; i < seq.results.size(); ++i) {
        EXPECT_EQ(seq.results[i].inputPath, par.results[i].inputPath);
        const std::string stem = fs::path(seq.results[i].inputPath).stem().string();
        for (const char* suffix : {"_gray.png", "_edges.png"}) {
            EXPECT_EQ(readBytes(dir / "seq" / (stem + suffix)), readBytes(dir / "par" / (stem + suffix)))
                << stem << suffix;
        }
    }
}

TEST(Parallel, MoreWorkersThanImagesIsFine) {
    TempDir dir;
    writeSamples(dir, 2);
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string(), {}, 16);
    EXPECT_EQ(report.results.size(), 2u);
    EXPECT_TRUE(report.failures.empty());
}

TEST(Parallel, AllFailuresAreCollected) {
    TempDir dir;
    for (int i = 0; i < 8; ++i) {
        writeBytes(dir / ("bad" + std::to_string(i) + ".png"), "garbage");
    }
    const BatchReport report = processFolder(dir.path().string(), (dir / "out").string(), {}, 4);
    EXPECT_TRUE(report.results.empty());
    EXPECT_EQ(report.failures.size(), 8u);
}

// ---------- Benchmarks ----------

TEST(Stats, MinMeanMax) {
    const Stats s = computeStats({4.0, 1.0, 7.0});
    EXPECT_DOUBLE_EQ(s.minMs, 1.0);
    EXPECT_DOUBLE_EQ(s.meanMs, 4.0);
    EXPECT_DOUBLE_EQ(s.maxMs, 7.0);
    const Stats empty = computeStats({});
    EXPECT_EQ(empty.meanMs, 0.0);
}

TEST(SyntheticImage, IsDeterministicPerSeed) {
    const cv::Mat a = makeSyntheticImage(64, 48, 7);
    const cv::Mat b = makeSyntheticImage(64, 48, 7);
    const cv::Mat c = makeSyntheticImage(64, 48, 8);
    EXPECT_EQ(a.size(), cv::Size(64, 48));
    EXPECT_EQ(a.type(), CV_8UC3);
    EXPECT_EQ(cv::norm(a, b, cv::NORM_INF), 0.0);
    EXPECT_GT(cv::norm(a, c, cv::NORM_INF), 0.0);
}

TEST(SizeBenchmark, ReportsOrderedStatsPerSize) {
    const std::vector<cv::Size> sizes = {{64, 48}, {128, 96}};
    const auto results = benchmarkSizes(sizes, 3);
    ASSERT_EQ(results.size(), 2u);
    for (size_t i = 0; i < results.size(); ++i) {
        EXPECT_EQ(cv::Size(results[i].width, results[i].height), sizes[i]);
        EXPECT_EQ(results[i].iterations, 3);
        EXPECT_LE(results[i].edges.minMs, results[i].edges.meanMs);
        EXPECT_LE(results[i].edges.meanMs, results[i].edges.maxMs);
    }
    EXPECT_THROW(benchmarkSizes(sizes, 0), std::invalid_argument);
}

TEST(ScalingBenchmark, RunsEveryModeAndRestoresOpenCvThreads) {
    const int before = cv::getNumThreads();
    const auto results = benchmarkScaling({{96, 64}}, 4, {1, 2, 4}, 1);
    EXPECT_EQ(cv::getNumThreads(), before);

    ASSERT_EQ(results.size(), 1u);
    const auto& runs = results[0].runs;
    ASSERT_EQ(runs.size(), 4u); // sequential, sequential + OpenCV threads, 2 workers, 4 workers
    EXPECT_EQ(runs[0].mode, "sequential");
    EXPECT_DOUBLE_EQ(runs[0].speedup, 1.0);
    EXPECT_EQ(runs[0].opencvThreads, 1);
    EXPECT_EQ(runs[2].workers, 2);
    EXPECT_EQ(runs[3].workers, 4);
    for (const auto& run : runs) {
        EXPECT_GT(run.wall.meanMs, 0.0);
        EXPECT_GT(run.imagesPerSecond, 0.0);
    }
}

TEST(ScalingBenchmark, RejectsBadArguments) {
    EXPECT_THROW(benchmarkScaling({{32, 32}}, 0, {1}, 1), std::invalid_argument);
    EXPECT_THROW(benchmarkScaling({{32, 32}}, 1, {1}, 0), std::invalid_argument);
    EXPECT_THROW(benchmarkScaling({{32, 32}}, 1, {0}, 1), std::invalid_argument);
}

TEST(ScalingBenchmark, CsvHasOneRowPerRun) {
    TempDir dir;
    const auto results = benchmarkScaling({{64, 48}, {96, 64}}, 2, {1, 2}, 1);
    writeScalingCsv((dir / "scale.csv").string(), results);
    EXPECT_EQ(countLines(dir / "scale.csv"), 1 + 2 * 3); // header + 2 sizes x 3 modes
}
