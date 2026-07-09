// Self-contained tests (no external framework) run through CTest.
#include "benchmark.hpp"
#include "processor.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace imgproc;

namespace {

int g_failures = 0;

#define CHECK(cond)                                                                     \
    do {                                                                                \
        if (!(cond)) {                                                                  \
            std::cerr << "  CHECK failed: " #cond " (" << __FILE__ << ':' << __LINE__  \
                      << ")\n";                                                         \
            ++g_failures;                                                               \
        }                                                                               \
    } while (0)

template <typename Ex, typename Fn>
bool throwsType(Fn&& fn) {
    try {
        fn();
    } catch (const Ex&) {
        return true;
    } catch (...) {
        return false;
    }
    return false;
}

// Fresh, empty scratch directory under the system temp folder.
fs::path makeTempDir(const std::string& name) {
    fs::path dir = fs::temp_directory_path() / ("imgproc_test_" + name);
    fs::remove_all(dir);
    fs::create_directories(dir);
    return dir;
}

// Black image with a white filled square in the middle.
cv::Mat squareImage(int size = 100) {
    cv::Mat img(size, size, CV_8UC3, cv::Scalar::all(0));
    cv::rectangle(img, cv::Point(size / 4, size / 4), cv::Point(3 * size / 4, 3 * size / 4),
                  cv::Scalar::all(255), cv::FILLED);
    return img;
}

// ---------- Tests ----------

void testGrayscaleChannelsAndSize() {
    cv::Mat color(48, 64, CV_8UC3, cv::Scalar(10, 20, 30));
    cv::Mat gray = toGrayscale(color);
    CHECK(gray.channels() == 1);
    CHECK(gray.rows == 48 && gray.cols == 64);
    CHECK(gray.type() == CV_8UC1);
}

void testGrayscaleValues() {
    // Pure white/black must stay white/black; pure blue uses the 0.114 weight.
    CHECK(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(255))).at<uchar>(0, 0) == 255);
    CHECK(toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar::all(0))).at<uchar>(0, 0) == 0);
    const int blue = toGrayscale(cv::Mat(4, 4, CV_8UC3, cv::Scalar(255, 0, 0))).at<uchar>(0, 0);
    CHECK(blue >= 28 && blue <= 30);
}

void testGrayscaleHandlesBgraAndGray() {
    cv::Mat bgra(10, 10, CV_8UC4, cv::Scalar(0, 0, 0, 255));
    CHECK(toGrayscale(bgra).channels() == 1);

    cv::Mat gray(10, 10, CV_8UC1, cv::Scalar(77));
    cv::Mat out = toGrayscale(gray);
    CHECK(out.channels() == 1 && out.at<uchar>(5, 5) == 77);
    CHECK(out.data != gray.data); // a copy, not an alias
}

void testGrayscaleRejectsEmpty() {
    CHECK(throwsType<std::invalid_argument>([] { toGrayscale(cv::Mat()); }));
}

void testEdgesOnUniformImageAreEmpty() {
    cv::Mat flat(100, 100, CV_8UC1, cv::Scalar(128));
    CHECK(cv::countNonZero(detectEdges(flat)) == 0);
}

void testEdgesFindSquareBoundary() {
    cv::Mat edges = detectEdges(toGrayscale(squareImage()));
    CHECK(edges.type() == CV_8UC1);
    CHECK(cv::countNonZero(edges) > 100); // perimeter of a 50px square is ~200px

    // Interior and far corners of the square should be edge-free.
    CHECK(edges.at<uchar>(50, 50) == 0);
    CHECK(edges.at<uchar>(5, 5) == 0);

    // Output is binary (0 or 255 only).
    cv::Mat nonBinary = (edges != 0) & (edges != 255);
    CHECK(cv::countNonZero(nonBinary) == 0);
}

void testEdgesWithoutBlur() {
    EdgeParams p;
    p.blurKernel = 0;
    CHECK(cv::countNonZero(detectEdges(toGrayscale(squareImage()), p)) > 100);
}

void testEdgesRejectsMultiChannel() {
    CHECK(throwsType<std::invalid_argument>([] { detectEdges(squareImage()); }));
}

void testLoadMissingFileThrows() {
    CHECK(throwsType<std::runtime_error>([] { loadImage("definitely_not_here_12345.png"); }));
}

void testProcessFileWritesOutputs() {
    const fs::path dir = makeTempDir("process");
    const fs::path input = dir / "square.png";
    saveImage(input.string(), squareImage());

    const ProcessResult r = processFile(input.string(), (dir / "out").string());
    CHECK(r.width == 100 && r.height == 100);
    CHECK(r.timings.grayscaleMs >= 0.0 && r.timings.edgeMs >= 0.0);
    CHECK(r.timings.totalMs() >= r.timings.processingMs());
    CHECK(fs::exists(dir / "out" / "square_gray.png"));
    CHECK(fs::exists(dir / "out" / "square_edges.png"));

    cv::Mat edges = loadImage((dir / "out" / "square_edges.png").string());
    CHECK(cv::countNonZero(toGrayscale(edges)) > 100);
    fs::remove_all(dir);
}

void testProcessFolderAveragesAndFailures() {
    const fs::path dir = makeTempDir("batch");
    for (int i = 0; i < 3; ++i) {
        saveImage((dir / ("img" + std::to_string(i) + ".png")).string(),
                  makeSyntheticImage(160, 120, static_cast<unsigned>(i)));
    }
    std::ofstream(dir / "notes.txt") << "not an image";     // ignored: unsupported extension
    std::ofstream(dir / "broken.jpg") << "garbage bytes";   // supported extension, fails to load

    const BatchReport report = processFolder(dir.string(), (dir / "out").string());
    CHECK(report.results.size() == 3);
    CHECK(report.failures.size() == 1);

    double sumGray = 0.0;
    for (const auto& r : report.results) sumGray += r.timings.grayscaleMs;
    const StageTimings avg = report.average();
    CHECK(std::abs(avg.grayscaleMs - sumGray / 3.0) < 1e-9);

    const fs::path csv = dir / "report.csv";
    writeBatchCsv(csv.string(), report);
    std::ifstream in(csv);
    int lines = 0;
    for (std::string line; std::getline(in, line);) ++lines;
    CHECK(lines == 4); // header + 3 rows
    fs::remove_all(dir);
}

void testProcessFolderRejectsNonDirectory() {
    CHECK(throwsType<std::runtime_error>([] { processFolder("no_such_dir_12345", "out"); }));
}

void testSupportedExtensions() {
    CHECK(isSupportedImage("a.png"));
    CHECK(isSupportedImage("a.JPG"));
    CHECK(isSupportedImage("dir/b.jpeg"));
    CHECK(!isSupportedImage("a.txt"));
    CHECK(!isSupportedImage("noext"));
}

void testSyntheticImageIsDeterministic() {
    cv::Mat a = makeSyntheticImage(64, 48, 7);
    cv::Mat b = makeSyntheticImage(64, 48, 7);
    CHECK(a.size() == cv::Size(64, 48) && a.type() == CV_8UC3);
    CHECK(cv::norm(a, b, cv::NORM_INF) == 0.0);
}

void testBenchmarkSizes() {
    const std::vector<cv::Size> sizes = {{64, 48}, {128, 96}};
    const auto results = benchmarkSizes(sizes, 3);
    CHECK(results.size() == 2);
    for (size_t i = 0; i < results.size(); ++i) {
        CHECK(results[i].width == sizes[i].width && results[i].height == sizes[i].height);
        CHECK(results[i].iterations == 3);
        CHECK(results[i].edges.minMs <= results[i].edges.meanMs);
        CHECK(results[i].edges.meanMs <= results[i].edges.maxMs);
    }
    CHECK(throwsType<std::invalid_argument>([&] { benchmarkSizes(sizes, 0); }));
}

} // namespace

int main() {
    const std::vector<std::pair<std::string, std::function<void()>>> tests = {
        {"grayscale channels and size", testGrayscaleChannelsAndSize},
        {"grayscale values", testGrayscaleValues},
        {"grayscale handles BGRA and gray", testGrayscaleHandlesBgraAndGray},
        {"grayscale rejects empty", testGrayscaleRejectsEmpty},
        {"edges on uniform image are empty", testEdgesOnUniformImageAreEmpty},
        {"edges find square boundary", testEdgesFindSquareBoundary},
        {"edges without blur", testEdgesWithoutBlur},
        {"edges rejects multi-channel", testEdgesRejectsMultiChannel},
        {"load missing file throws", testLoadMissingFileThrows},
        {"processFile writes outputs", testProcessFileWritesOutputs},
        {"processFolder averages and failures", testProcessFolderAveragesAndFailures},
        {"processFolder rejects non-directory", testProcessFolderRejectsNonDirectory},
        {"supported extensions", testSupportedExtensions},
        {"synthetic image is deterministic", testSyntheticImageIsDeterministic},
        {"benchmark sizes", testBenchmarkSizes},
    };

    int failedTests = 0;
    for (const auto& [name, fn] : tests) {
        const int before = g_failures;
        try {
            fn();
        } catch (const std::exception& e) {
            std::cerr << "  unexpected exception: " << e.what() << '\n';
            ++g_failures;
        }
        const bool ok = g_failures == before;
        failedTests += ok ? 0 : 1;
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << name << '\n';
    }

    std::cout << '\n' << (tests.size() - failedTests) << '/' << tests.size() << " tests passed\n";
    return failedTests == 0 ? 0 : 1;
}
