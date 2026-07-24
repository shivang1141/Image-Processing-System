#include "benchmark.hpp"
#include "compare.hpp"
#include "parse.hpp"
#include "processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace fs = std::filesystem;
using namespace imgproc;

namespace {

void printUsage(const char* prog) {
    std::cout <<
        "Usage:\n"
        "  " << prog << " process  <image>  [-o outdir] [canny options]\n"
        "      Grayscale + Canny on one image; saves <name>_gray.png / <name>_edges.png.\n\n"
        "  " << prog << " batch    <folder> [-o outdir] [--csv file] [--workers N] [canny options]\n"
        "      Process every image in a folder and report per-image and average times.\n"
        "      --workers N processes N images at once (0 = one per CPU core, default 1).\n\n"
        "  " << prog << " compare  [image]  [-o outdir] [--blurs K,...] [--lows T,...] [--highs T,...]\n"
        "                          [--repeats N] [--csv file]\n"
        "      Run every blur x low x high combination and save labelled contact sheets\n"
        "      (synthetic 1280x720 image unless an image is given).\n\n"
        "  " << prog << " bench    [--sizes WxH,...] [--iterations N] [--image file]\n"
        "                          [--csv file] [canny options]\n"
        "      Benchmark grayscale + Canny across image sizes (synthetic image unless --image).\n\n"
        "  " << prog << " scale    [--sizes small,medium,large] [--images N] [--workers 1,2,4,...]\n"
        "                          [--repeats N] [--image file] [--csv file] [canny options]\n"
        "      Compare sequential processing with N images processed concurrently.\n\n"
        "  " << prog << " generate <folder> [--count N] [--sizes WxH,...]\n"
        "      Write synthetic sample images for trying out 'batch'.\n\n"
        "Sizes are WxH or the presets small (640x480), medium (1920x1080), large (3840x2160).\n\n"
        "Canny options:\n"
        "  --low T       low threshold  (default 50)\n"
        "  --high T      high threshold (default 150)\n"
        "  --blur K      Gaussian blur kernel before Canny, 0 = off (default 5)\n"
        "  --aperture A  Sobel aperture 3, 5 or 7 (default 3)\n"
        "  --threads N   OpenCV worker threads, 1 = single-threaded (default: OpenCV decides)\n";
}

struct Options {
    std::vector<std::string> positional;
    std::string outDir = "output";
    std::string csvPath;
    std::string imagePath;
    std::vector<cv::Size> sizes = {{320, 240}, {640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    bool sizesGiven = false;
    int iterations = 50;
    int count = 10;
    int threads = -1;
    int workers = 1;
    std::string workerList; // batch: one count (default 1); scale: a list (default 1,2,4,8,all cores)
    int images = 16;
    int repeats = 3;
    bool repeatsGiven = false;
    std::vector<int> blurs = {0, 3, 5, 9};
    std::vector<double> lows = {25, 50, 100};
    std::vector<double> highs = {75, 150, 250};
    EdgeParams edge;
};

Options parseArgs(int argc, char** argv, int start) {
    Options opt;
    for (int i = start; i < argc; ++i) {
        const std::string arg = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) {
                throw std::invalid_argument("Missing value for " + arg);
            }
            return argv[++i];
        };

        if (arg == "-o" || arg == "--output") opt.outDir = next();
        else if (arg == "--csv") opt.csvPath = next();
        else if (arg == "--image") opt.imagePath = next();
        else if (arg == "--sizes") { opt.sizes = parseSizes(next()); opt.sizesGiven = true; }
        else if (arg == "--iterations") opt.iterations = parseInt(next(), "--iterations");
        else if (arg == "--count") opt.count = parseInt(next(), "--count");
        else if (arg == "--threads") opt.threads = parseInt(next(), "--threads");
        else if (arg == "--workers") opt.workerList = next();
        else if (arg == "--images") opt.images = parseInt(next(), "--images");
        else if (arg == "--repeats") { opt.repeats = parseInt(next(), "--repeats"); opt.repeatsGiven = true; }
        else if (arg == "--blurs") opt.blurs = parseIntList(next(), "--blurs");
        else if (arg == "--lows") opt.lows = parseDoubleList(next(), "--lows");
        else if (arg == "--highs") opt.highs = parseDoubleList(next(), "--highs");
        else if (arg == "--low") opt.edge.lowThreshold = parseDouble(next(), "--low");
        else if (arg == "--high") opt.edge.highThreshold = parseDouble(next(), "--high");
        else if (arg == "--blur") opt.edge.blurKernel = parseInt(next(), "--blur");
        else if (arg == "--aperture") opt.edge.apertureSize = parseInt(next(), "--aperture");
        else if (!arg.empty() && arg[0] == '-') throw std::invalid_argument("Unknown option: " + arg);
        else opt.positional.push_back(arg);
    }
    validateParams(opt.edge);
    if (opt.iterations < 1) throw std::invalid_argument("--iterations must be >= 1");
    if (opt.count < 1) throw std::invalid_argument("--count must be >= 1");
    if (opt.images < 1) throw std::invalid_argument("--images must be >= 1");
    if (opt.repeats < 1) throw std::invalid_argument("--repeats must be >= 1");
    return opt;
}

int runProcess(const Options& opt) {
    if (opt.positional.size() != 1) {
        throw std::invalid_argument("process: expected exactly one image path");
    }
    const ProcessResult r = processFile(opt.positional[0], opt.outDir, opt.edge);

    std::cout << std::fixed << std::setprecision(3)
              << "Image:      " << r.inputPath << " (" << r.width << "x" << r.height << ")\n"
              << "Load:       " << r.timings.loadMs << " ms\n"
              << "Grayscale:  " << r.timings.grayscaleMs << " ms\n"
              << "Canny:      " << r.timings.edgeMs << " ms\n"
              << "Save:       " << r.timings.saveMs << " ms\n"
              << "Processing: " << r.timings.processingMs() << " ms (grayscale + Canny)\n"
              << "Total:      " << r.timings.totalMs() << " ms\n"
              << "Output:     " << fs::absolute(opt.outDir).string() << '\n';
    return 0;
}

int runBatch(const Options& opt) {
    if (opt.positional.size() != 1) {
        throw std::invalid_argument("batch: expected exactly one folder path");
    }
    const int workers =
        opt.workerList.empty() ? 1 : resolveWorkers(parseInt(opt.workerList, "--workers"));
    // Several images at once, each also using OpenCV's thread pool, oversubscribes the
    // CPU. Keep OpenCV single-threaded per image unless --threads says otherwise.
    if (workers > 1 && opt.threads < 0) {
        cv::setNumThreads(1);
    }

    const BatchReport report = processFolder(opt.positional[0], opt.outDir, opt.edge, workers);
    if (report.results.empty() && report.failures.empty()) {
        std::cerr << "No supported images found in " << opt.positional[0] << '\n';
        return 1;
    }

    printBatchReport(std::cout, report);
    if (!opt.csvPath.empty()) {
        writeBatchCsv(opt.csvPath, report);
        std::cout << "CSV written to " << opt.csvPath << '\n';
    }
    return report.failures.empty() ? 0 : 2;
}

int runCompare(const Options& opt) {
    if (opt.positional.size() > 1) {
        throw std::invalid_argument("compare: expected at most one image path");
    }
    const cv::Mat image = opt.positional.empty() ? makeSyntheticImage(1280, 720)
                                                 : loadImage(opt.positional[0]);
    const std::string name = opt.positional.empty() ? "synthetic"
                                                    : fs::path(opt.positional[0]).stem().string();
    std::cout << "Image: " << (opt.positional.empty() ? "synthetic 1280x720" : opt.positional[0])
              << " (" << image.cols << "x" << image.rows << ")\n\n";

    const auto grid = buildParamGrid(opt.blurs, opt.lows, opt.highs, opt.edge.apertureSize);
    if (grid.empty()) {
        throw std::invalid_argument("compare: no valid combinations (every --lows value is above every --highs value)");
    }
    const auto trials = compareParams(toGrayscale(image), grid, opt.repeatsGiven ? opt.repeats : 5);
    printComparison(std::cout, trials);

    const fs::path out(opt.outDir);
    fs::create_directories(out);
    const fs::path stages = out / (name + "_stages.png");
    saveImage(stages.string(), makeStageStrip(image, opt.edge));
    std::cout << "\nWrote " << stages.string() << '\n';
    for (int blur : opt.blurs) {
        const fs::path sheet = out / (name + "_sweep_blur" + std::to_string(blur) + ".png");
        saveImage(sheet.string(), makeContactSheet(trials, blur, opt.lows, opt.highs));
        std::cout << "Wrote " << sheet.string() << '\n';
    }
    if (!opt.csvPath.empty()) {
        writeComparisonCsv(opt.csvPath, trials);
        std::cout << "CSV written to " << opt.csvPath << '\n';
    }
    return 0;
}

int runBench(const Options& opt) {
    cv::Mat source;
    if (!opt.imagePath.empty()) {
        source = loadImage(opt.imagePath);
        std::cout << "Source image: " << opt.imagePath << '\n';
    } else {
        std::cout << "Source image: synthetic\n";
    }
    std::cout << "OpenCV " << CV_VERSION << ", threads: " << cv::getNumThreads()
              << ", Canny low/high: " << opt.edge.lowThreshold << '/' << opt.edge.highThreshold
              << ", blur: " << opt.edge.blurKernel << "\n\n";

    const auto results = benchmarkSizes(opt.sizes, opt.iterations, source, opt.edge);
    printSizeBenchmark(std::cout, results);
    if (!opt.csvPath.empty()) {
        writeSizeBenchmarkCsv(opt.csvPath, results);
        std::cout << "CSV written to " << opt.csvPath << '\n';
    }
    return 0;
}

int runScale(const Options& opt) {
    cv::Mat source;
    if (!opt.imagePath.empty()) {
        source = loadImage(opt.imagePath);
    }
    const std::vector<cv::Size> sizes =
        opt.sizesGiven ? opt.sizes
                       : std::vector<cv::Size>{sizePreset("small"), sizePreset("medium"), sizePreset("large")};
    std::set<int> unique;
    for (int w : parseIntList(opt.workerList.empty() ? "1,2,4,8,0" : opt.workerList, "--workers")) {
        unique.insert(resolveWorkers(w));
    }
    const std::vector<int> workers(unique.begin(), unique.end());

    std::cout << "OpenCV " << CV_VERSION << ", CPU threads: " << std::thread::hardware_concurrency()
              << ", source: " << (source.empty() ? "synthetic" : opt.imagePath)
              << ", repeats: " << opt.repeats << '\n';

    const auto results = benchmarkScaling(sizes, opt.images, workers, opt.repeats, source, opt.edge);
    printScaling(std::cout, results);
    if (!opt.csvPath.empty()) {
        writeScalingCsv(opt.csvPath, results);
        std::cout << "\nCSV written to " << opt.csvPath << '\n';
    }
    return 0;
}

int runGenerate(const Options& opt) {
    if (opt.positional.size() != 1) {
        throw std::invalid_argument("generate: expected exactly one output folder");
    }
    const fs::path dir(opt.positional[0]);
    fs::create_directories(dir);

    for (int i = 0; i < opt.count; ++i) {
        const cv::Size size = opt.sizes[static_cast<size_t>(i) % opt.sizes.size()];
        std::ostringstream name;
        name << "sample_" << std::setw(3) << std::setfill('0') << i << '_'
             << size.width << 'x' << size.height << ".png";
        saveImage((dir / name.str()).string(),
                  makeSyntheticImage(size.width, size.height, static_cast<unsigned>(i + 1)));
    }
    std::cout << "Wrote " << opt.count << " images to " << fs::absolute(dir).string() << '\n';
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    if (argc < 2 || std::string(argv[1]) == "-h" || std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return argc < 2 ? 1 : 0;
    }

    const std::string command = argv[1];
    try {
        const Options opt = parseArgs(argc, argv, 2);
        if (opt.threads >= 0) {
            cv::setNumThreads(opt.threads);
        }

        if (command == "process") return runProcess(opt);
        if (command == "batch") return runBatch(opt);
        if (command == "compare") return runCompare(opt);
        if (command == "bench") return runBench(opt);
        if (command == "scale") return runScale(opt);
        if (command == "generate") return runGenerate(opt);

        std::cerr << "Unknown command: " << command << "\n\n";
        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
