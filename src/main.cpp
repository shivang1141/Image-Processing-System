#include "benchmark.hpp"
#include "processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;
using namespace imgproc;

namespace {

void printUsage(const char* prog) {
    std::cout <<
        "Usage:\n"
        "  " << prog << " process  <image>  [-o outdir] [canny options]\n"
        "      Grayscale + Canny on one image; saves <name>_gray.png / <name>_edges.png.\n\n"
        "  " << prog << " batch    <folder> [-o outdir] [--csv file] [canny options]\n"
        "      Process every image in a folder and report per-image and average times.\n\n"
        "  " << prog << " bench    [--sizes WxH,WxH,...] [--iterations N] [--image file]\n"
        "                          [--csv file] [canny options]\n"
        "      Benchmark grayscale + Canny across image sizes (synthetic image unless --image).\n\n"
        "  " << prog << " generate <folder> [--count N] [--sizes WxH,...]\n"
        "      Write synthetic sample images for trying out 'batch'.\n\n"
        "Canny options:\n"
        "  --low T     low threshold  (default 50)\n"
        "  --high T    high threshold (default 150)\n"
        "  --blur K    Gaussian blur kernel before Canny, 0 = off (default 5)\n"
        "  --threads N OpenCV worker threads, 1 = single-threaded (default: OpenCV decides)\n";
}

std::vector<cv::Size> parseSizes(const std::string& text) {
    std::vector<cv::Size> sizes;
    std::stringstream ss(text);
    std::string item;
    while (std::getline(ss, item, ',')) {
        const auto x = item.find_first_of("xX");
        if (x == std::string::npos) {
            throw std::invalid_argument("Bad size '" + item + "', expected WxH");
        }
        const int w = std::stoi(item.substr(0, x));
        const int h = std::stoi(item.substr(x + 1));
        if (w <= 0 || h <= 0) {
            throw std::invalid_argument("Size must be positive: " + item);
        }
        sizes.emplace_back(w, h);
    }
    return sizes;
}

struct Options {
    std::vector<std::string> positional;
    std::string outDir = "output";
    std::string csvPath;
    std::string imagePath;
    std::vector<cv::Size> sizes = {{320, 240}, {640, 480}, {1280, 720}, {1920, 1080}, {3840, 2160}};
    int iterations = 50;
    int count = 10;
    int threads = -1;
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
        else if (arg == "--sizes") opt.sizes = parseSizes(next());
        else if (arg == "--iterations") opt.iterations = std::stoi(next());
        else if (arg == "--count") opt.count = std::stoi(next());
        else if (arg == "--threads") opt.threads = std::stoi(next());
        else if (arg == "--low") opt.edge.lowThreshold = std::stod(next());
        else if (arg == "--high") opt.edge.highThreshold = std::stod(next());
        else if (arg == "--blur") opt.edge.blurKernel = std::stoi(next());
        else if (!arg.empty() && arg[0] == '-') throw std::invalid_argument("Unknown option: " + arg);
        else opt.positional.push_back(arg);
    }
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
    const BatchReport report = processFolder(opt.positional[0], opt.outDir, opt.edge);
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
        if (command == "bench") return runBench(opt);
        if (command == "generate") return runGenerate(opt);

        std::cerr << "Unknown command: " << command << "\n\n";
        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
