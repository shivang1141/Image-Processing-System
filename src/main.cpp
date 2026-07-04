#include "processor.hpp"

#include <opencv2/core.hpp>
#include <opencv2/core/utility.hpp>

#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
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
        "Canny options:\n"
        "  --low T     low threshold  (default 50)\n"
        "  --high T    high threshold (default 150)\n"
        "  --blur K    Gaussian blur kernel before Canny, 0 = off (default 5)\n"
        "  --threads N OpenCV worker threads, 1 = single-threaded (default: OpenCV decides)\n";
}

struct Options {
    std::vector<std::string> positional;
    std::string outDir = "output";
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

        std::cerr << "Unknown command: " << command << "\n\n";
        printUsage(argv[0]);
        return 1;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
}
