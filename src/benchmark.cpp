#include "benchmark.hpp"
#include "timer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace imgproc {

namespace {

std::ofstream openCsv(const std::string& path) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV for writing: " + path);
    }
    out << std::fixed << std::setprecision(4);
    return out;
}

} // namespace

// ---------- Folder (batch) processing ----------

StageTimings BatchReport::average() const {
    StageTimings avg;
    if (results.empty()) {
        return avg;
    }
    for (const auto& r : results) {
        avg.loadMs += r.timings.loadMs;
        avg.grayscaleMs += r.timings.grayscaleMs;
        avg.edgeMs += r.timings.edgeMs;
        avg.saveMs += r.timings.saveMs;
    }
    const double n = static_cast<double>(results.size());
    avg.loadMs /= n;
    avg.grayscaleMs /= n;
    avg.edgeMs /= n;
    avg.saveMs /= n;
    return avg;
}

bool isSupportedImage(const std::string& path) {
    static const std::vector<std::string> kExtensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"};

    std::string ext = fs::path(path).extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
}

BatchReport processFolder(const std::string& inputDir,
                          const std::string& outputDir,
                          const EdgeParams& params) {
    if (!fs::is_directory(inputDir)) {
        throw std::runtime_error("Not a directory: " + inputDir);
    }

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && isSupportedImage(entry.path().string())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());

    BatchReport report;
    for (const auto& file : files) {
        try {
            report.results.push_back(processFile(file.string(), outputDir, params));
        } catch (const std::exception& e) {
            report.failures.push_back(file.string() + ": " + e.what());
        }
    }
    return report;
}

void printBatchReport(std::ostream& os, const BatchReport& report) {
    os << std::fixed << std::setprecision(3);
    os << std::left << std::setw(32) << "Image" << std::right
       << std::setw(12) << "Size"
       << std::setw(10) << "Load"
       << std::setw(10) << "Gray"
       << std::setw(10) << "Canny"
       << std::setw(10) << "Save"
       << std::setw(10) << "Total" << "   (ms)\n";
    os << std::string(94, '-') << '\n';

    for (const auto& r : report.results) {
        std::string name = fs::path(r.inputPath).filename().string();
        if (name.size() > 30) {
            name = name.substr(0, 27) + "...";
        }
        const std::string size = std::to_string(r.width) + "x" + std::to_string(r.height);
        os << std::left << std::setw(32) << name << std::right
           << std::setw(12) << size
           << std::setw(10) << r.timings.loadMs
           << std::setw(10) << r.timings.grayscaleMs
           << std::setw(10) << r.timings.edgeMs
           << std::setw(10) << r.timings.saveMs
           << std::setw(10) << r.timings.totalMs() << '\n';
    }
    os << std::string(94, '-') << '\n';

    const StageTimings avg = report.average();
    os << std::left << std::setw(44) << ("AVERAGE (" + std::to_string(report.results.size()) + " images)")
       << std::right
       << std::setw(10) << avg.loadMs
       << std::setw(10) << avg.grayscaleMs
       << std::setw(10) << avg.edgeMs
       << std::setw(10) << avg.saveMs
       << std::setw(10) << avg.totalMs() << '\n';
    os << "Average processing time (grayscale + Canny): " << avg.processingMs() << " ms\n";

    if (!report.failures.empty()) {
        os << "\nFailed (" << report.failures.size() << "):\n";
        for (const auto& f : report.failures) {
            os << "  " << f << '\n';
        }
    }
}

void writeBatchCsv(const std::string& path, const BatchReport& report) {
    std::ofstream out = openCsv(path);
    out << "image,width,height,load_ms,grayscale_ms,canny_ms,save_ms,total_ms\n";
    for (const auto& r : report.results) {
        out << fs::path(r.inputPath).filename().string() << ','
            << r.width << ',' << r.height << ','
            << r.timings.loadMs << ',' << r.timings.grayscaleMs << ','
            << r.timings.edgeMs << ',' << r.timings.saveMs << ','
            << r.timings.totalMs() << '\n';
    }
}

} // namespace imgproc
