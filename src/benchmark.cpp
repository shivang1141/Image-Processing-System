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

Stats computeStats(const std::vector<double>& samples) {
    Stats s;
    if (samples.empty()) {
        return s;
    }
    double sum = 0.0;
    for (double v : samples) {
        sum += v;
    }
    s.meanMs = sum / static_cast<double>(samples.size());
    s.minMs = *std::min_element(samples.begin(), samples.end());
    s.maxMs = *std::max_element(samples.begin(), samples.end());
    return s;
}

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

// ---------- Image-size benchmark ----------

cv::Mat makeSyntheticImage(int width, int height, unsigned seed) {
    cv::Mat image(height, width, CV_8UC3);

    // Horizontal/vertical colour gradient as a background.
    for (int y = 0; y < height; ++y) {
        auto* row = image.ptr<cv::Vec3b>(y);
        for (int x = 0; x < width; ++x) {
            row[x] = cv::Vec3b(static_cast<uchar>(255 * x / std::max(1, width - 1)),
                               static_cast<uchar>(255 * y / std::max(1, height - 1)),
                               128);
        }
    }

    // Some solid shapes so Canny has real edges to find.
    cv::RNG rng(seed);
    const int shapes = 20;
    for (int i = 0; i < shapes; ++i) {
        cv::Point center(rng.uniform(0, width), rng.uniform(0, height));
        cv::Scalar color(rng.uniform(0, 256), rng.uniform(0, 256), rng.uniform(0, 256));
        int radius = std::max(2, rng.uniform(std::min(width, height) / 40 + 1,
                                             std::min(width, height) / 8 + 2));
        if (i % 2 == 0) {
            cv::circle(image, center, radius, color, cv::FILLED);
        } else {
            cv::rectangle(image, center, center + cv::Point(radius * 2, radius), color, cv::FILLED);
        }
    }

    // Light noise to mimic a real photo.
    cv::Mat noise(image.size(), CV_16SC3);
    rng.fill(noise, cv::RNG::NORMAL, cv::Scalar::all(0), cv::Scalar::all(8));
    cv::add(image, noise, image, cv::noArray(), CV_8UC3);
    return image;
}

std::vector<SizeBenchmarkResult> benchmarkSizes(const std::vector<cv::Size>& sizes,
                                                int iterations,
                                                const cv::Mat& source,
                                                const EdgeParams& params) {
    if (iterations < 1) {
        throw std::invalid_argument("benchmarkSizes: iterations must be >= 1");
    }

    std::vector<SizeBenchmarkResult> results;
    for (const auto& size : sizes) {
        cv::Mat image;
        if (source.empty()) {
            image = makeSyntheticImage(size.width, size.height);
        } else {
            cv::resize(source, image, size, 0, 0, cv::INTER_AREA);
        }

        // Warm-up run (first call pays for allocations / lazy init).
        detectEdges(toGrayscale(image), params);

        std::vector<double> grayTimes;
        std::vector<double> edgeTimes;
        grayTimes.reserve(iterations);
        edgeTimes.reserve(iterations);

        for (int i = 0; i < iterations; ++i) {
            Timer timer;
            cv::Mat gray = toGrayscale(image);
            grayTimes.push_back(timer.elapsedMs());

            timer.reset();
            cv::Mat edges = detectEdges(gray, params);
            edgeTimes.push_back(timer.elapsedMs());
        }

        SizeBenchmarkResult r;
        r.width = size.width;
        r.height = size.height;
        r.iterations = iterations;
        r.grayscale = computeStats(grayTimes);
        r.edges = computeStats(edgeTimes);
        results.push_back(r);
    }
    return results;
}

void printSizeBenchmark(std::ostream& os, const std::vector<SizeBenchmarkResult>& results) {
    os << std::fixed << std::setprecision(3);
    os << std::left << std::setw(14) << "Size" << std::right
       << std::setw(12) << "Megapixels"
       << std::setw(8) << "Iters"
       << std::setw(12) << "Gray mean"
       << std::setw(12) << "Gray min"
       << std::setw(12) << "Canny mean"
       << std::setw(12) << "Canny min"
       << std::setw(12) << "Total mean" << "   (ms)\n";
    os << std::string(99, '-') << '\n';

    for (const auto& r : results) {
        const std::string size = std::to_string(r.width) + "x" + std::to_string(r.height);
        const double mp = static_cast<double>(r.width) * r.height / 1e6;
        os << std::left << std::setw(14) << size << std::right
           << std::setw(12) << mp
           << std::setw(8) << r.iterations
           << std::setw(12) << r.grayscale.meanMs
           << std::setw(12) << r.grayscale.minMs
           << std::setw(12) << r.edges.meanMs
           << std::setw(12) << r.edges.minMs
           << std::setw(12) << (r.grayscale.meanMs + r.edges.meanMs) << '\n';
    }
}

void writeSizeBenchmarkCsv(const std::string& path, const std::vector<SizeBenchmarkResult>& results) {
    std::ofstream out = openCsv(path);
    out << "width,height,megapixels,iterations,"
           "gray_mean_ms,gray_min_ms,gray_max_ms,"
           "canny_mean_ms,canny_min_ms,canny_max_ms,total_mean_ms\n";
    for (const auto& r : results) {
        out << r.width << ',' << r.height << ','
            << static_cast<double>(r.width) * r.height / 1e6 << ','
            << r.iterations << ','
            << r.grayscale.meanMs << ',' << r.grayscale.minMs << ',' << r.grayscale.maxMs << ','
            << r.edges.meanMs << ',' << r.edges.minMs << ',' << r.edges.maxMs << ','
            << (r.grayscale.meanMs + r.edges.meanMs) << '\n';
    }
}

} // namespace imgproc
