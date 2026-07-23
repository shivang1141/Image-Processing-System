#include "benchmark.hpp"
#include "timer.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <atomic>
#include <cctype>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <map>
#include <optional>
#include <ostream>
#include <set>
#include <stdexcept>
#include <thread>

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

std::string lowerExtension(const fs::path& path) {
    std::string ext = path.extension().string();
    std::transform(ext.begin(), ext.end(), ext.begin(),
                   [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
    return ext;
}

// Calls fn(0) .. fn(count - 1), spread over `workers` threads that each pull the
// next index from a shared counter. fn must not throw.
template <typename Fn>
void runParallel(size_t count, int workers, Fn&& fn) {
    if (workers <= 1 || count <= 1) {
        for (size_t i = 0; i < count; ++i) {
            fn(i);
        }
        return;
    }

    std::atomic<size_t> next{0};
    auto worker = [&] {
        for (size_t i = next.fetch_add(1); i < count; i = next.fetch_add(1)) {
            fn(i);
        }
    };

    std::vector<std::thread> pool;
    const size_t threads = std::min(count, static_cast<size_t>(workers));
    pool.reserve(threads);
    for (size_t t = 0; t < threads; ++t) {
        pool.emplace_back(worker);
    }
    for (auto& t : pool) {
        t.join();
    }
}

// Output stem for each file; files sharing a stem get their extension appended.
std::vector<std::string> outputStems(const std::vector<fs::path>& files) {
    std::map<std::string, int> stemCount;
    for (const auto& f : files) {
        ++stemCount[f.stem().string()];
    }

    std::set<std::string> used;
    std::vector<std::string> stems;
    stems.reserve(files.size());
    for (const auto& f : files) {
        std::string base = f.stem().string();
        if (stemCount[base] > 1) {
            std::string ext = lowerExtension(f);
            if (!ext.empty()) {
                base += "_" + ext.substr(1);
            }
        }
        std::string name = base;
        for (int k = 2; used.count(name) != 0; ++k) {
            name = base + "_" + std::to_string(k);
        }
        used.insert(name);
        stems.push_back(name);
    }
    return stems;
}

std::string sizeLabel(int width, int height) {
    return std::to_string(width) + "x" + std::to_string(height);
}

} // namespace

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

double BatchReport::imagesPerSecond() const {
    return wallMs > 0.0 ? 1000.0 * static_cast<double>(results.size()) / wallMs : 0.0;
}

bool isSupportedImage(const std::string& path) {
    static const std::vector<std::string> kExtensions = {
        ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp", ".ppm", ".pgm"};

    const std::string ext = lowerExtension(path);
    return std::find(kExtensions.begin(), kExtensions.end(), ext) != kExtensions.end();
}

int resolveWorkers(int requested) {
    if (requested < 0) {
        throw std::invalid_argument("Worker count must be >= 0 (0 = one per CPU core)");
    }
    if (requested == 0) {
        return static_cast<int>(std::max(1u, std::thread::hardware_concurrency()));
    }
    return requested;
}

BatchReport processFolder(const std::string& inputDir,
                          const std::string& outputDir,
                          const EdgeParams& params,
                          int workers) {
    if (!fs::is_directory(inputDir)) {
        throw std::runtime_error("Not a directory: " + inputDir);
    }
    validateParams(params);

    std::vector<fs::path> files;
    for (const auto& entry : fs::directory_iterator(inputDir)) {
        if (entry.is_regular_file() && isSupportedImage(entry.path().string())) {
            files.push_back(entry.path());
        }
    }
    std::sort(files.begin(), files.end());
    const std::vector<std::string> stems = outputStems(files);

    BatchReport report;
    report.workers = resolveWorkers(workers);
    if (files.empty()) {
        return report;
    }
    fs::create_directories(outputDir);

    // Warm-up so the first image does not pay for OpenCV's lazy initialisation.
    detectEdges(toGrayscale(makeSyntheticImage(64, 48)), params);

    std::vector<std::optional<ProcessResult>> results(files.size());
    std::vector<std::string> errors(files.size());

    Timer wall;
    runParallel(files.size(), report.workers, [&](size_t i) {
        try {
            results[i] = processFile(files[i].string(), outputDir, params, stems[i]);
        } catch (const std::exception& e) {
            errors[i] = files[i].string() + ": " + e.what();
        }
    });
    report.wallMs = wall.elapsedMs();

    for (size_t i = 0; i < files.size(); ++i) {
        if (results[i]) {
            report.results.push_back(std::move(*results[i]));
        } else {
            report.failures.push_back(errors[i]);
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
        os << std::left << std::setw(32) << name << std::right
           << std::setw(12) << sizeLabel(r.width, r.height)
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
    os << "Wall time: " << report.wallMs << " ms with " << report.workers << " worker(s), "
       << std::setprecision(1) << report.imagesPerSecond() << " images/s\n"
       << std::setprecision(3);

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
        const double mp = static_cast<double>(r.width) * r.height / 1e6;
        os << std::left << std::setw(14) << sizeLabel(r.width, r.height) << std::right
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
