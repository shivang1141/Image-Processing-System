#include "compare.hpp"
#include "timer.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>
#include <fstream>
#include <iomanip>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace imgproc {

namespace {

const cv::Scalar kBackground(245, 245, 245);
const cv::Scalar kText(40, 40, 40);
const cv::Scalar kMuted(110, 110, 110);
const cv::Scalar kEmptyCell(200, 200, 200);
constexpr int kFont = cv::FONT_HERSHEY_SIMPLEX;

// "50" for whole numbers, "37.5" otherwise.
std::string formatNumber(double v) {
    std::ostringstream ss;
    if (std::floor(v) == v) {
        ss << static_cast<long long>(v);
    } else {
        ss << v;
    }
    return ss.str();
}

std::string blurLabel(int kernel) {
    if (kernel <= 0) {
        return "no blur";
    }
    const int k = kernel | 1;
    return "blur " + std::to_string(k) + "x" + std::to_string(k);
}

// Resizes to `width` keeping the aspect ratio, as a 3-channel image.
cv::Mat toThumb(const cv::Mat& image, int width) {
    cv::Mat bgr;
    if (image.channels() == 1) {
        cv::cvtColor(image, bgr, cv::COLOR_GRAY2BGR);
    } else {
        bgr = image;
    }
    const int height = std::max(1, static_cast<int>(std::lround(
                                       static_cast<double>(image.rows) * width / image.cols)));
    cv::Mat thumb;
    cv::resize(bgr, thumb, cv::Size(width, height), 0, 0, cv::INTER_AREA);
    return thumb;
}

void drawText(cv::Mat& canvas, const std::string& text, cv::Point org, double scale,
              const cv::Scalar& color, int thickness = 1) {
    cv::putText(canvas, text, org, kFont, scale, color, thickness, cv::LINE_AA);
}

const ParamTrial* findTrial(const std::vector<ParamTrial>& trials, int blur, double low, double high) {
    for (const auto& t : trials) {
        if (t.params.blurKernel == blur && t.params.lowThreshold == low &&
            t.params.highThreshold == high) {
            return &t;
        }
    }
    return nullptr;
}

} // namespace

EdgeMetrics measureEdges(const cv::Mat& edges) {
    if (edges.empty() || edges.channels() != 1 || edges.depth() != CV_8U) {
        throw std::invalid_argument("measureEdges: expected a non-empty 8-bit single-channel edge map");
    }
    EdgeMetrics m;
    m.edgePixels = cv::countNonZero(edges);
    m.densityPct = 100.0 * m.edgePixels / static_cast<double>(edges.total());

    cv::Mat labels;
    const int components = cv::connectedComponents(edges != 0, labels, 8, CV_32S);
    m.segments = components - 1; // label 0 is the background
    return m;
}

std::vector<EdgeParams> buildParamGrid(const std::vector<int>& blurs,
                                       const std::vector<double>& lows,
                                       const std::vector<double>& highs,
                                       int apertureSize) {
    std::vector<EdgeParams> grid;
    for (int blur : blurs) {
        for (double low : lows) {
            for (double high : highs) {
                if (low > high) {
                    continue;
                }
                EdgeParams p;
                p.blurKernel = blur;
                p.lowThreshold = low;
                p.highThreshold = high;
                p.apertureSize = apertureSize;
                validateParams(p);
                grid.push_back(p);
            }
        }
    }
    return grid;
}

std::vector<ParamTrial> compareParams(const cv::Mat& gray,
                                      const std::vector<EdgeParams>& grid,
                                      int repeats) {
    if (repeats < 1) {
        throw std::invalid_argument("compareParams: repeats must be >= 1");
    }
    if (grid.empty()) {
        return {};
    }

    // Warm-up so the first combination does not pay for OpenCV's lazy initialisation.
    detectEdges(gray, grid.front());

    std::vector<ParamTrial> trials;
    trials.reserve(grid.size());
    for (const auto& params : grid) {
        ParamTrial t;
        t.params = params;
        double totalMs = 0.0;
        for (int r = 0; r < repeats; ++r) {
            Timer timer;
            t.edges = detectEdges(gray, params);
            totalMs += timer.elapsedMs();
        }
        t.meanMs = totalMs / repeats;
        t.metrics = measureEdges(t.edges);
        trials.push_back(std::move(t));
    }
    return trials;
}

cv::Mat makeContactSheet(const std::vector<ParamTrial>& trials,
                         int blurKernel,
                         const std::vector<double>& lows,
                         const std::vector<double>& highs,
                         int thumbWidth) {
    const ParamTrial* first = nullptr;
    for (const auto& t : trials) {
        if (t.params.blurKernel == blurKernel) {
            first = &t;
            break;
        }
    }
    if (first == nullptr || lows.empty() || highs.empty()) {
        throw std::invalid_argument("makeContactSheet: no trials for blur " + std::to_string(blurKernel));
    }

    const cv::Size thumbSize = toThumb(first->edges, thumbWidth).size();
    const int gap = 8, caption = 26, rowHeader = 96, colHeader = 34, title = 46;
    const int cellW = thumbSize.width + gap;
    const int cellH = thumbSize.height + caption + gap;
    const int cols = static_cast<int>(highs.size());
    const int rows = static_cast<int>(lows.size());

    cv::Mat sheet(title + colHeader + rows * cellH + gap, rowHeader + cols * cellW + gap, CV_8UC3,
                  kBackground);

    drawText(sheet, "Canny parameter sweep  -  " + blurLabel(blurKernel) +
                        "  (rows: low threshold, columns: high threshold)",
             {gap, 30}, 0.65, kText, 2);

    for (int c = 0; c < cols; ++c) {
        drawText(sheet, "high " + formatNumber(highs[c]),
                 {rowHeader + c * cellW + 4, title + 22}, 0.6, kText, 2);
    }

    for (int r = 0; r < rows; ++r) {
        const int y = title + colHeader + r * cellH;
        drawText(sheet, "low " + formatNumber(lows[r]), {gap, y + thumbSize.height / 2}, 0.6, kText, 2);

        for (int c = 0; c < cols; ++c) {
            const int x = rowHeader + c * cellW;
            const cv::Rect cell(x, y, thumbSize.width, thumbSize.height);
            const ParamTrial* t = findTrial(trials, blurKernel, lows[r], highs[c]);
            if (t == nullptr) {
                sheet(cell).setTo(kEmptyCell);
                drawText(sheet, "n/a (low > high)", {x + 10, y + thumbSize.height / 2}, 0.5, kMuted);
                continue;
            }
            cv::Mat thumb = toThumb(t->edges, thumbSize.width);
            cv::resize(thumb, thumb, thumbSize); // guard against 1px rounding differences
            thumb.copyTo(sheet(cell));

            std::ostringstream label;
            label << std::fixed << std::setprecision(2) << t->metrics.densityPct << "% edges   "
                  << t->metrics.segments << " segments";
            drawText(sheet, label.str(), {x + 2, y + thumbSize.height + 18}, 0.45, kMuted);
        }
    }
    return sheet;
}

cv::Mat makeStageStrip(const cv::Mat& bgr, const EdgeParams& params, int thumbWidth) {
    const cv::Mat gray = toGrayscale(bgr);
    const cv::Mat blurred = blurImage(gray, params.blurKernel);
    const cv::Mat edges = detectEdges(gray, params);

    const std::vector<std::pair<std::string, cv::Mat>> panels = {
        {"1. Original", bgr},
        {"2. Grayscale", gray},
        {"3. Gaussian " + blurLabel(params.blurKernel), blurred},
        {"4. Canny " + formatNumber(params.lowThreshold) + " / " + formatNumber(params.highThreshold), edges},
    };

    const cv::Size thumbSize = toThumb(bgr, thumbWidth).size();
    const int gap = 8, label = 34;
    cv::Mat strip(label + thumbSize.height + gap,
                  gap + static_cast<int>(panels.size()) * (thumbSize.width + gap), CV_8UC3, kBackground);

    for (size_t i = 0; i < panels.size(); ++i) {
        const int x = gap + static_cast<int>(i) * (thumbSize.width + gap);
        drawText(strip, panels[i].first, {x, 24}, 0.6, kText, 2);
        cv::Mat thumb = toThumb(panels[i].second, thumbSize.width);
        cv::resize(thumb, thumb, thumbSize);
        thumb.copyTo(strip(cv::Rect(x, label, thumbSize.width, thumbSize.height)));
    }
    return strip;
}

void printComparison(std::ostream& os, const std::vector<ParamTrial>& trials) {
    os << std::fixed << std::setprecision(2);
    os << std::left << std::setw(12) << "Blur" << std::right
       << std::setw(8) << "Low"
       << std::setw(8) << "High"
       << std::setw(12) << "Edge px"
       << std::setw(11) << "Density %"
       << std::setw(11) << "Segments"
       << std::setw(11) << "Time ms" << '\n';
    os << std::string(73, '-') << '\n';
    for (const auto& t : trials) {
        os << std::left << std::setw(12) << blurLabel(t.params.blurKernel) << std::right
           << std::setw(8) << formatNumber(t.params.lowThreshold)
           << std::setw(8) << formatNumber(t.params.highThreshold)
           << std::setw(12) << t.metrics.edgePixels
           << std::setw(11) << t.metrics.densityPct
           << std::setw(11) << t.metrics.segments
           << std::setw(11) << t.meanMs << '\n';
    }
}

void writeComparisonCsv(const std::string& path, const std::vector<ParamTrial>& trials) {
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("Failed to open CSV for writing: " + path);
    }
    out << std::fixed << std::setprecision(4);
    out << "blur,low,high,aperture,edge_pixels,density_pct,segments,mean_ms\n";
    for (const auto& t : trials) {
        out << t.params.blurKernel << ',' << t.params.lowThreshold << ',' << t.params.highThreshold << ','
            << t.params.apertureSize << ',' << t.metrics.edgePixels << ',' << t.metrics.densityPct << ','
            << t.metrics.segments << ',' << t.meanMs << '\n';
    }
}

} // namespace imgproc
