#include "processor.hpp"

#include <filesystem>
#include <iostream>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <image> [outdir]\n";
        return 1;
    }
    const std::string outDir = argc > 2 ? argv[2] : "output";

    try {
        const cv::Mat image = imgproc::loadImage(argv[1]);
        const cv::Mat gray = imgproc::toGrayscale(image);
        const cv::Mat edges = imgproc::detectEdges(gray);

        fs::create_directories(outDir);
        const std::string stem = fs::path(argv[1]).stem().string();
        imgproc::saveImage((fs::path(outDir) / (stem + "_gray.png")).string(), gray);
        imgproc::saveImage((fs::path(outDir) / (stem + "_edges.png")).string(), edges);
        std::cout << "Saved " << stem << "_gray.png and " << stem << "_edges.png ("
                  << image.cols << "x" << image.rows << ")\n";
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << '\n';
        return 1;
    }
    return 0;
}
