#pragma once

#include <chrono>

namespace imgproc {

// Small steady-clock stopwatch reporting elapsed time in milliseconds.
class Timer {
public:
    Timer() : start_(Clock::now()) {}

    void reset() { start_ = Clock::now(); }

    double elapsedMs() const {
        return std::chrono::duration<double, std::milli>(Clock::now() - start_).count();
    }

private:
    using Clock = std::chrono::steady_clock;
    Clock::time_point start_;
};

} // namespace imgproc
