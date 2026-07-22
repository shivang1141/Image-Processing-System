cmake_minimum_required(VERSION 3.16)
project(ImageProcessingSystem VERSION 1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)
find_package(Threads REQUIRED)

# Core processing library shared by the app and the tests.
add_library(imgproc_core
    src/processor.cpp
    src/benchmark.cpp
    src/compare.cpp
    src/parse.cpp
)
target_include_directories(imgproc_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(imgproc_core PUBLIC ${OpenCV_LIBS} Threads::Threads)
target_compile_options(imgproc_core PRIVATE -Wall -Wextra -Wpedantic)

add_executable(imgproc src/main.cpp)
target_link_libraries(imgproc PRIVATE imgproc_core)

enable_testing()
add_executable(imgproc_tests tests/test_processor.cpp)
target_link_libraries(imgproc_tests PRIVATE imgproc_core)
add_test(NAME imgproc_tests COMMAND imgproc_tests)
