cmake_minimum_required(VERSION 3.16)
project(ImageProcessingSystem VERSION 2.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

option(IMGPROC_BUILD_TESTS "Build the GoogleTest suite" ON)

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

if(IMGPROC_BUILD_TESTS)
    enable_testing()

    # Prefer an installed GoogleTest (apt: libgtest-dev); otherwise download it.
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.tar.gz)
        set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
        set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(imgproc_tests
        tests/test_processor.cpp
        tests/test_batch.cpp
        tests/test_compare.cpp
        tests/test_parse.cpp
    )
    target_link_libraries(imgproc_tests PRIVATE imgproc_core GTest::gtest_main)

    include(GoogleTest)
    gtest_discover_tests(imgproc_tests DISCOVERY_TIMEOUT 60)

endif()
