// Command-line value parsing: sizes, presets, numbers and lists.
#include "parse.hpp"

#include <gtest/gtest.h>

using namespace imgproc;

TEST(ParseSizes, ParsesWxHList) {
    const auto sizes = parseSizes("640x480,1920X1080");
    ASSERT_EQ(sizes.size(), 2u);
    EXPECT_EQ(sizes[0], cv::Size(640, 480));
    EXPECT_EQ(sizes[1], cv::Size(1920, 1080));
}

TEST(ParseSizes, AcceptsPresetsMixedWithSizes) {
    const auto sizes = parseSizes("small,Medium,LARGE,100x50");
    ASSERT_EQ(sizes.size(), 4u);
    EXPECT_EQ(sizes[0], cv::Size(640, 480));
    EXPECT_EQ(sizes[1], cv::Size(1920, 1080));
    EXPECT_EQ(sizes[2], cv::Size(3840, 2160));
    EXPECT_EQ(sizes[3], cv::Size(100, 50));
}

TEST(ParseSizes, RejectsMalformedInput) {
    for (const char* bad : {"", "640x", "x480", "640x480x2", "axb", "0x480", "-5x10", "huge"}) {
        EXPECT_THROW(parseSizes(bad), std::invalid_argument) << "'" << bad << "'";
    }
}

TEST(ParseNumbers, StrictWholeStringParsing) {
    EXPECT_EQ(parseInt("42", "n"), 42);
    EXPECT_EQ(parseInt("-3", "n"), -3);
    EXPECT_DOUBLE_EQ(parseDouble("37.5", "t"), 37.5);
    for (const char* bad : {"", "ten", "4x", "3.5", "99999999999"}) {
        EXPECT_THROW(parseInt(bad, "n"), std::invalid_argument) << "'" << bad << "'";
    }
    for (const char* bad : {"", "abc", "1.5.2", "2e"}) {
        EXPECT_THROW(parseDouble(bad, "t"), std::invalid_argument) << "'" << bad << "'";
    }
}

TEST(ParseNumbers, ErrorMessageNamesTheOption) {
    try {
        parseInt("ten", "--iterations");
        FAIL() << "expected std::invalid_argument";
    } catch (const std::invalid_argument& e) {
        EXPECT_STREQ(e.what(), "Invalid --iterations: 'ten' (expected an integer)");
    }
}

TEST(ParseLists, ParsesCommaSeparatedValues) {
    EXPECT_EQ(parseIntList("1,2,4,8", "w"), (std::vector<int>{1, 2, 4, 8}));
    EXPECT_EQ(parseDoubleList("25,50.5", "t"), (std::vector<double>{25, 50.5}));
    EXPECT_THROW(parseIntList("1,,2", "w"), std::invalid_argument);
    EXPECT_THROW(parseIntList("1,two", "w"), std::invalid_argument);
}
