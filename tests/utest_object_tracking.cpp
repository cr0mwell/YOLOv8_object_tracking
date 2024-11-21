#include <fstream>

#include "gtest/gtest.h"

using namespace std;


vector<string> getClasses(const char* classesFile) {
    vector<string> classes;
    ifstream ifs(classesFile);
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    return classes;
}


class ObjectDetectionTest : public ::testing::Test {
protected:
    const char* sessionFile = "../json/session_dump.json";
    const char* outputVideoPath = "../media/output.avi";
    const char* classesFile = "../models/yolo/coco.names";
};


// Test the output video file exists
TEST_F(ObjectDetectionTest, TestOutputVideoExists) {
    FILE* f = fopen(outputVideoPath, "r");
    ASSERT_TRUE(f != NULL);
}

// Test the output session file exists
TEST_F(ObjectDetectionTest, TestOutputSessionExists) {
    FILE* f = fopen(sessionFile, "r");
    ASSERT_TRUE(f != NULL);
}

// Test the classes file contains valid data
TEST_F(ObjectDetectionTest, TestOutputClasses) {
    vector<string> classes = getClasses(classesFile);
    EXPECT_EQ(classes.size(), 80);
}