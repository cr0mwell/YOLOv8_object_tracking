#ifndef STRUCTURES_H
#define STRUCTURES_H

#include <limits>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>

using namespace std;


enum class OnnxProviders { CPU, CUDA };

// Object detection model
const OnnxProviders provider = OnnxProviders::CPU;
const string onnxLogId = "yolov8_inference";
const string modelBasePath = "../models/yolo/";
const string classesFile = modelBasePath + "coco.names";
const float confThreshold = 0.35;
const float nmsThreshold = 0.5;
const float maskThreshold = 0.5;

// Media data variables
const string inputVideoPath = "../media/input.mp4";
const string outputVideoPath = "../media/output.mp4";

// Session file
const string sessionFile = "../json/session_dump.json";

// Tracking data
const int INT_INF = numeric_limits<int>::max();
const int ORPHANED = 0;
const int REMOVED = -1;
const int CONFIRMED = 3;

const int KEY_POINT_NUM = 17;

namespace Tasks {
    inline const string DETECT = "detect";
    inline const string SEGMENT = "segment";
    inline const string POSE = "pose";
}

struct PoseKeyPoint {
    float x = 0;
    float y = 0;
    float confidence = 0;
};

struct Results {
    int classId {};
    int trackId {};
    float confidence {};
    cv::Rect bbox;
    // Segmentation mask (optional)
    cv::Mat mask;
    // Pose estimation keypoints (optional)
    vector<PoseKeyPoint> keyPoints {};
};

struct Track {
    Track(int tId, cv::Rect tBbox) : id(tId), bbox(tBbox) {};
    int id {};
    cv::Rect bbox;
    int score {1};
};

struct PoseParams {
    float kptThreshold = 0.5;
    int kptRadius = 5;
    bool isDrawKptLine = true;
    vector<vector<int>>skeleton = {{16, 14}, {14, 12}, {17, 15}, {15, 13},
                                   {12, 13},{6, 12},{7, 13},{6, 7},{6, 8},{7, 9},
                                   {8, 10},{9, 11},{2, 3},{1, 2},{1, 3},{2, 4},
                                   {3, 5},{4, 6},{5, 7}};
    vector<cv::Scalar> posePalette = {cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102),
                                      cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255), cv::Scalar(153, 204, 255),
                                      cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255),
                                      cv::Scalar(51, 153, 255), cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102),
                                      cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
                                      cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255),
                                      cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)};
    vector<int> limbColor = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
    vector<int> kptColor = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };
};

struct MaskParams {
    int networkWidth {};
    int networkHeight {};
    float maskThreshold {};
    cv::Size srcImgShape;
    vector<float> params;
};

#endif