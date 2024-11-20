#ifndef INFERENCE_DRIVER_ONNX_H
#define INFERENCE_DRIVER_ONNX_H

#include <unordered_map>

#include "helpers.hpp"

using namespace std;


class OnnxDriver {
public:
    OnnxDriver(string modelPath, const string logId, const OnnxProviders provider, const int classes);

    // Getters
    const vector<string>& getClasses();
    const vector<int64_t>& getInputTensorShape();
    const int& getWidth();
    const int& getHeight();
    const string& getTask();
    
    // Inference
    vector<Results> predict(cv::Mat& image, const float& confThreshold, const float& nmsThreshold, const float& maskThreshold, bool verbose=true);
    
protected:
    const string modelPath_;
    Ort::Env env {nullptr};
    Ort::Session session {nullptr};

    vector<const char*> inputNames;
    vector<const char*> outputNames;
    std::vector<Ort::AllocatedStringPtr> tmpInVector;
    std::vector<Ort::AllocatedStringPtr> tmpOutVector;
    Ort::ModelMetadata modelMetadata {nullptr};
    unordered_map<string, string> metadata;
    
    vector<int> imgSize_;
    int stride_ {};
    vector<string> classes_;
    unsigned numClasses {};
    vector<int64_t> inputTensorShape_;
    cv::Size cvSize_;
    string task_;
};

#endif