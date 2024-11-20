#ifndef HELPERS_H
#define HELPERS_H

#include <fstream>

#include "../include/json.hpp"
#include "structures.h"

using namespace std;

template<typename T> void processInput(char &option);
bool getBoolOption(string message);
string getTask();
cv::Scalar generateColor();
vector<cv::Scalar> generateColors(int numClasses);
int64_t getTensorParamSize(vector<int64_t> shape);
vector<int> parseVectorString(const string& input);
void createBBox(float* pdata, vector<float>& scaleFactors, vector<cv::Rect>& boxes);
void populateResults(cv::Mat img, vector<Results>& results, PoseParams& poseParams, const string& task, vector<cv::Scalar> color, vector<string> classes);
void jsonDump(nlohmann::json& j, vector<Results>& results, unsigned frame);

#endif