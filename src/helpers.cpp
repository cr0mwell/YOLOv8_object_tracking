#include <regex>
#include <random>
#include <map>

#include "helpers.hpp"

using namespace std;


template<typename T>
void processInput(T &option) {
    cin >> option;
    cin.clear();
    cin.ignore(numeric_limits<streamsize>::max(), '\n');
}

bool getBoolOption(string message) {
    char drawOption;
    cout << message;
    processInput(drawOption);
    
    while (tolower(drawOption) != 'y' && tolower(drawOption) != 'n') {
        cout << "\nInvalid value entered (enter 'y' or 'n') ";
        processInput(drawOption);
    }
    
    return (drawOption == 'y');
}

void printTaskOptions() {
    cout << "Enter the task to perform (1-3):" << endl;
    cout << "1. Detection" << endl;
    cout << "2. Detection and segmentation" << endl;
    cout << "3. Detection and pose estimation" << endl;
}

void printModelOptions() {
    cout << "Enter the selected model size (1-5):" << endl;
    cout << "1. Nano" << endl;
    cout << "2. Small" << endl;
    cout << "3. Medium" << endl;
    cout << "4. Large" << endl;
    cout << "5. XLarge" << endl;
}

string getTask() {
    int task;
    vector<int> validOptions = {1, 2, 3};
    unordered_map<int, string> tasks {{1, ""},
                                      {2, "-seg"},
                                      {3, "-pose"}};
    printTaskOptions();
    
    processInput(task);
    while (find(validOptions.begin(), validOptions.end(), task) == validOptions.end()) {
        cout << "\nInvalid value entered" << endl;
        printTaskOptions();
        processInput(task);
    }
    
    return tasks[task];
}

string getModelSize() {
    int modelSize;
    vector<int> validOptions = {1, 2, 3, 4, 5};
    unordered_map<int, string> sizes {{1, "n"},
                                      {2, "s"},
                                      {3, "m"},
                                      {4, "l"},
                                      {5, "x"}};
    printModelOptions();
    
    processInput(modelSize);
    while (find(validOptions.begin(), validOptions.end(), modelSize) == validOptions.end()) {
        cout << "\nInvalid value entered" << endl;
        printModelOptions();
        processInput(modelSize);
    }
    
    return sizes[modelSize];
}

cv::Scalar generateColor() {
    random_device rd;
    mt19937 generator(rd());
    uniform_int_distribution<int> distribution(0, 255);

    cv::Scalar color;
    for (unsigned i=0; i<3; i++) {
        color[i] = distribution(generator);
    }

    return color;
}

vector<cv::Scalar> generateColors(int numClasses) {
    vector<cv::Scalar> colors;
    
    for (unsigned i=0; i< numClasses; i++) {
        cv::Scalar color = generateColor();
        colors.push_back(color);
    }
    return colors;
}

int64_t getTensorParamSize(vector<int64_t> shape){
    unsigned tensorParams {1};
    for (unsigned value: shape)
        tensorParams *= value;
    return tensorParams;
}

vector<int> parseVectorString(const string& input) {
    // Expected input examples: "[640, 640]"
    regex number_pattern(R"(\d+)");

    vector<int> result;
    sregex_iterator iterator(input.begin(), input.end(), number_pattern);
    sregex_iterator end;

    while(iterator != end) {
        result.push_back(stoi(iterator->str()));
        iterator++;
    }

    return result;
}

void createBBox(float* pdata, vector<float>& scaleFactors, vector<cv::Rect>& boxes) {
    //rect [x,y,w,h]
    float x = pdata[0] / scaleFactors[0];
    float y = pdata[1] / scaleFactors[1];
    float w = pdata[2] / scaleFactors[0];
    float h = pdata[3] / scaleFactors[1];
    int left = max(int(x - 0.5 * w), 0);
    int top = max(int(y - 0.5 * h), 0);
    
    boxes.push_back(cv::Rect(left, top, int(w + 0.5), int(h + 0.5)));
}

void populateResults(cv::Mat img, vector<Results>& results, PoseParams& poseParams,
                     const string& task, vector<cv::Scalar> color, vector<string> classes) {
    cv::Mat mask = img.clone();
    auto imgShape = img.size();
    
    bool segmentation = task == Tasks::SEGMENT;
    bool poseEstimation = task == Tasks::POSE;
    
    for (const auto& res: results) {
        // DETECTION
        float left = res.bbox.x;
        float top = res.bbox.y;
        
        // Draw bounding box
        // Gray color for unconfirmed tracks
        rectangle(img, res.bbox, res.trackId == 0 ? cv::Scalar(128, 128, 128) : color[res.classId], 2);

        // Create label
        stringstream labelStream;
        labelStream << "id: " << res.trackId << " " << classes[res.classId] << " " << fixed << setprecision(2) << res.confidence;
        string label = labelStream.str();

        cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
        cv::Rect rectToFill(left-1, top-textSize.height-5, textSize.width+2, textSize.height+5);
        cv::Scalar textColor = cv::Scalar(255.0, 255.0, 255.0);
        rectangle(img, rectToFill, res.trackId == 0 ? cv::Scalar(128, 128, 128) : color[res.classId], -1);
        putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, textColor, 2);
        
        // SEGMENTATION
        // Draw mask if available
        if (segmentation && res.mask.rows > 0 && res.mask.cols > 0)
            mask(res.bbox & cv::Rect(0, 0, img.cols, img.rows)).setTo(res.trackId == 0 ? cv::Scalar(128, 128, 128) : color[res.classId], res.mask);
        
        // POSE ESTIMATION
        // Check if keypoints are available
        if (poseEstimation && !res.keyPoints.empty()) {
            if (res.keyPoints.size() != KEY_POINT_NUM)
                continue;
            
            for (unsigned i=0; i<res.keyPoints.size(); i++) {
                PoseKeyPoint kpt = res.keyPoints[i];
                if (kpt.confidence < poseParams.kptThreshold)
                    continue;
                cv::Scalar kptColor = poseParams.posePalette[poseParams.kptColor[i]];
                cv::circle(img, cv::Point(kpt.x, kpt.y), poseParams.kptRadius, kptColor, -1, 8);
            }
            if (poseParams.isDrawKptLine) {
                for (unsigned i=0; i<poseParams.skeleton.size(); i++) {
                    PoseKeyPoint kpt0 = res.keyPoints[poseParams.skeleton[i][0] - 1];
                    PoseKeyPoint kpt1 = res.keyPoints[poseParams.skeleton[i][1] - 1];
                    if (kpt0.confidence < poseParams.kptThreshold || kpt1.confidence < poseParams.kptThreshold)
                        continue;
                    cv::Scalar kptColor = poseParams.posePalette[poseParams.limbColor[i]];
                    cv::line(img, cv::Point(kpt0.x, kpt0.y), cv::Point(kpt1.x, kpt1.y), kptColor, 2, 8);
                }
            }
        }
    }

    // Combine the image and mask
    addWeighted(img, 0.6, mask, 0.4, 0, img);
}

void jsonDump(nlohmann::json& j, vector<Results>& results, unsigned frame) {
    vector<nlohmann::json::object_t> resultsJson;
    for(auto &res: results) {
        nlohmann::json::object_t resultJson;
        resultJson["classId"] = res.classId;
        resultJson["trackId"] = res.trackId;
        resultJson["confidence"] = res.confidence;
        resultJson["bbox"] = nlohmann::json::array_t({res.bbox.x, res.bbox.y, res.bbox.width, res.bbox.height});
        if(size(res.keyPoints) != 0) {
            vector<nlohmann::json::array_t> keypoints;
            for(auto &kp: res.keyPoints)
                keypoints.push_back({kp.x, kp.y, kp.confidence});
            resultJson["keypoints"] = nlohmann::json(keypoints);
        }
            
        resultsJson.push_back(resultJson);
    }
    
    j["results_frame_" + to_string(frame)] = nlohmann::json(resultsJson);
}