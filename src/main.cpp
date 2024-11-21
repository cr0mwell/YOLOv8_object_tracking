#include <iostream>
#include <fstream>
#include <chrono>
#include <ctime>

#include "inferenceDriverONNX.hpp"
#include "trackManager.hpp"

using namespace std;
using json = nlohmann::json;


int main(int argc, char** argv)
{
    /* INIT VARIABLES AND DATA STRUCTURES */
    
    // Set runtime variables
    // Visualize results frame by frame during the detection?
    bool toVisualizeProcess = getBoolOption("Visualize the results during processing? ");
    
    // Visualize results after processing?
    bool toVisualizeOutput = getBoolOption("Visualize the video results after processing? ");
    
    // Select task and set an appropriate model name extension
    string nameExtension = getTask();
    
    // Set the selected the model size
    string modelSize = getModelSize();
    
    bool verboseOutput = false;
    string modelFile = modelBasePath + "yolov8" + modelSize + nameExtension + ".onnx";
    
    /* READING FILES */
    
    // Input video file
    cv::VideoCapture inputVideo(inputVideoPath);
    if (!inputVideo.isOpened()) {
        cout << "Failed to read the input file: " <<  inputVideoPath << endl;
        return 1;
    }
    
    // Get video resolution and frame rate
    double fps = inputVideo.get(cv::CAP_PROP_FPS);
    int frameWidth = inputVideo.get(cv::CAP_PROP_FRAME_WIDTH);
    int frameHeight = inputVideo.get(cv::CAP_PROP_FRAME_HEIGHT);
    cout << "\nInput video resolution: " << frameHeight << "x" << frameWidth << ", frame rate: " << fps << endl;
    
    // Create video output object
    cv::VideoWriter outputVideo(outputVideoPath,
                                cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
                                fps,
                                cv::Size(frameWidth, frameHeight));
    
    // Create session file stream
    ofstream jf(sessionFile);
    json js;
    
    if (!jf.is_open()) {
        cout << "Failed to open session_dump.json file";
        return 1;
    
    } else {
        auto start = chrono::system_clock::to_time_t(chrono::system_clock::now());
        js["session start"] = string(ctime(&start));
        js["model_name"] = modelFile;
    }
    
    // Load class names from classesFile
    vector<string> classes;
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classes.push_back(line);
    
    // Loading the model
    PoseParams poseParams = PoseParams();
    OnnxDriver model(modelFile, onnxLogId, provider, classes.size());
    vector<cv::Scalar> colors = generateColors(classes.size());
    
    // Creating TrackManager
    TrackManager trackMgr;
    
    /* MAIN LOOP OVER ALL IMAGES */
    
    cv::Mat frame;
    unsigned f {1};
    
    while (inputVideo.read(frame)) {
        // Running inference
        vector<Results> objs = model.predict(frame, confThreshold, nmsThreshold, maskThreshold, verboseOutput);
        
        // Tracking management
        trackMgr.updateTracks(objs, verboseOutput);
        
        // Dump results to the file
        jsonDump(js, objs, f);
        
        populateResults(frame, objs, poseParams, model.getTask(), colors, classes);
        
        // Write video frame to output
        outputVideo.write(frame);
        
        if(toVisualizeProcess) {
            cv::imshow("Video frame " + to_string(f), frame);
            
            auto key = cv::waitKey(0);
            
            // Breaking the loop on 'ESC' key press
            if (key == 27)
                break;
            else
                cv::destroyAllWindows();
        }
        
        if(verboseOutput)
            cout << "Processed frame " << f << endl;
        f++;
    }
    
    auto end = chrono::system_clock::to_time_t(chrono::system_clock::now());
    js["session_end"] = string(ctime(&end));
    
    // Writting the session to the session_dump.json
    jf << js;
    jf.close();
    outputVideo.release();
    inputVideo.release();
    
    if(toVisualizeOutput) {
        cv::VideoCapture outputVideo(outputVideoPath);
        
        if (!outputVideo.isOpened()) {
            cout << "Failed to read the output file: " <<  outputVideoPath << endl;
            return 1;
        }
        
        while(outputVideo.read(frame)) {
            cv::imshow("Output video", frame);
            
            // Breaking the loop on 'ESC' key press
            if(cv::waitKey(30) == 27)
                break; 
        }
    }

    outputVideo.release();

    return 0;
}