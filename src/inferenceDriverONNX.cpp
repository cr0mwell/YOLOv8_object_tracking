#include <iostream>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "inferenceDriverONNX.hpp"

using namespace std;
using namespace cv;


OnnxDriver::OnnxDriver(string modelPath, const string logId, const OnnxProviders provider, const int classes)
        : modelPath_(modelPath), numClasses(classes) {

    env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, logId.c_str());
    
    // Setting the provider (CPU or GPU)
    Ort::SessionOptions sessionOptions = Ort::SessionOptions();
    vector<string> availableProviders = Ort::GetAvailableProviders();
    OrtCUDAProviderOptions cudaOption;

    if (provider == OnnxProviders::CUDA) {
        auto cudaAvailable = std::find(availableProviders.begin(), availableProviders.end(), "CUDAExecutionProvider");
        if (cudaAvailable == availableProviders.end())
            cout << "CUDA is not supported on the current device. Using CPU." << endl;
        else
            sessionOptions.AppendExecutionProvider_CUDA(cudaOption);
    
    } else if (provider != OnnxProviders::CPU) {
        throw runtime_error("Unknown provider");
    }
    
    sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_EXTENDED);
    
    // Creating a session
    #ifdef _WIN32
        std::wstring model_path(modelPath.begin(), modelPath.end());
        session = Ort::Session(env, model_path.c_str(), sessionOptions);
    #else
        session = Ort::Session(env, modelPath.c_str(), sessionOptions);
    #endif
    
    // Set inputNames
    Ort::AllocatorWithDefaultOptions inputAllocator;
    auto inputNodesNum = session.GetInputCount();
    for (unsigned i=0; i<inputNodesNum; i++) {
        auto inputName = session.GetInputNameAllocated(i, inputAllocator);
        tmpInVector.push_back(move(inputName));
        inputNames.push_back(tmpInVector.back().get());
    }
    
    // Set outputNames
    Ort::AllocatorWithDefaultOptions outputAllocator;
    auto outputNodesNum = session.GetOutputCount();
    for (unsigned i=0; i<outputNodesNum; i++)
    {
        auto outputName = session.GetOutputNameAllocated(i, outputAllocator);
        tmpOutVector.push_back(move(outputName));
        outputNames.push_back(tmpOutVector.back().get());
    }
    
    auto shapeInput = session.GetInputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    auto shapeOutput = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
    
    // Initialize model metadata
    modelMetadata = session.GetModelMetadata();
    Ort::AllocatorWithDefaultOptions metadataAllocator;

    vector<Ort::AllocatedStringPtr> metadataAllocatedKeys = modelMetadata.GetCustomMetadataMapKeysAllocated(metadataAllocator);
    vector<string> metadataKeys;
    metadataKeys.reserve(metadataAllocatedKeys.size());

    for (const Ort::AllocatedStringPtr& allocatedString: metadataAllocatedKeys) {
        metadataKeys.push_back(string(allocatedString.get()));
    }
    
    cout << "\n##############################" << endl;
    cout << "Model metadata:" << endl;
    cout << "##############################" << endl;
    
    cout << "Inference device: " << (provider == OnnxProviders::CPU ? "CPU" : "GPU") << endl;
    
    std::cout << "Model input shape: (";
    for(unsigned i=0; i<shapeInput.size(); i++)
        cout << shapeInput[i] << ", ";
    cout << ")" << endl;

    std::cout << "Model output shape: (";
    for(unsigned i=0; i<shapeOutput.size(); i++)
        cout << shapeOutput[i] << ", ";
    cout << ")" << endl;
    
    for (const string& key: metadataKeys) {
        Ort::AllocatedStringPtr metadataValue = modelMetadata.LookupCustomMetadataMapAllocated(key.c_str(), metadataAllocator);
        if (metadataValue != nullptr) {
            metadata[key] = string(metadataValue.get());
            cout << key << ": " << metadata[key] << endl;
        }
    }
    cout << "##############################" << endl;
    
    // Set task
    auto task = metadata.find("task");
    if (task != metadata.end())
        task_ = task->second;
    
    // Set input tensor shape
    auto imgSizeIt = metadata.find("imgsz");
    if(imgSizeIt != metadata.end())
        imgSize_ = parseVectorString(imgSizeIt->second);
    
    // Set inputTensorShape_
    if(!imgSize_.empty()) {
        inputTensorShape_ = {1, 3, getHeight(), getWidth()};
    }
}

const vector<int64_t>& OnnxDriver::getInputTensorShape() {
    return inputTensorShape_;
}

const int& OnnxDriver::getWidth() {
    return imgSize_[1];
}

const int& OnnxDriver::getHeight() {
    return imgSize_[0];
}

const string& OnnxDriver::getTask() {
    return task_;
}

void getMask(const Mat& maskProposals, const Mat& maskProtos, Results& result, const MaskParams& maskParams) {
    int netWidth = maskParams.networkWidth;
    int netHeight = maskParams.networkHeight;
    int segChannels = maskProtos.size[1];
    int segHeight = maskProtos.size[2];
    int segWidth = maskProtos.size[3];
    float maskThreshold = maskParams.maskThreshold;
    vector<float> params = maskParams.params;
    Size srcImgShape = maskParams.srcImgShape;
    Rect bbox = cv::Rect(cv::Point(result.bbox.x, result.bbox.y),
                         cv::Point(min(result.bbox.x + result.bbox.width, srcImgShape.width),
                                   min(result.bbox.y + result.bbox.height, srcImgShape.height)));

    // Crop from mask_protos
    int rangX = floor(bbox.x * params[0] / netWidth * segWidth);
    int rangY = floor(bbox.y * params[1] / netHeight * segHeight);
    int rangW = ceil(((bbox.x + bbox.width) * params[0]) / netWidth * segWidth) - rangX;
    int rangH = ceil(((bbox.y + bbox.height) * params[1]) / netHeight * segHeight) - rangY;

    rangW = max(rangW, 1);
    rangH = max(rangH, 1);
    if(rangX + rangW > segWidth) {
        if(segWidth - rangX > 0)
            rangW = segWidth - rangX;
        else
            rangX -= 1;
    }
    if(rangY + rangH > segHeight) {
        if(segHeight - rangY > 0)
            rangH = segHeight - rangY;
        else
            rangY -= 1;
    }

    std::vector<Range> roiRanges;
    roiRanges.push_back(Range(0, 1));
    roiRanges.push_back(Range::all());
    roiRanges.push_back(Range(rangY, rangH + rangY));
    roiRanges.push_back(Range(rangX, rangW + rangX));

    // Crop
    Mat tempMaskProtos = maskProtos(roiRanges).clone();
    Mat protos = tempMaskProtos.reshape(0, {segChannels, rangW * rangH});
    Mat matmulRes = (maskProposals * protos).t();
    Mat masksFeature = matmulRes.reshape(1, {rangH,rangW});
    Mat dest, mask;

    // Sigmoid
    exp(-masksFeature, dest);
    dest = 1.0 / (1.0 + dest);

    int left = floor((netWidth / segWidth * rangX - params[2]) / params[0]);
    int top = floor((netHeight / segHeight * rangY - params[3]) / params[1]);
    int width = ceil(netWidth / segWidth * rangW / params[0]);
    int height = ceil(netHeight / segHeight * rangH / params[1]);

    resize(dest, mask, Size(width, height), INTER_NEAREST);
    Rect maskRect = bbox - Point(left, top);
    maskRect &= Rect(0, 0, width, height);
    mask = mask(maskRect) > maskThreshold;
    if (mask.rows != bbox.height || mask.cols != bbox.width)
        resize(mask, mask, bbox.size(), INTER_NEAREST);
    
    result.mask = mask;
}

vector<Results> OnnxDriver::predict(Mat& image, const float& confThreshold, const float& nmsThreshold, const float& maskThreshold, bool verbose) {
    auto preprocessTimeStart = chrono::high_resolution_clock::now();
    
    // PREPROCESSING
    // Generate 4D blob from input image
    Mat blob;
    double scalefactor = 1/255.0;
    Size size = Size(getWidth(), getHeight());
    Scalar mean = Scalar(0,0,0);
    bool swapRB = true;
    bool crop = false;
    
    dnn::blobFromImage(image, blob, scalefactor, size, mean, swapRB, crop);
    
    Ort::MemoryInfo memoryInfo = Ort::MemoryInfo::CreateCpu(OrtAllocatorType::OrtArenaAllocator, OrtMemType::OrtMemTypeDefault);
    vector<int64_t> inputTensorShape = getInputTensorShape();
    int64_t inputTensorSize = getTensorParamSize(inputTensorShape);
    vector<Ort::Value> inputTensors;
    inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo, (float*)blob.data, inputTensorSize, inputTensorShape.data(), inputTensorShape.size()));
    
    auto end = std::chrono::high_resolution_clock::now();
    double preprocessTime = std::chrono::duration<double>(end - preprocessTimeStart).count();
    
    // INFERENCE
    auto inferenceTimeStart = chrono::high_resolution_clock::now();
    vector<Ort::Value> outputTensors = session.Run(Ort::RunOptions {nullptr},
                                                   inputNames.data(),
                                                   inputTensors.data(),
                                                   inputNames.size(),
                                                   outputNames.data(),
                                                   outputNames.size());
    
    end = std::chrono::high_resolution_clock::now();
    double inferenceTime = std::chrono::duration<double>(end - inferenceTimeStart).count();
    
    // POSTPROCESSING
    auto postprocessTimeStart = chrono::high_resolution_clock::now();
    
    // Get output info
    vector<int64_t> detectionShape = outputTensors[0].GetTensorTypeAndShapeInfo().GetShape();
    float* allData = outputTensors[0].GetTensorMutableData<float>();
    // [bs, features, preds_num]=>[bs, preds_num, features] (1, 116, 8400) for YOLOv8 with 640*640 input
    Mat detections = Mat(Size((int)detectionShape[2], (int)detectionShape[1]), CV_32F, allData).t();
    
    // Create container for the results
    vector<Results> results;
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    vector<vector<PoseKeyPoint>> poseKpts;
    vector<vector<float>> allMaskProposals, selectedMaskProposals;
    Size imageSize = image.size();
    vector<float> scaleFactors {(float)getWidth() / (float)imageSize.width,
                                (float)getHeight() / (float)imageSize.height};
    int dataWidth = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape()[1];
    int keyPointLength = dataWidth - 5;
    string task = getTask();
    
    // Key point shape: [x, y, confidence]
    if(task == Tasks::POSE && KEY_POINT_NUM * 3 != keyPointLength)
        throw invalid_argument("Pose should be shape [x, y, confidence] with 17-points");
    
    // Output format of YOLOv8 model:
    // BBox rect parameters {x, y, w, h}
    // number of classes
    // number of semantic masks weights (available for segmentation models only)
    int rows = detections.rows;
    float* pdata = (float*)detections.data;

    for(unsigned i=0; i<rows; i++)
    {
        if(task != Tasks::POSE) {
            Point classId;
            double maxConf;
            Mat scores(1, numClasses, CV_32FC1, pdata + 4);
            minMaxLoc(scores, 0, &maxConf, 0, &classId);

            //if (maxConf > confThreshold)
            // Filter out all but persons
            if(classId.x == 0 && maxConf > confThreshold)
            {
                if(task == Tasks::SEGMENT) {
                    vector<float> maskProto(pdata + 4 + numClasses, pdata + dataWidth);
                    allMaskProposals.push_back(maskProto);
                }
                createBBox(pdata, scaleFactors, boxes);
                classIds.push_back(classId.x);
                confidences.push_back((float) maxConf);
            }
        
        } else {
            classIds.push_back(0);
            confidences.push_back(pdata[4]);
            createBBox(pdata, scaleFactors, boxes);
            
            std::vector<PoseKeyPoint> kpts;
            for(unsigned i=0; i<keyPointLength; i += 3) {
                PoseKeyPoint kpt;
                kpt.x = pdata[5 + i] / scaleFactors[0];
                kpt.y = pdata[6 + i] / scaleFactors[1];
                kpt.confidence = pdata[7 + i];
                kpts.push_back(kpt);
            }
            poseKpts.push_back(kpts);
        }
        
        // Next prediction
        pdata += dataWidth;
    }
    
    // Non-maximum supression
    vector<int> nmsResults;
    dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, nmsResults);
    for(const auto &i: nmsResults)
    {
        Results result;
        result.classId = classIds[i];
        result.confidence = confidences[i];
        result.bbox = boxes[i];
        if(task == Tasks::POSE)
            result.keyPoints = poseKpts[i];
        if(task == Tasks::SEGMENT)
            selectedMaskProposals.push_back(allMaskProposals[i]);
        results.push_back(result);
    }
    
    // Segmentation masks postprocessing
    if(task == Tasks::SEGMENT) {
        vector<int64_t> maskShape = outputTensors[1].GetTensorTypeAndShapeInfo().GetShape();
        std::vector<int> maskProtosShape = {1, (int)maskShape[1],(int)maskShape[2],(int)maskShape[3]};
        MaskParams maskParams;
        maskParams.params = scaleFactors;
        maskParams.srcImgShape = imageSize;
        maskParams.networkHeight = getWidth();
        maskParams.networkWidth = getHeight();
        maskParams.maskThreshold = maskThreshold;
        Mat maskProtos = Mat(maskProtosShape, CV_32F, outputTensors[1].GetTensorMutableData<float>());
        for (unsigned i=0; i<selectedMaskProposals.size(); i++)
            getMask(Mat(selectedMaskProposals[i]).t(), maskProtos, results[i], maskParams);
    }
    
    end = std::chrono::high_resolution_clock::now();
    double postprocessTime = std::chrono::duration<double>(end - postprocessTimeStart).count();

    if (verbose) {
        cout << fixed << setprecision(1);
        cout << "image: " << image.rows << "x" << image.cols << " " << results.size() << " objs, ";
        cout << (preprocessTime + inferenceTime + postprocessTime) * 1000.0 << "ms" << endl;
        cout << "Speed: " << (preprocessTime * 1000.0) << "ms preprocess, ";
        cout << (inferenceTime * 1000.0) << "ms inference, ";
        cout << (postprocessTime * 1000.0) << "ms postprocess per image ";
        cout << "at shape (1, " << image.channels() << ", " << image.rows << ", " << image.cols << ")" << endl;
    }

    return results;
}