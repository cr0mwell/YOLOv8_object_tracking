#ifndef TRACK_MANAGER_H
#define TRACK_MANAGER_H

#include <opencv2/opencv.hpp>
#include <vector>

#include "structures.h"

using namespace std;


class TrackManager {
private:
    int currentId {1};
    vector<Track> tracks;
    cv::Mat associations;
    vector<int> unassignedTracks;
    vector<int> unassignedDetections;
    
    void associate(vector<Results>& results);
    float iou(const cv::Rect& tBbox, const cv::Rect& dBbox);
    cv::Point getAssociation();
    
public:
    void updateTracks(vector<Results>& results, bool verbose);
};

#endif