#include <iostream>
#include <numeric>

#include "trackManager.hpp"

using namespace std;


void TrackManager::associate(vector<Results>& results) {
    int T = tracks.size();
    int D = results.size();
    
    associations = cv::Mat(T, D, CV_32F, cv::Scalar(0.0));
    unassignedTracks.clear();
    unassignedTracks.resize(T);
    unassignedDetections.clear();
    unassignedDetections.resize(D);
    iota(unassignedTracks.begin(), unassignedTracks.end(), 0);
    iota(unassignedDetections.begin(), unassignedDetections.end(), 0);
    //cout << "unassignedTracks size: " << unassignedTracks.size() << endl;
    //cout << "unassignedDetections size: " << unassignedDetections.size() << endl;
    
    // Creating the associations matrix
    for(unsigned i=0; i<T; i++)
        for(unsigned j=0; j<D; j++)
            associations.at<float>(i, j) = iou(tracks[i].bbox, results[j].bbox);
}

float TrackManager::iou(const cv::Rect& tBbox, const cv::Rect& dBbox) {
    // Calculating intersection
    float tW = tBbox.x + tBbox.width;
    float tH = tBbox.y + tBbox.height;
    float dW = dBbox.x + dBbox.width;
    float dH = dBbox.y + dBbox.height;
    
    float intWidth = min(tW, dW) - max(tBbox.x, dBbox.x);
    float intHeight = min(tH, dH) - max(tBbox.y, dBbox.y);
    
    float intersection = intWidth > 0.0 && intHeight > 0.0 ? intWidth*intHeight : 0.0;
    
    // Calculating union (areaTBox + areaDBox - union_)
    float areaTBox = tBbox.width * tBbox.height;
    float areaDBox = dBbox.width * dBbox.height;
    float union_ = areaTBox + areaDBox - intersection;
    
    return union_ > 0.0 ? intersection/union_ : 0.0;;
}

cv::Point TrackManager::getAssociation() {
    
    if(associations.empty()) {
        return cv::Point(INT_INF, INT_INF);
    
    } else {
        cv::Point indices;
        double maxVal;
        cv::minMaxLoc(associations, 0, &maxVal, 0, &indices);
        //cout << "associations: " << endl << associations << endl;
        if(maxVal != 0.0) {
            // Setting row and column to inf
            associations.col(indices.x) = 0.0;
            associations.row(indices.y) = 0.0;
            
            // Remove respective elements from unassignedTracks and unassignedDetections
            unassignedTracks[indices.y] = REMOVED;
            unassignedDetections[indices.x] = REMOVED;
            //cout << fixed << setprecision(4) << "Got " << maxVal << " at " << indices.y << ", " << indices.x << endl;
            return indices;
        } else {
            return cv::Point(INT_INF, INT_INF);
        }
    }
}

void TrackManager::updateTracks(vector<Results>& results, bool verbose) {
    // Creating association matrix
    associate(results);
    
    // Updating tracks and results
    while(true) {
        cv::Point indices = TrackManager::getAssociation();
        int dId = indices.x;
        int tId = indices.y;
        
        // No more associations left
        if(tId == INT_INF) {
            break;
        }
        
        // Updating track
        tracks[tId].score = min(tracks[tId].score + 1, CONFIRMED);
        tracks[tId].bbox = results[dId].bbox;
        
        // Setting Result.trackId if the track is confirmed
        if(tracks[tId].score == CONFIRMED) {
            results[dId].trackId = tracks[tId].id;
            //cout << "Setting track id for result " << dId << ": " << tracks[tId].id << endl;
        }
    }
    
    // Creating new
    for(auto const& i: unassignedDetections)
        if(i != REMOVED) {
            if(verbose)
                cout << "Creating new track " << currentId << endl;
            tracks.push_back(Track(currentId, results[i].bbox));
            currentId++;
            
            // Resetting trackId for the Result (lost track case)
            results[i].trackId = ORPHANED;
        }
    
    // Managing unassigned tracks
    for(auto const& i: unassignedTracks)
        if(i != REMOVED) {
            tracks[i].score -= 1;
            if(verbose)
                cout << "Decreasing score for track " << tracks[i].id << " to " << tracks[i].score << endl;
            
            // Removing orphaned tracks
            if(tracks[i].score == ORPHANED) {
                if(verbose)
                    cout << "Removing track " << i << "(id:" << tracks[i].id << ")" << endl;
                tracks.erase(tracks.begin() + i);
            }
        }
}