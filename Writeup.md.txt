# SFND 2D Feature Tracking




### MP.1 Data Buffer Optimization

* Implement a vector for dataBuffer objects whose size does not exceed a limit (e.g. 2 elements). 
* This can be achieved by pushing in new elements on one end and removing elements on the other end.

'''c++
    int dataBufferSize = 2;       // no. of images which are held in memory (ring buffer) at the same time
    //vector<DataFrame> dataBuffer; // list of data frames which are held in memory at the same time
    bool bVis = false;            // visualize results
    boost::circular_buffer<DataFrame> dataBuffer;
    dataBuffer.set_capacity(dataBufferSize);
'''

### MP.2 Keypoint Detection

* Implement detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly.

'MidTermProject_Camera_Student.cpp'
'''c++
        else if (detectorType.compare("HARRIS") == 0)
        {
            detKeypointsHarris(keypoints, imgGray);
        }
        else if (detectorType.compare("FAST") == 0 || detectorType.compare("ORB") == 0 || detectorType.compare("BRISK") == 0
        || detectorType.compare("AKAZE") == 0 || detectorType.compare("SIFT") == 0)
        {
            detKeypointsModern(keypoints, imgGray, detectorType, false);
        }
'''
'matching2D_Student.cpp'
'''c++
void detKeypointsHarris(std::vector<cv::KeyPoint> &keypoints, cv::Mat &imgGray)
{
                // Detector parameters
            int blockSize = 2; // for every pixel, a blockSize x blockSize neighborhood is considered
            int apertureSize = 3; // aperture parameter for Sobel operator (must be odd)
            int minResponse = 100; // minimum value for a corner in the 8bit scaled response matrix
            double k = 0.04; // Harriss parameter 

            // Detect Harris corners and normalize output
            cv::Mat dst, dst_norm, dst_norm_scaled;
            dst = cv::Mat::zeros(imgGray.size(), CV_32FC1);
            cv::cornerHarris(imgGray, dst, blockSize, apertureSize, k, cv::BORDER_DEFAULT);
            cv::normalize(dst, dst_norm, 0, 255, cv::NORM_MINMAX, CV_32FC1, cv::Mat());
            cv::convertScaleAbs(dst_norm, dst_norm_scaled);
            // Visualize results
            /*string windowName = "Harris COrner Detector Response Matrix";
            cv::namedWindow(windowName, 4);
            cv::imshow(windowName, dst_norm_scaled);
            cv::waitKey(0);*/

            //vector<cv::KeyPoint> keypoints;
            double maxOverlap = 0.0;
            for (size_t j = 0; j < dst_norm.rows; j++){
                for (size_t i = 0; i < dst_norm.cols; i++){
                    int response = (int)dst_norm.at<float>(j,i);
                    if(response > minResponse){
                        cv::KeyPoint newKeyPoint;
                        newKeyPoint.pt = cv::Point2f(i, j);
                        newKeyPoint.size = 2 * apertureSize;
                        newKeyPoint.response = response;

                        // Perform Non-maximum suppression (NMS) in local neigborhood around new key points
                        bool bOverlap = false;
                        for (auto it = keypoints.begin(); it != keypoints.end(); ++it){
                            double kptOverlap = cv::KeyPoint::overlap(newKeyPoint, *it);
                            if (kptOverlap > maxOverlap){
                                bOverlap = true;
                                if (newKeyPoint.response > (*it).response){
                                    *it =newKeyPoint;
                                    break;
                                }
                                
                            }
                        }
                        if(!bOverlap){
                            keypoints.push_back(newKeyPoint);
                        }
                    }
                } // eof loop over cols
            } // eof loop over rows
            /*
            string windowName = "Harris Corner Detection Results";
            cv::namedWindow(windowName, 5);
            cv::Mat visImage = dst_norm_scaled.clone();
            cv::drawKeypoints(dst_norm_scaled, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
            cv::imshow(windowName, visImage);
            cv::waitKey(0);*/
}

void detKeypointsModern(std::vector<cv::KeyPoint> &keypoints, cv::Mat &img, std::string detectorType, bool bVis)
{
    cout << "Detector Created: " << detectorType << endl;cout << "Detector Created: " << detectorType << endl;
    cv::Ptr<cv::Feature2D> detector;
    //Create detectors
    if(detectorType.compare("FAST") == 0){
        detector = cv::FastFeatureDetector::create();
        detector->detect(img, keypoints);
    } 
    else if(detectorType.compare("ORB") == 0) {
        detector = cv::ORB::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
        detector->detect(img, keypoints);
    }
    else if(detectorType.compare("AKAZE")== 0){
        detector = cv::AKAZE::create();
        detector->detect(img, keypoints);
    } 
    else if(detectorType.compare("SIFT") == 0)
    {
        detector = cv::xfeatures2d::SIFT::create();
        detector->detect(img, keypoints);
    }
    // Visualize
    if (bVis){
        cv::Mat visImage = img.clone();
        cv::drawKeypoints(img, keypoints, visImage, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
        string windowName = detectorType + " Detection Results";
        cv::namedWindow(windowName, 6);
        imshow(windowName, visImage);
        cv::waitKey(0);
    }

}
'''

### MP.3 Keypoint Removal

* Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing.

'''c++
        // only keep keypoints on the preceding vehicle
        bool bFocusOnVehicle = true;
        cv::Rect vehicleRect(535, 180, 180, 150); //cx = 535, cy = 180, w = 180, h = 150
        if (bFocusOnVehicle)
        {
            // Iterate through keypoints
            for(keypoint = keypoints.begin(); keypoint != keypoints.end(); ++keypoint)
            {
                if (vehicleRect.contains(keypoint->pt)){
                   cv::KeyPoint keyPointInRange;
                   keyPointInRange.pt = cv::Point2f(keypoint->pt);
                   keyPointInRange.size = 1;
                keypoints_target.push_back(keyPointInRange);
                }
            }
            keypoints = keypoints_target;
        }

        keypoints_csv << "," << keypoints.size();
        //// EOF STUDENT ASSIGNMENT
'''

### MP.4 Keypoint Descriptors

* Implement descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly.
'MidTermProject_Camera_Student.cpp'
'''c++
            vector<cv::DMatch> matches;
            string matcherType = "MAT_BF";        // MAT_BF, MAT_FLANN
            string descriptor_Type; // DES_BINARY, DES_HOG
            if (descriptorType.compare("SIFT") == 0)
            {
                descriptor_Type = "DES_HOG";
            } else {
                descriptor_Type = "DES_BINARY";
            }
'''
'matching2D_Student.cpp'
'''c++
    if (descriptorType.compare("BRISK") == 0)
    {

        int threshold = 30;        // FAST/AGAST detection threshold score.
        int octaves = 3;           // detection octaves (use 0 to do single scale)
        float patternScale = 1.0f; // apply this scale to the pattern used for sampling the neighbourhood of a keypoint.

        extractor = cv::BRISK::create(threshold, octaves, patternScale);
    }
    else if (descriptorType.compare("ORB") == 0)
    {
        extractor = cv::ORB::create(); // Created with Default Parameters
    }
    else if (descriptorType.compare("FREAK") == 0){
        extractor = cv::xfeatures2d::FREAK::create();
    }
    else if (descriptorType.compare("AKAZE") == 0){
        extractor = cv::AKAZE::create();
    }
    else if (descriptorType.compare("SIFT") == 0 ){
        extractor = cv::xfeatures2d::SIFT::create();
    }
    else if (descriptorType.compare("BRIEF") == 0){
        extractor = cv::xfeatures2d::BriefDescriptorExtractor::create();
    }
'''

### MP.5 Descriptor Matching

* Implement FLANN matching as well as k-nearest neighbor selection. Both methods must be selectable using the respective strings in the main function.
'matching2D_Student.cpp'
'''c++

    if (matcherType.compare("MAT_BF") == 0)
    {
        cout << "matchDescriptors() " << descriptorType << endl;
        
        //Bug fix for SIFT
        if (descriptorType.compare("DES_HOG") == 0){
            cout << "Converting Data Source " << endl;
            if (descSource.type() != CV_32F){
                descSource.convertTo(descSource, CV_32F);
                descRef.convertTo(descRef, CV_32F);
            }
        }
        int normType = descriptorType.compare("DES_BINARY") == 0 ? cv::NORM_HAMMING : cv::NORM_L2;
        matcher = cv::BFMatcher::create(normType, crossCheck);
        cout << "BF matching cross-check=" << crossCheck;
    }
    else if (matcherType.compare("MAT_FLANN") == 0)
    {
        if (descSource.type() != CV_32F){
            descSource.convertTo(descSource, CV_32F);
            descRef.convertTo(descRef, CV_32F);
        }
        matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::FLANNBASED);
    }
'''

### MP.6 Descriptor Distance Ratio

* Use the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints.

'''c++
  // perform matching task
    if (selectorType.compare("SEL_NN") == 0)
    { // nearest neighbor (best match)

        matcher->match(descSource, descRef, matches); // Finds the best match for each descriptor in desc1
    }
    else if (selectorType.compare("SEL_KNN") == 0)
    { // k nearest neighbors (k=2)
        vector<vector<cv::DMatch>> knn_matches;
        double t = (double)cv::getTickCount();
        matcher->knnMatch(descSource, descRef, knn_matches, 2); // finds the 2 best matches
        t = ((double)cv::getTickCount()-t) / cv::getTickFrequency();
        cout << " (KNN) with n=" << knn_matches.size() << " matches in " << 1000 * t / 1.0 << " ms" << endl;

        // filter matches using descriptor distance ratio test
        double minDescDistRatio = 0.8;
        for (auto it = knn_matches.begin(); it != knn_matches.end(); ++it){
            if ((*it)[0].distance < (minDescDistRatio * (*it)[1].distance))
            {
                matches.push_back((*it)[0]);
                
            }
        }
        cout << "# keypoints removed = " << knn_matches.size() - matches.size() << endl;
        // ...
    }
'''

### MP.7 Performance Evaluation 1

* Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

SHITOMASI,125,118,123,120,120,113,114,123,111,112
HARRIS,17,14,18,21,26,43,18,31,26,34
FAST,419,427,404,423,386,414,418,406,396,401
BRISK,264,282,282,277,297,279,289,272,266,254
ORB,92,102,106,113,109,125,130,129,127,128
AKAZE,166,157,161,155,163,164,173,175,177,179
SIFT,138,132,124,137,134,140,137,148,159,137

### MP.8 Performance Evaluation 2

* Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.
SHITOMASI,BRISK, 95, 88, 80, 90, 82, 79, 85, 86, 82
SHITOMASI,BRIEF, 115, 111, 104, 101, 102, 102, 100, 109, 100
SHITOMASI,ORB, 106, 102, 99, 102, 103, 97, 98, 104, 97
SHITOMASI,FREAK, 86, 90, 86, 88, 86, 80, 81, 86, 85
HARRIS,BRISK, 12, 10, 14, 15, 16, 16, 15, 23, 21
HARRIS,BRIEF, 14, 11, 15, 20, 24, 26, 16, 24, 23
HARRIS,ORB, 12, 12, 15, 18, 24, 20, 15, 24, 22
HARRIS,FREAK, 13, 13, 15, 15, 17, 20, 12, 21, 18
FAST,BRISK, 256, 243, 241, 239, 215, 251, 248, 243, 247
FAST,BRIEF, 320, 332, 299, 331, 276, 327, 324, 315, 307
FAST,ORB, 307, 308, 298, 321, 283, 315, 323, 302, 311
FAST,FREAK, 251, 247, 233, 255, 231, 265, 251, 253, 247
BRISK,BRISK, 151, 163, 146, 173, 150, 162, 182, 183, 161
BRISK,BRIEF, 178, 205, 185, 179, 183, 195, 207, 189, 183
BRISK,ORB, 169, 199, 175, 187, 179, 185, 200, 203, 178
BRISK,FREAK, 144, 162, 147, 170, 148, 172, 173, 178, 166
ORB,BRISK, 48, 46, 57, 61, 56, 68, 64, 80, 73
ORB,BRIEF, 49, 43, 45, 59, 53, 78, 68, 84, 66
ORB,ORB, 49, 46, 58, 58, 52, 75, 67, 83, 74
ORB,FREAK, 42, 43, 52, 52, 58, 63, 58, 75, 69
AKAZE,BRISK, 135, 121, 128, 126, 126, 133, 137, 144, 143
AKAZE,BRIEF, 141, 134, 131, 130, 134, 146, 150, 148, 152
AKAZE,ORB, 139, 134, 132, 124, 139, 141, 144, 157, 150
AKAZE,FREAK, 126, 126, 130, 117, 126, 128, 147, 146, 133
SIFT,SIFT, 75, 72, 65, 75, 59, 64, 64, 69, 81

### MP.9 Performance Evaluation 3

* Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.
SHITOMASI,BRISK,366.285,370.378,364.969,356.25,363.45,365.581,370.177,368.369,363.362
SHITOMASI,BRIEF,28.9851,28.9502,29.421,28.4481,28.3939,30.1844,28.8138,28.5457,27.0149
SHITOMASI,ORB,29.4755,28.3518,28.4538,29.8754,29.0279,28.8227,29.0977,30.7016,27.8302
SHITOMASI,FREAK,68.6146,63.0452,64.0332,66.7698,64.2949,63.0614,62.8321,64.0848,63.6239
HARRIS,BRISK,355.252,356.928,353.723,357.129,369.735,351.345,354.866,360.338,365.776
HARRIS,BRIEF,21.7653,21.5766,21.8582,21.6764,38.6132,21.8998,25.4265,24.4936,28.8987
HARRIS,ORB,23.0807,23.6432,23.72,24.0069,39.2318,21.8016,24.5845,23.8784,28.7705
HARRIS,FREAK,61.8482,63.3452,62.4174,62.9939,76.2793,62.2193,64.9552,63.0208,67.9051
FAST,BRISK,351.411,349.203,349.208,347.155,350.03,349.593,348.27,350.509,350.386
FAST,BRIEF,14.4748,14.0269,13.903,14.6337,14.286,15.343,13.8435,14.5278,13.9084
FAST,ORB,14.2071,14.9675,14.5059,15.2509,14.5437,15.2289,13.8224,14.5477,13.8632
FAST,FREAK,55.8214,56.1202,59.4118,58.6718,60.693,58.5312,56.1733,57.1622,55.8713
BRISK,BRISK,719.939,719.113,725.071,725.931,734.487,716.665,722.147,725.687,719.66
BRISK,BRIEF,387.728,387.375,387.598,389.003,386.77,386.173,385.11,389.276,390.409
BRISK,ORB,387.432,391.647,387.172,388.662,388.538,390.62,383.502,386.274,386.221
BRISK,FREAK,429.304,426.044,425.589,431.649,431.203,432.019,433.775,428.29,432.484
ORB,BRISK,348.163,349.059,349.835,351.034,351.961,348.743,346.762,351.912,350.204
ORB,BRIEF,19.3439,16.6097,16.7527,16.5852,16.8661,17.2476,17.4521,18.9999,17.0905
ORB,ORB,16.5956,17.2186,17.2501,16.8695,17.2193,17.1188,18.1924,17.2376,17.2522
ORB,FREAK,56.7917,57.5433,56.559,57.4599,57.865,59.1882,57.5429,57.4843,59.0854
AKAZE,BRISK,411.781,418.712,424.073,425.196,424.591,422.902,417.35,424.14,417.8
AKAZE,BRIEF,80.5827,83.2824,84.3952,83.221,84.0877,84.6529,86.9041,86.3011,79.8876
AKAZE,ORB,86.6018,89.8448,82.7054,85.4521,86.4968,80.7931,81.3817,78.943,89.1851
AKAZE,FREAK,129.692,130.089,125.275,122.82,130.367,126.909,130.756,130.978,125.101
SIFT,SIFT,164.409,165.37,160.673,165.005,160.93,168.834,163.823,164.222,161.888


### Justification

The Fast Detector with the Brief Descriptor is my recommendation based on observations.
The combination takes less than 15 ms and returns the highest number of matched key points
Points which are in the region of interest.
The next two combinations would be the Fast & Orb and then the Fast & Brief even though it takes more time
