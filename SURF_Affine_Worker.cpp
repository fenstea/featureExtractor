#include <array>
#include <cmath>
#include <stdio.h>
#include <iostream>
#include "opencv2/core/core.hpp"
#include "opencv2/features2d/features2d.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include "opencv2/nonfree/features2d.hpp"
#include "opencv2/video/tracking.hpp"

const int ITERATION_NUMBER = 5;
const double THRESHOLD = 10.0;
const double MATCHES_DISTANCE_THRESHOLD = 1.3;
const double MIN_DISTANCE = 2.0;
const double NEAR_ZERO = 0.01;
using namespace std;
using namespace cv;

void readme();

template <typename T>
inline T sqr(T squared) {
    return squared * squared;
}

bool isRectangleOk(Mat& A) {
    double det = A.at<double>(0, 0) * A.at<double>(1, 1) - A.at<double>(1, 0) * A.at<double>(0, 1);

    if (abs(det) < NEAR_ZERO) {
        return false;
    }

    return true;
}

bool isRectangleOk(std::vector<Point2f>& scene_corners) {
    std::array<double, 4> side;
    side[0] = sqrt(sqr(scene_corners[0].x - scene_corners[1].x) +
                          sqr(scene_corners[0].y - scene_corners[1].y));
    side[1] = sqrt(sqr(scene_corners[1].x - scene_corners[2].x) +
                          sqr(scene_corners[1].y - scene_corners[2].y));
    side[2] = sqrt(sqr(scene_corners[2].x - scene_corners[3].x) +
                          sqr(scene_corners[2].y - scene_corners[3].y));
    side[3] = sqrt(sqr(scene_corners[3].x - scene_corners[0].x) +
                          sqr(scene_corners[3].y - scene_corners[0].y));

    for (int index = 0; index < 4; ++index) {
        if (side[index] < MIN_DISTANCE) {
            return false;
        }
    }

    return true;
}

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
    if( argc != 4 ) {
        readme();
        return -1;
    }

    const char* img_object_name = argv[1];
    const char* img_scene_name = argv[2];
    const char* output_image_name = argv[3];

    Mat img_object = imread(img_object_name, IMREAD_GRAYSCALE );
    Mat img_scene = imread(img_scene_name, IMREAD_GRAYSCALE );

//    std::cout << img_object_name << ": " << img_object.cols << " x " << img_object.rows << "\n";
//    std::cout << img_scene_name << ": " << img_scene.cols << " x " << img_scene.rows << "\n";

    if( !img_object.data || !img_scene.data ) {
        std::cout << " --(!) Error reading images " << std::endl;
        return -1;
    }

    //-- Step 1: Detect the keypoints using SURF Detector
    int minHessian = 200;

    SurfFeatureDetector detector( minHessian );

    std::vector<KeyPoint> keypoints_object, keypoints_scene;

    detector.detect(img_object, keypoints_object );
    detector.detect(img_scene, keypoints_scene );

    /*
    std::cout << "Keypoints" << std::endl;
    for (size_t i = 0; i < keypoints_object.size(); ++i) {
        std::cout << "[" << keypoints_object[i].angle << " " 
                         << keypoints_object[i].pt.x << " "
                         << keypoints_object[i].pt.y << " "
                         << keypoints_object[i].size << "]\n";
    }
    */

    /*
    std::cout << "Keypoints count\n";
    std::cout << keypoints_object.size() << std::endl;
    std::cout << keypoints_scene.size() << std::endl;
    */

    //-- Step 2: Calculate descriptors (feature vectors)
    SurfDescriptorExtractor extractor;
    Mat descriptors_object, descriptors_scene;

    extractor.compute( img_object, keypoints_object, descriptors_object );
    extractor.compute( img_scene, keypoints_scene, descriptors_scene );

/*
    std::cout << "Descriptors" << std::endl;
    std::cout << descriptors_object << std::endl;
*/

    //-- Step 3: Matching descriptor vectors using FLANN matcher
//    FlannBasedMatcher matcher;
    std::vector<vector<DMatch> > allMatches;
    std::vector<DMatch> matches;
//    matcher.match( descriptors_object, descriptors_scene, matches );
    const string algorithmName = "FlannBased";
//    const string algorithmName = "BruteForce";
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create(algorithmName);
    matcher->knnMatch(descriptors_object, descriptors_scene, allMatches, 2);

    // Looking at matches[i][0] and matches[i][1]
    for (size_t index = 0; index < allMatches.size(); ++index) {
/*        
        std::cout << "[" << index << "] "
                  << allMatches[index][0].distance << " " 
                  << allMatches[index][1].distance << std::endl;
*/                  
        if (MATCHES_DISTANCE_THRESHOLD * allMatches[index][0].distance < allMatches[index][1].distance) {
            matches.push_back(allMatches[index][0]);
        }
    }

    if (0 == matches.size()) {
        // Not enough good matches
        return 1;
    }

/*
    std::cout << "Matches" << std::endl;
    for (size_t i = 0; i < matches.size(); ++i) {
        std::cout << "[" << i << "] [" 
                         << matches[i].distance << " " 
                         << matches[i].imgIdx << " "
                         << matches[i].queryIdx << " "
                         << matches[i].trainIdx << "]\n";
    }
*/

    std::array<CvScalar, ITERATION_NUMBER> colors = { CV_RGB(0, 0, 255), // Blue
                                                      CV_RGB(0, 255, 0), // Green
                                                      CV_RGB(255, 255, 0), // Yellow
                                                      CV_RGB(255, 0, 0), // Red
                                                      CV_RGB(0, 255, 255)}; // Cyan

    Mat output_image;
    bool wasGoodIterationBefore = false;

    for (int counter = 0; counter < ITERATION_NUMBER; ++counter) {
        //-- Localize the object from img_1 in img_2
        std::vector<Point2f> obj;
        std::vector<Point2f> scene;

        for (size_t i = 0; i < matches.size(); ++i) {
                //-- Get the keypoints from all matches
                obj.push_back(keypoints_object[matches[i].queryIdx].pt);
                scene.push_back(keypoints_scene[matches[i].trainIdx].pt);
        }

        std::vector<Point3f> obj3d;
        for (size_t i = 0; i < obj.size(); ++i) {
            obj3d.push_back(Point3f(obj[i].x, obj[i].y, 0));
        }

        std::vector<Point3f> scene3d;
        for (size_t i = 0; i < scene.size(); ++i) {
            scene3d.push_back(Point3f(scene[i].x, scene[i].y, 0));
        }

        Mat A;
        Mat inliers;

        estimateAffine3D(obj3d, scene3d, A, inliers, THRESHOLD, 0.99);

/*
        // Inlier's mask for drawMatches
        vector<char> mask;
        mask.resize(matches.size());
        for (size_t i = 1; i < inliers.rows; ++i) {
            mask[i] = inliers.at<int>(i, 0);
//            std::cout << inliers.at<int>(i, 0) << " " << mask[i] << "\n";
        }
*/

        std::vector<int> appropriateMatchesIndices;
        std::vector<DMatch> customMatches;
        std::vector<KeyPoint> keypoints_object_matched(keypoints_object.size()), 
                              keypoints_scene_matched(keypoints_scene.size());

        int inliers_count = 0;
        for (size_t i = 1; i < inliers.rows; ++i) {
            if (inliers.at<int>(i, 0) == 1) {
                keypoints_object_matched[matches[i].queryIdx] = keypoints_object[matches[i].queryIdx];
                keypoints_scene_matched[matches[i].trainIdx] = keypoints_scene[matches[i].trainIdx];
                customMatches.push_back(matches[i]);
                appropriateMatchesIndices.push_back(i);
                ++inliers_count;
            }
        }

        Mat A2D(2, 3, CV_64F);
        A2D.at<double>(0, 0) = A.at<double>(0, 0);
        A2D.at<double>(0, 1) = A.at<double>(0, 1);
        A2D.at<double>(0, 2) = A.at<double>(0, 3);
        A2D.at<double>(1, 0) = A.at<double>(1, 0);
        A2D.at<double>(1, 1) = A.at<double>(1, 1);
        A2D.at<double>(1, 2) = A.at<double>(1, 3);

//        std::cout << A2D << "\n";
//        std::cout << "Inliers count: " << inliers_count << "\n";
//        std::cout << "Inliers\n" << inliers << "\n";

        std::vector<Point2f> transformed_obj(obj.size());
        transform(obj, transformed_obj, A2D);
/*
        std::vector<int> appropriateMatchesIndices;
        std::vector<DMatch> customMatches;

        for (size_t index = 0; index < obj.size(); ++index) {
            double distance = sqrt(sqr(scene[index].x - transformed_obj[index].x) +
                                   sqr(scene[index].y - transformed_obj[index].y));

            if (distance < THRESHOLD) {
                customMatches.push_back(matches[index]);
                appropriateMatchesIndices.push_back(index);
            }
        }

        if (customMatches.size() == 0) {
            break;
        }

        std::cout << "Apropriate matches count: " << customMatches.size() << std::endl;
*/

        int flag = DrawMatchesFlags::DRAW_OVER_OUTIMG;
        if (!wasGoodIterationBefore) {
            flag = DrawMatchesFlags::DEFAULT;
        }

        std::vector<Point2f> obj_corners(4);
        std::vector<Point2f> scene_corners(4);

        obj_corners[0] = cvPoint(0, 0); 
        obj_corners[1] = cvPoint(img_object.cols, 0);
        obj_corners[2] = cvPoint(img_object.cols, img_object.rows); 
        obj_corners[3] = cvPoint(0, img_object.rows);

        transform(obj_corners, scene_corners, A2D);

        if (isRectangleOk(A2D)) {
            if (!wasGoodIterationBefore) {
                wasGoodIterationBefore = true;
            }

            Mat iterationOutputImage;

            drawMatches(img_object, keypoints_object_matched, 
                        img_scene, keypoints_scene_matched,
                        customMatches, iterationOutputImage, 
                        colors[counter], Scalar::all(-1),
                        vector<char>(), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

            std::string iterationMatchesOutput;
            iterationMatchesOutput = output_image_name;
            iterationMatchesOutput += "_";
            iterationMatchesOutput += std::to_string(counter);
            iterationMatchesOutput += ".jpg";

            imwrite(iterationMatchesOutput, iterationOutputImage);

            drawMatches(img_object, keypoints_object_matched, 
                        img_scene, keypoints_scene_matched,
                        customMatches, output_image, 
                        colors[counter], Scalar::all(-1),
                        vector<char>(), flag);

            //-- Draw lines between the corners
            line(output_image, 
                 scene_corners[0] + Point2f(img_object.cols, 0), 
                 scene_corners[1] + Point2f(img_object.cols, 0), 
                 colors[counter], 4);
            line(output_image, 
                 scene_corners[1] + Point2f(img_object.cols, 0), 
                 scene_corners[2] + Point2f(img_object.cols, 0), 
                 colors[counter], 4);
            line(output_image, 
                 scene_corners[2] + Point2f(img_object.cols, 0), 
                 scene_corners[3] + Point2f(img_object.cols, 0), 
                 colors[counter], 4);
            line(output_image, 
                 scene_corners[3] + Point2f(img_object.cols, 0), 
                 scene_corners[0] + Point2f(img_object.cols, 0), 
                 colors[counter], 4);
        }

        for (int index = appropriateMatchesIndices.size() - 1; index >= 0; --index) {
            matches.erase(matches.begin() + appropriateMatchesIndices[index]);
        }

//        imshow("Matches", output_image);
//        waitKey(0);
    }

    std::string all_matches_output;
    all_matches_output = output_image_name;
    all_matches_output += ".jpg";
    imwrite(all_matches_output, output_image);

    return 0;
}

/**
 * @function readme
 */
void readme()
{ std::cout << " Usage: ./SURF_Affine <img1> <img2> <output_img>" << std::endl; }
