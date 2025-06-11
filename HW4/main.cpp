//
//  main.cpp
//  HW4
//
//  Created by 이용규 on 6/11/25.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


#include <iostream>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std::chrono_literals;

enum MODE {
    DILATION = 0,
    EROSION  = 255,
};

/** erode or dilation template function.
 
 Assume that the se is 3 x 3 matrix
 
 */
void eodTemplate(const Mat& src, Mat& dst, const Mat& se, const MODE mode) {
    dst = Mat::zeros(src.size(), src.type());
    for (int y = 0; y < src.rows; y++) for (int x = 0; x < src.cols; x++) {
        uchar comp = mode;
        for(int t = -1; t < 2; t++) for(int s = -1; s < 2; s++) {
            if(y+t < 0 || src.rows <= y+t) continue;
            if(x+s < 0 || src.cols <= x+s) continue;
            if(se.at<uchar>(t+1, s+1) == 0) continue;
            
            auto v = src.at<uchar>(y+t, x+s);
            if     (mode == EROSION ) comp = v < comp ? v : comp; // find minimum
            else if(mode == DILATION) comp = v > comp ? v : comp; // find maximum
        }
        dst.at<uchar>(y, x) = comp;
    }
}

/** Gray-scale erosion. */
void erosion(const Mat& src, Mat& dst, const Mat& se) {
    eodTemplate(src, dst, se, MODE::EROSION);
}

/** Gray-scale dilation. */
void dilation(const Mat& src, Mat& dst, const Mat& se) {
    eodTemplate(src, dst, se, MODE::DILATION);
}

void opening(const Mat& src, Mat& dst, const Mat& se) {
    Mat mid;
    eodTemplate(src, mid, se, MODE::EROSION);
    eodTemplate(mid, dst, se, MODE::DILATION);
}

void closing(const Mat& src, Mat& dst, const Mat& se) {
    Mat mid;
    eodTemplate(src, mid, se, MODE::DILATION);
    eodTemplate(mid, dst, se, MODE::EROSION);
}

bool matComp(const Mat& A, const uchar c) {
    for (int y = 0; y < A.rows; y++) for (int x = 0; x < A.cols; x++){
        if(A.at<uchar>(y, x) != c) {
            return false;
        }
    }
    return true;
}


int main() {
    std::this_thread::sleep_for(1000ms);
    
    Mat X = imread("sk2.png", 0);
    X.convertTo(X, CV_8UC1);
    
//    imshow("src", X);
    
    Mat S, Y, Z;
    S = Mat::zeros(X.size(), X.type());
    
    Mat se;
    se = Mat::zeros(3, 3, CV_8UC1);
    //    se.at<uchar>(0, 0) = 1;
    se.at<uchar>(0, 1) = 1;
    //    se.at<uchar>(0, 2) = 1;
    se.at<uchar>(1, 0) = 1;
    se.at<uchar>(1, 1) = 1;
    se.at<uchar>(1, 2) = 1;
    //    se.at<uchar>(2, 0) = 1;
    se.at<uchar>(2, 1) = 1;
    //    se.at<uchar>(2, 2) = 1;
    
    while (true) {
        // erode once
        erosion(X, Y, se);
        // opening once
        opening(Y, Z, se);
        // S = S or (Y-Z)
        S = S | (Y-Z);
        Y.copyTo(X);
        if(matComp(Y, 0)) break;
    }
    
    
    imshow("Skeleton", S);
    imwrite("20212027178_LeeYongKyu_ImageProcessing_HW4.png", S);
    waitKey();
    
    return 0;
}
