//
//  main.cpp
//  OpenCVTest
//
//  Created by 이용규 on 4/22/25.
//


#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>



using namespace cv;

int main() {
    Mat src = imread("test.jpeg", 0);
    src.convertTo(src, CV_32FC1, 1/255.f);
    
    Mat chan[2], dst;
    dst.convertTo(dst, CV_32FC2);
    dft(src, dst, DFT_COMPLEX_OUTPUT);
    
    split(dst, chan);
    
    Mat mag;
    magnitude(chan[0], chan[1], mag);
    
    
    imshow("DFT", mag / 1000);
    
    dst.at<Vec2f>(0, 0) *= 0;
    
    Mat rec;
    idft(dst, rec, DFT_SCALE | DFT_REAL_OUTPUT);
    imshow("Rec", rec);
    
    waitKey();
    
    return 0;
    
}
