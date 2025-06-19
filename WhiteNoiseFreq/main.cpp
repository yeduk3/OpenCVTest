//
//  main.cpp
//  OpenCVTest
//
//  Created by 이용규 on 6/17/25.
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

void showmag(const char* name, const Mat& img) {
    Mat f;
    img.copyTo(f);
    
    Mat F;
    dft(f, F, DFT_COMPLEX_OUTPUT);
    
    Mat chan[2];
    split(F, chan);
    Mat mag;
    magnitude(chan[0], chan[1], mag);
    imshow(name, mag / 500.0);
}

int main(int argc, const char * argv[]) {
    std::this_thread::sleep_for(1000ms);
    
    Mat src, pns;
    src = imread("src.png", 0);
    pns = imread("pns.png", 0);
    src.convertTo(src, CV_32FC1, 1 / 255.f);
    pns.convertTo(pns, CV_32FC1, 1 / 255.f);
    
    showmag("src", src);
    showmag("pns", pns);
    
    waitKey();
    
    
    return 0;
}
