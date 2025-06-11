//
//  main.cpp
//  OpenCVTest
//
//  Created by 이용규 on 5/28/25.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <complex>
#include <iostream>
#include <chrono>
#include <thread>

using namespace cv;
using namespace std::chrono_literals;


/** Returns motion blurring psf in frequency domain.
 
 - Parameters:
    - parameter F: Blur target image.
    - parameter a: Horizontal movement
    - parameter b: Vertical movement
    - parameter T: Period. 1 is best.
 
 
 */
Mat motionBlurDeg(const Mat& target, const float a, const float b, const float T = 1) {
    Mat H(target.size(), CV_32FC2);
    
    const float PI = CV_PI;
    for (int v = 0; v < H.rows; v++) for(int u = 0; u < H.cols; u++) {
        float uu = u,vv = v;
        if(H.cols < 2*u) uu = H.cols - u;
        if(H.rows < 2*v) vv = H.rows - v;
        float puavb = PI*(uu*a+vv*b);
        if(puavb < 1e-10) {
            H.at<Vec2f>(v,u) = Vec2f(1.f, 0.f);
            continue;
        }
        
        float val = T/puavb;
        val *= sinf(puavb);
        std::complex<float> i(0.f, 1.f);
        auto valc = val * std::exp(-i*puavb);
        H.at<Vec2f>(v, u) = {valc.real(), valc.imag()};
    }
    
    // Preserve DC
    H.at<Vec2f>(0,0) = {1.f, 1.f};
    
    return H;
}

int main(int argc, const char * argv[]) {
    std::this_thread::sleep_for(1000ms);
    
    // Read image.
    Mat src = imread("src.png", 0);
    src.convertTo(src, CV_32FC1, 1 / 255.f); // [0, 255] -> [0, 1] in float
    imshow("src", src);
    Mat F;
    dft(src, F, DFT_COMPLEX_OUTPUT);
    
    // Get blur psf for the image.
    Mat H;
    float a, b;
    a = 0.006f;
    b = 0.009f;
    float T = 1.f;
    H = motionBlurDeg(F, a, b, T);
    
    // Blur psf's mag
    Mat chan[2], mag;
    split(H, chan);
    magnitude(chan[0], chan[1], mag);
    imshow("DFT", mag);
    
    // Blurred image.
    Mat G;
    mulSpectrums(F, H, G, false);
    
    Mat g;
    idft(G, g, DFT_REAL_OUTPUT | DFT_SCALE);
    
    imshow("Motion Blur", g);
    waitKey();
    
    return 0;
}
