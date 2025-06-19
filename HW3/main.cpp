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
    imshow(name, mag / 300.0);
}


int main(int argc, const char * argv[]) {
    std::this_thread::sleep_for(1000ms);
    
    Mat deg = imread("deg.png", 0);
    deg.convertTo(deg, CV_32F, 1 / 255.f); // [0, 255] -> [0, 1] in float
    
    Mat G;
    dft(deg, G, DFT_COMPLEX_OUTPUT);
    
    showmag("test", deg);
    
    Mat ker = imread("ker.png", 0);
    ker.convertTo(ker, CV_32F, 1 / 255.f);
    ker/=sum(ker)[0];
    Mat H;
    dft(ker, H, DFT_COMPLEX_OUTPUT);
    
    // wiener filter
    const float K = 0.01;
    Mat H2, GdivH, Kterm, F;
    mulSpectrums(H, H, H2, true);
    divSpectrums(G, H, GdivH, false);
    divSpectrums(H2, H2+K, Kterm, false);
    mulSpectrums(GdivH, Kterm, F, false);
    
    Mat rec;
    idft(F, rec, DFT_REAL_OUTPUT | DFT_SCALE);
    
    // https://www.aladin.co.kr/shop/wproduct.aspx?ItemId=12637898
    rec.convertTo(rec, CV_8U, 255);
    imshow("THE KALEVALA", rec);
//    imwrite("iphw3_202127178_leeyongkyu_result.png", rec);
    
    waitKey();
    
    return 0;
}
