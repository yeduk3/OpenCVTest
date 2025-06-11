//
//  main.cpp
//  OpenCVTest
//
//  Created by 이용규 on 4/2/25.
//

#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>


#include <iostream>

using namespace cv;

void gaussian2DSeperable(const Mat& src, Mat& dst, float sigma_x, float sigma_y) {
    Mat tmp;
    tmp.create(src.size(), CV_32FC1);
    int K_x = int(ceil((sigma_x-0.8)/0.3 +1));
    sigma_x *= sigma_x;
    if(sigma_x == 0) src.copyTo(tmp);
    else for (int y = 0; y < src.rows; y++) {
        for(int x = 0; x < src.cols; x++) {
            float wsum = 0;
            float sum = 0;
            for (int s = -K_x; s <= K_x; s++) {
                float w = exp(-s * s / sigma_x);
                wsum += w;
                
                int xx = min(max(x + s, 0), src.cols-1);

                sum += src.at<float>(y, xx) * w;
            }
            sum /= wsum;
            tmp.at<float>(y, x) = sum;
        }
    }
    
    dst.create(tmp.size(), CV_32FC1);
    int K_y = int(ceil((sigma_y-0.8)/0.3 +1));
    sigma_y *= sigma_y;
    if (sigma_y == 0) tmp.copyTo(dst);
    else for (int y = 0; y < tmp.rows; y++) {
        for(int x = 0; x < tmp.cols; x++) {
            float wsum = 0;
            float sum = 0;
            for (int t = -K_y; t <= K_y; t++) {
                float w = exp(-t * t / sigma_y);
                wsum += w;
                
                int yy = min(max(y + t, 0), tmp.rows-1);
                
                sum += tmp.at<float>(yy, x) * w;
            }
            sum /= wsum;
            dst.at<float>(y, x) = sum;
        }
    }
}

int main(int argc, const char * argv[]) {
    Mat src = imread("test.jpeg", 0); // 0 makes to read image in gray scale
    src.convertTo(src, CV_32FC1, 1 / 255.f); // [0, 255] -> [0, 1] in float

    // My Gaussian Filter
    Mat dst;
    gaussian2DSeperable(src, dst, 7, 7);
    
    // Built-in Gaussian Filter
//    Mat cvDst;
//    GaussianBlur(src, cvDst, Size(), 7, 7, BORDER_REPLICATE);
    
    imshow("OpenCVTest My Gaussian", dst);
//    imshow("OpenCVTest Gaussian Blur", cvDst);
    
    waitKey();
    
    return 0;
}
