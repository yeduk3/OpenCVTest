//
//  main.cpp
//  OpenCVTest
//
//  Created by 이용규 on 5/14/25.
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


void gaussian2DSeperable(const Mat& src, Mat& dst, float sigma_x, float sigma_y) {
    Mat tmp;
    tmp.create(src.size(), src.type());
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
    
    dst.create(tmp.size(), tmp.type());
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

void sort(std::vector<float>& v) {
    for (int i = 0; i < v.size()-1; i++) {
        float minidx = i;
        for(int j = i+1; j < v.size(); j++) {
            if(v[j] < v[minidx]) minidx = j;
        }
        float tmp = v[i];
        v[i] = v[minidx];
        v[minidx] = tmp;
    }
}

void alphaTrimmedMeanFilter(const Mat& src, Mat& dst, int kernelSize, int alpha) {
    assert(kernelSize % 2 == 1 && "kernelSize should be odd.");
    dst.create(src.size(), src.type());
    std::vector<float> values(kernelSize*kernelSize);
    int kd = kernelSize / 2;
    for (int y = 0; y < src.rows; y++) for(int x = 0; x < src.cols; x++) {
        int i = 0;
        for(int t = -kd; t <= kd; t++) for(int s = -kd; s <= kd; s++) {
            int ky = y + t;
            if(ky < 0) ky = -ky;
            else if(ky >= src.rows) ky = 2 * src.rows - ky - 1;
            int kx = x + s;
            if(kx < 0) kx = -kx;
            else if(kx >= src.cols) kx = 2 * src.cols - kx - 1;
            
            values[i++] = src.at<float>(ky, kx);
        }
        
        sort(values);
        float sum = 0.f;
        for(int p = alpha; p < values.size() - alpha; p++) {
            sum += values[p];
        }
        sum /= float(values.size() - 2*alpha);
        dst.at<float>(y, x) = sum;
    }
}

void contraharmonicFilter(const Mat& src, Mat& dst, int kernelSize, float Q) {
    assert(kernelSize % 2 == 1 && "kernelSize should be odd.");
    dst.create(src.size(), src.type());
    int kd = kernelSize / 2;
    for (int y = 0; y < src.rows; y++) for(int x = 0; x < src.cols; x++) {
        float numerator = 0.f, denominator = 0.f;
        for(int t = -kd; t <= kd; t++) for(int s = -kd; s <= kd; s++) {
            int ky = y + t;
            if(ky < 0) ky = -ky;
            else if(ky >= src.rows) ky = 2 * src.rows - ky - 1;
            int kx = x + s;
            if(kx < 0) kx = -kx;
            else if(kx >= src.cols) kx = 2 * src.cols - kx - 1;
            
            float v = src.at<float>(ky, kx);
            float a = pow(v, Q);
            numerator += a * v; // Q+1
            denominator += a;   // Q
        }
        
        if(denominator == 0.f) denominator = 1.f;
        assert(!isnan(numerator / denominator));
        assert(!isinf(numerator / denominator));
        dst.at<float>(y, x) = numerator / denominator;
    }
}

void adaptiveFilter(const Mat& src, Mat& dst, int kernelSize, float sigmaL) {
    assert(kernelSize % 2 == 1 && "kernelSize should be odd.");
    dst.create(src.size(), src.type());
    std::vector<float> values(kernelSize*kernelSize);
    int kd = kernelSize / 2;
    for (int y = 0; y < src.rows; y++) for(int x = 0; x < src.cols; x++) {
        int i = 0;
        for(int t = -kd; t <= kd; t++) for(int s = -kd; s <= kd; s++) {
            int ky = y + t;
            if(ky < 0) ky = -ky;
            else if(ky >= src.rows) ky = 2 * src.rows - ky - 1;
            int kx = x + s;
            if(kx < 0) kx = -kx;
            else if(kx >= src.cols) kx = 2 * src.cols - kx - 1;
            
            values[i++] = src.at<float>(ky, kx);
            
        }
        
        sort(values);
        float v = src.at<float>(y, x);
        float sigma = 0.f;
        float mean = 0.f;
        for(int p = 0; p < values.size(); p++) {
            mean += values[p];
        }
        mean /= values.size();
        for(int p = 0; p < values.size(); p++) {
            sigma += (values[p] - mean) * (values[p] - mean);
        }
        sigma /= values.size();
        
        dst.at<float>(y, x) = v - sigma/sigmaL*(v-mean);
    }
}

void adaptiveMedianFilter(const Mat& src, Mat& dst, const int windowSize = 5, const int maxWindowSize = 9) {
    assert(windowSize % 2 == 1 && "windowSize should be odd.");
    assert(maxWindowSize % 2 == 1 && "maxWindowSize should be odd.");
    assert(maxWindowSize > windowSize && "maxWindowSize should be greater than windowSize.");
    
    dst.create(src.size(), src.type());
    for (int y = 0; y < src.rows; y++) for(int x = 0; x < src.cols; x++) {
//        if(x % 500 == 0) printf("y=%d, x=%d processing\n", y, x);

        std::vector<float> values;
        int ws = windowSize;
        int hws = ws/2;
        float ret = -1;
        // stage b
        while(ret < 0) {
            values.clear();
            hws = windowSize / 2;
            for(int t = -hws; t <= hws; t++) for(int s = -hws; s <= hws; s++) {
                int ky = y + t;
                if(ky < 0) ky = -ky;
                else if(ky >= src.rows) ky = 2 * src.rows - ky - 1;
                int kx = x + s;
                if(kx < 0) kx = -kx;
                else if(kx >= src.cols) kx = 2 * src.cols - kx - 1;
                
                values.push_back(src.at<float>(ky, kx));
                
            }
            
            
            sort(values);
            float median = values[values.size()/2];
            auto a1 = median - values[0];
            auto a2 = median - values[values.size()-1];
            
            if(a1 > 0 && a2 < 0) {
                // stage b
                float z = src.at<float>(y, x);
                auto b1 = z - values[0];
                auto b2 = z - values[values.size()-1];
                if(b1 > 0 && b2 < 0) ret = z;
                else ret = median;
            }
            else ws+=2;
            
            if(ws > maxWindowSize) ret = median;
        }

        dst.at<float>(y, x) = ret;
    }
}

int main(int argc, const char * argv[]) {
    std::this_thread::sleep_for(1000ms);
    
    Mat src = imread("src.png", 0);
    src.convertTo(src, CV_32FC1, 1 / 255.f); // [0, 255] -> [0, 1] in float
    
    Mat gns = imread("gns.png", 0); // 0 makes to read image in gray scale
    gns.convertTo(gns, CV_32FC1, 1 / 255.f); // [0, 255] -> [0, 1] in float
    Mat pns = imread("pns.png", 0);
    pns.convertTo(pns, CV_32FC1, 1 / 255.f); // [0, 255] -> [0, 1] in float
    
    // cv::PSNR(Img 1, Img 2, maximum value)
    printf("GNS's PSNR: %.3f\n", PSNR(src, gns, 1.f));
//    imshow("Gaussian Noise - Before", gns);
    printf("PNS's PSNR: %.3f\n", PSNR(src, pns, 1.f));
//    imshow("Pepper Noise - Before", pns);
    
    // Restore Gaussian Noise. //
    Mat restored_gns;
    restored_gns.create(gns.size(), gns.type());
    gaussian2DSeperable(gns, restored_gns, 1.f, 1.f);
    
    printf("Restored GNS's PSNR: %.3f\n", PSNR(src, restored_gns, 1.f));
    imshow("Gaussian Noise - After", restored_gns);
    
    // Restore Pepper Noise. //
    Mat restored_pns;
    restored_pns.create(pns.size(), pns.type());
    
//    medianBlur(pns, restored_pns, 5);
//    alphaTrimmedMeanFilter(pns, restored_pns, 5, 10);
//    contraharmonicFilter(pns, restored_pns, 3, 1.5f);
//    adaptiveFilter(pns, restored_pns, 5, 6e-2);
    adaptiveMedianFilter(pns, restored_pns);
    
    
    printf("Restored PNS's PSNR: %.3f\n", PSNR(src, restored_pns, 1.f));
    imshow("Pepper Noise - After", restored_pns);
    
    waitKey();
    
    return 0;
}
