/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 4: Calibration and Augmented Reality

  Task 7: Detect robust features
*/

#include <opencv2/opencv.hpp>
#include <iostream>

int main(int argc, char *argv[])
{
    cv::VideoCapture capdev(0);
    if (!capdev.isOpened())
    {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    cv::namedWindow("Harris Features", cv::WINDOW_AUTOSIZE);

    // trackbar-controlled threshold to experiment live.
    // higher threshold = fewer, stronger corners.
    int threshSlider = 130; // 0..255, maps to the harris response cutoff
    cv::createTrackbar("Threshold", "Harris Features", &threshSlider, 255, nullptr);

    std::cout << "Controls: q=quit, s=save. Adjust the Threshold slider.\n";

    cv::Mat frame, gray, harris, harrisNorm;
    int saveCounter = 0;

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // harris corner response
        // blockSize=2 (neighborhood), ksize=3 (Sobel aperture), k=0.04 (harris param)
        harris = cv::Mat::zeros(frame.size(), CV_32FC1);
        cv::cornerHarris(gray, harris, 2, 3, 0.04);

        // normalize the response to 0..255 so the slider threshold is meaningful
        cv::normalize(harris, harrisNorm, 0, 255, cv::NORM_MINMAX, CV_32FC1);

        // draw a circle wherever the response exceeds the threshold
        int cornerCount = 0;
        for (int r = 0; r < harrisNorm.rows; r++)
        {
            const float *hrow = harrisNorm.ptr<float>(r);
            for (int c = 0; c < harrisNorm.cols; c++)
            {
                if (hrow[c] > threshSlider)
                {
                    cv::circle(frame, cv::Point(c, r), 4, cv::Scalar(0, 255, 0), 1);
                    cornerCount++;
                }
            }
        }

        // show the count on the frame
        cv::putText(frame, "Corners: " + std::to_string(cornerCount),
                    cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 0.7,
                    cv::Scalar(0, 255, 255), 2);

        cv::imshow("Harris Features", frame);

        int key = cv::waitKey(10);
        if (key == 'q')
            break;
        else if (key == 's')
        {
            std::string fname = "harris_" + std::to_string(saveCounter++) + ".jpg";
            cv::imwrite(fname, frame);
            std::cout << "Saved " << fname << "\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}