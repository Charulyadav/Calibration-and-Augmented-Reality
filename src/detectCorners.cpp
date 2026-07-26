/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 4: Calibration and Augmented Reality

  Task 1: Detect and Extract Target Corners
  Task 2: Select Calibration Images
  Task 3: Calibrate the Camera
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

int main(int argc, char *argv[])
{
    cv::VideoCapture capdev(0); // change 0 to 1 or 2 for different device usage
    if (!capdev.isOpened())
    {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    // the checkerboard has 9 columns and 6 rows of internal corners
    // internal corners = where 4 squares meet, not the outer edge
    cv::Size patternSize(9, 6);

    cv::namedWindow("Corners", cv::WINDOW_AUTOSIZE);
    std::cout << "Looking for a " << patternSize.width << "x" << patternSize.height
              << " internal-corner checkerboard.\n"
              << "Controls: q=quit, s=save frame\n"
              << "  c - calibrate (needs >= 5 saved images)\n"
              << "  w - write intrinsics to intrinsics.yml\n";

    cv::Mat frame, gray;
    std::vector<cv::Point2f> cornerSet; // detected corner locations
    int saveCounter = 0;

    // storage for calibration data
    std::vector<cv::Vec3f> pointSet;                  // 3D world coords (same every time)
    std::vector<std::vector<cv::Vec3f>> pointList;    // one pointSet per saved image
    std::vector<std::vector<cv::Point2f>> cornerList; // one cornerSet per saved image

    // build the 3D world point set once - it's identical for every image.
    // order must match findChessboardCorners output: row by row, left to right, top-left origin, Y going down as negative (Z toward viewer).
    for (int row = 0; row < patternSize.height; row++)
    { // 6 rows
        for (int col = 0; col < patternSize.width; col++)
        { // 9 cols
            pointSet.push_back(cv::Vec3f((float)col, (float)-row, 0.0f));
        }
    }

    // track the most recent successful detection so 's' saves a valid frame
    std::vector<cv::Point2f> lastCorners;
    cv::Mat lastFrame;
    bool haveValidDetection = false;

    // calibration outputs
    cv::Mat cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1); // 3x3 identity to start
    cameraMatrix.at<double>(0, 2) = frame.cols / 2.0;    // u0 = image center x
    cameraMatrix.at<double>(1, 2) = frame.rows / 2.0;    // v0 = image center y

    std::vector<double> distCoeffs; // we'll size it at calibration time
    bool calibrated = false;

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
        {
            std::cout << "Frame is empty\n";
            break;
        }

        // findChessboardCorners works on the color or gray image but cornerSubPix needs grayscale, so convert once here
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        // try to find the checkerboard corners
        // the flags help with robustness:
        // ADAPTIVE_THRESH - handles uneven lighting
        // NORMALIZE_IMAGE - normalizes brightness before thresholding
        // FAST_CHECK      - quick early-out when no board is present
        bool found = cv::findChessboardCorners(
            gray, patternSize, cornerSet,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                cv::CALIB_CB_FAST_CHECK);

        if (found)
        {
            // refine corner locations to sub-pixel accuracy
            // this dramatically improves calibration quality later
            cv::cornerSubPix(
                gray, cornerSet,
                cv::Size(11, 11), // search window half-size
                cv::Size(-1, -1), // no dead zone
                cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

            // draw the detected corners onto the frame
            cv::drawChessboardCorners(frame, patternSize, cornerSet, found);

            // remember this successful detection in case the user presses 's'
            lastCorners = cornerSet;
            lastFrame = frame.clone();
            haveValidDetection = true;
        }
        else
        {
            haveValidDetection = false;
        }

        cv::imshow("Corners", frame);

        int key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
        else if (key == 's')
        {
            if (haveValidDetection)
            {
                // save the 2D corners and the (constant) 3D world points as a matched pair
                cornerList.push_back(lastCorners);
                pointList.push_back(pointSet);

                std::string fname = "calib_" + std::to_string(cornerList.size()) + ".jpg";
                cv::imwrite(fname, lastFrame);

                std::cout << "Saved calibration image #" << cornerList.size()
                          << " (" << lastCorners.size() << " corners). "
                          << "Saved " << fname << "\n";
            }
            else
            {
                std::cout << "No valid checkerboard detected - not saving.\n";
            }
        }
        else if (key == 'c')
        {
            if (cornerList.size() < 5)
            {
                std::cout << "Need at least 5 calibration images. Have "
                          << cornerList.size() << ".\n";
            }
            else
            {
                // let OpenCV estimate the camera matrix from scratch
                cameraMatrix = cv::Mat::eye(3, 3, CV_64FC1);
                distCoeffs = std::vector<double>(5, 0.0);

                std::vector<cv::Mat> rvecs, tvecs; // per-image rotations and translations

                std::cout << "\n*** BEFORE calibration ***\n";
                std::cout << "Camera matrix:\n"
                          << cameraMatrix << "\n";

                // run calibration
                // CALIB_FIX_ASPECT_RATIO forces fx == fy (square pixels)
                double rms = cv::calibrateCamera(
                    pointList, cornerList, frame.size(),
                    cameraMatrix, distCoeffs, rvecs, tvecs,
                    cv::CALIB_FIX_ASPECT_RATIO);

                std::cout << "\n*** AFTER calibration ***\n";
                std::cout << "Camera matrix:\n"
                          << cameraMatrix << "\n";
                std::cout << "Distortion: [";
                for (double d : distCoeffs)
                    std::cout << d << " ";
                std::cout << "]\n";
                std::cout << "Reprojection error (RMS): " << rms << " pixels\n";
                std::cout << "(Used " << cornerList.size() << " images)\n\n";

                calibrated = true;
            }
        }
        else if (key == 'w')
        {
            if (!calibrated)
            {
                std::cout << "Calibrate first (press 'c') before saving.\n";
            }
            else
            {
                // use OpenCV's FileStorage to write a clean file
                cv::FileStorage fs("intrinsics.yml", cv::FileStorage::WRITE);
                fs << "camera_matrix" << cameraMatrix;
                fs << "distortion_coefficients" << distCoeffs;
                fs.release();
                std::cout << "Wrote camera_matrix and distortion_coefficients to intrinsics.yml\n";
            }
        }
    }

    cv::destroyAllWindows();
    return 0;
}