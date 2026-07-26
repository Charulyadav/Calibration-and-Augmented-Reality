/*
  Charul - 002535330
  Spring 2026
  CS 5330 Computer Vision
  Project 4: Calibration and Augmented Reality

  Task 4: Calculate Current Position of the Camera
  Task 5: Project Outside Corners or 3D Axes
  Task 6: Create a Virtual Object
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>

/*
  draws a virtual jug floating above the board, built from 3D line segments. the jug is defined in world coordinates (units = board squares), projected
  into the image with the current pose and drawn as colored lines. uses multiple colors: body (cyan), spout (red), handle (magenta) so the orientation
  is obvious as the camera moves. the spout makes it asymmetric.

  arguments:
    frame        - image to draw on
    rvec, tvec   - board pose from solvePnP
    cameraMatrix, distCoeffs - camera intrinsics
*/
void drawJug(cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec,
             const cv::Mat &cameraMatrix, const std::vector<double> &distCoeffs)
{
    const int SIDES = 12;              // polygon sides approximating circles
    const float cx = 4.0f, cy = -2.5f; // center the jug over the board middle
    const float bodyR = 1.5f;          // body radius (squares)
    const float neckR = 0.9f;          // neck radius
    const float zBase = 1.0f;          // bottom of jug floats 1 square above board
    const float zBodyTop = 3.0f;       // top of the body
    const float zNeck = 4.0f;          // top of the neck

    // helper to make a circle of 3D points at a given height and radius
    auto makeRing = [&](float z, float r)
    {
        std::vector<cv::Vec3f> ring;
        for (int i = 0; i < SIDES; i++)
        {
            float a = (float)(2.0 * CV_PI * i / SIDES);
            ring.push_back(cv::Vec3f(cx + r * cos(a), cy + r * sin(a), z));
        }
        return ring;
    };

    // build the four rings that define the jug's silhouette
    std::vector<cv::Vec3f> bottom = makeRing(zBase, bodyR);
    std::vector<cv::Vec3f> bodyTop = makeRing(zBodyTop, bodyR);
    std::vector<cv::Vec3f> neckBottom = makeRing(zBodyTop, neckR);
    std::vector<cv::Vec3f> neckTop = makeRing(zNeck, neckR);

    // helper: project a set of 3D points and return their 2D image locations
    auto project = [&](const std::vector<cv::Vec3f> &pts)
    {
        std::vector<cv::Point2f> img;
        cv::projectPoints(pts, rvec, tvec, cameraMatrix, distCoeffs, img);
        return img;
    };

    std::vector<cv::Point2f> pBottom = project(bottom);
    std::vector<cv::Point2f> pBodyTop = project(bodyTop);
    std::vector<cv::Point2f> pNeckBottom = project(neckBottom);
    std::vector<cv::Point2f> pNeckTop = project(neckTop);

    cv::Scalar bodyColor(255, 255, 0);   // cyan
    cv::Scalar spoutColor(0, 0, 255);    // red
    cv::Scalar handleColor(255, 0, 255); // magenta

    // draw the rings (connect consecutive points around each circle)
    auto drawRing = [&](const std::vector<cv::Point2f> &ring, cv::Scalar color)
    {
        for (int i = 0; i < SIDES; i++)
            cv::line(frame, ring[i], ring[(i + 1) % SIDES], color, 2);
    };
    drawRing(pBottom, bodyColor);
    drawRing(pBodyTop, bodyColor);
    drawRing(pNeckTop, bodyColor);

    // vertical lines connecting bottom -> bodyTop (the body walls)
    for (int i = 0; i < SIDES; i++)
        cv::line(frame, pBottom[i], pBodyTop[i], bodyColor, 2);
    // Neck walls: bodyTop(neck radius) -> neckTop
    for (int i = 0; i < SIDES; i++)
        cv::line(frame, pNeckBottom[i], pNeckTop[i], bodyColor, 2);

    // spout: a triangular lip sticking out one side of the neck
    std::vector<cv::Vec3f> spout3d = {
        cv::Vec3f(cx + neckR, cy, zNeck),               // neck rim, +X side
        cv::Vec3f(cx + neckR + 1.0f, cy, zNeck + 0.6f), // spout tip, out and up
        cv::Vec3f(cx + neckR, cy + 0.5f, zNeck),        // rim point 2
        cv::Vec3f(cx + neckR, cy - 0.5f, zNeck),        // rim point 3
    };
    std::vector<cv::Point2f> pSpout = project(spout3d);
    cv::line(frame, pSpout[0], pSpout[1], spoutColor, 2);
    cv::line(frame, pSpout[2], pSpout[1], spoutColor, 2);
    cv::line(frame, pSpout[3], pSpout[1], spoutColor, 2);

    // handle: an arc of line segments on the opposite side (-X)
    std::vector<cv::Vec3f> handle3d;
    int HSEG = 8;
    for (int i = 0; i <= HSEG; i++)
    {
        float t = (float)i / HSEG;                            // 0..1 along the handle
        float z = zBase + 0.5f + t * (zNeck - zBase - 0.5f);  // rise up the side
        float xout = -bodyR - 0.8f * sin((float)(CV_PI * t)); // bow outward
        handle3d.push_back(cv::Vec3f(cx + xout, cy, z));
    }
    std::vector<cv::Point2f> pHandle = project(handle3d);
    for (int i = 0; i < HSEG; i++)
        cv::line(frame, pHandle[i], pHandle[i + 1], handleColor, 2);
}

/*
  hides the checkerboard by filling its quadrilateral in the image. the fill color is sampled from a point just outside the board so it
  blends with the surrounding surface, making the board "disappear." must be called BEFORE drawing the virtual object, so the object sits
  on top of the clean fill rather than behind it.

  arguments:
    frame        - image to modify
    rvec, tvec   - board pose
    cameraMatrix, distCoeffs - intrinsics
    patternSize  - checkerboard internal corner dimensions
*/
void hideTarget(cv::Mat &frame, const cv::Mat &rvec, const cv::Mat &tvec,
                const cv::Mat &cameraMatrix, const std::vector<double> &distCoeffs,
                cv::Size patternSize)
{
    // define the board's outer boundary in 3D, slightly larger than the internal corners so we cover the whole printed pattern including its border.
    int w = patternSize.width - 1;  // 8 (0..8 internal corners span)
    int h = patternSize.height - 1; // 5
    float pad = 1.0f;               // extend 1 square past the corners

    std::vector<cv::Vec3f> boardQuad = {
        cv::Vec3f(-pad, pad, 0),           // top-left
        cv::Vec3f(w + pad, pad, 0),        // top-right
        cv::Vec3f(w + pad, -(h + pad), 0), // bottom-right
        cv::Vec3f(-pad, -(h + pad), 0),    // bottom-left
    };

    std::vector<cv::Point2f> quad2f;
    cv::projectPoints(boardQuad, rvec, tvec, cameraMatrix, distCoeffs, quad2f);

    // convert to integer points for fillConvexPoly
    std::vector<cv::Point> quad;
    for (const auto &p : quad2f)
        quad.push_back(cv::Point((int)p.x, (int)p.y));

    // sample a fill color from just outside the board (above the top edge).
    // use the midpoint of the top edge, pushed further up in the image.
    cv::Point samplePt((quad[0].x + quad[1].x) / 2,
                       (quad[0].y + quad[1].y) / 2 - 40);
    cv::Scalar fillColor(60, 60, 60); // fallback gray
    if (samplePt.x >= 0 && samplePt.x < frame.cols &&
        samplePt.y >= 0 && samplePt.y < frame.rows)
    {
        cv::Vec3b s = frame.at<cv::Vec3b>(samplePt);
        fillColor = cv::Scalar(s[0], s[1], s[2]);
    }

    // fill the board quadrilateral with the sampled color
    cv::fillConvexPoly(frame, quad, fillColor);
}

int main(int argc, char *argv[])
{
    // read calibration from the file written in Task 3
    cv::Mat cameraMatrix;
    std::vector<double> distCoeffs;
    cv::FileStorage fs("intrinsics.yml", cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Could not open intrinsics.yml. Run calibration first.\n";
        return -1;
    }
    fs["camera_matrix"] >> cameraMatrix;
    fs["distortion_coefficients"] >> distCoeffs;
    fs.release();

    std::cout << "Loaded camera matrix:\n"
              << cameraMatrix << "\n";

    // build the 3D world point set
    cv::Size patternSize(9, 6);
    std::vector<cv::Vec3f> pointSet;
    for (int row = 0; row < patternSize.height; row++)
    {
        for (int col = 0; col < patternSize.width; col++)
        {
            pointSet.push_back(cv::Vec3f((float)col, (float)-row, 0.0f));
        }
    }

    cv::VideoCapture capdev(0);
    if (!capdev.isOpened())
    {
        std::cout << "Unable to open video device\n";
        return -1;
    }

    cv::namedWindow("Pose", cv::WINDOW_AUTOSIZE);
    std::cout << "Controls: q=quit\n";

    cv::Mat frame, gray;
    std::vector<cv::Point2f> cornerSet;

    bool hideMode = true;

    for (;;)
    {
        capdev >> frame;
        if (frame.empty())
            break;

        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);

        bool found = cv::findChessboardCorners(
            gray, patternSize, cornerSet,
            cv::CALIB_CB_ADAPTIVE_THRESH | cv::CALIB_CB_NORMALIZE_IMAGE |
                cv::CALIB_CB_FAST_CHECK);

        if (found)
        {
            cv::cornerSubPix(gray, cornerSet, cv::Size(11, 11), cv::Size(-1, -1),
                             cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1));

            // solve for the board's pose relative to the camera
            cv::Mat rvec, tvec;
            bool ok = cv::solvePnP(pointSet, cornerSet, cameraMatrix, distCoeffs,
                                   rvec, tvec);

            if (ok)
            {
                // print pose
                std::cout << "rvec: [" << rvec.at<double>(0) << ", "
                          << rvec.at<double>(1) << ", " << rvec.at<double>(2) << "]  "
                          << "tvec: [" << tvec.at<double>(0) << ", "
                          << tvec.at<double>(1) << ", " << tvec.at<double>(2) << "]\n";

                // draw 3D axes attached to the board origin
                // define the 4 points: origin + tip of each axis (length 2 squares)
                std::vector<cv::Vec3f> axisPoints;
                axisPoints.push_back(cv::Vec3f(0, 0, 0));  // origin
                axisPoints.push_back(cv::Vec3f(2, 0, 0));  // X tip
                axisPoints.push_back(cv::Vec3f(0, -2, 0)); // Y tip (negative = down in our setup)
                axisPoints.push_back(cv::Vec3f(0, 0, 2));  // Z tip (out of the board)

                // project those 3D points into the 2D image using the pose + intrinsics
                std::vector<cv::Point2f> imagePoints;
                cv::projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs, imagePoints);

                // draw the three axis lines from the origin
                cv::line(frame, imagePoints[0], imagePoints[1], cv::Scalar(0, 0, 255), 3); // X = red
                cv::line(frame, imagePoints[0], imagePoints[2], cv::Scalar(0, 255, 0), 3); // Y = green
                cv::line(frame, imagePoints[0], imagePoints[3], cv::Scalar(255, 0, 0), 3); // Z = blue

                // also draw the four outer corners of the board as circles
                std::vector<cv::Vec3f> boardCorners;
                boardCorners.push_back(cv::Vec3f(0, 0, 0));  // top-left
                boardCorners.push_back(cv::Vec3f(8, 0, 0));  // top-right
                boardCorners.push_back(cv::Vec3f(8, -5, 0)); // bottom-right
                boardCorners.push_back(cv::Vec3f(0, -5, 0)); // bottom-left
                std::vector<cv::Point2f> cornerImagePoints;
                cv::projectPoints(boardCorners, rvec, tvec, cameraMatrix, distCoeffs, cornerImagePoints);
                for (const auto &pt : cornerImagePoints)
                {
                    cv::circle(frame, pt, 6, cv::Scalar(0, 255, 255), -1); // yellow dots
                }

                // keep the corner grid too
                // cv::drawChessboardCorners(frame, patternSize, cornerSet, found);

                if (hideMode)
                {
                    hideTarget(frame, rvec, tvec, cameraMatrix, distCoeffs, patternSize);
                }

                // draw the virtual jug
                drawJug(frame, rvec, tvec, cameraMatrix, distCoeffs);
            }
        }

        cv::imshow("Pose", frame);
        int key = cv::waitKey(10);
        if (key == 'q')
        {
            break;
        }
        else if (key == 'h')
        {
            hideMode = !hideMode;
            std::cout << "Hide target: " << (hideMode ? "ON" : "OFF") << "\n";
        }
    }

    cv::destroyAllWindows();
    return 0;
}