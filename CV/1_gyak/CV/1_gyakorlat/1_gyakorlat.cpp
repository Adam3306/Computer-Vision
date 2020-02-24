#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>

using namespace cv;

#define HEIGHT 600
#define WIDTH 800


Mat image;
int iksz, ipszilon;
int ball_X , ball_Y;

void redraw()
{
    rectangle(image, Point(0, 0), Point(WIDTH, HEIGHT), Scalar(0, 0, 0), cv::FILLED);

    rectangle(image, Point(0, 0), Point(10, HEIGHT+10), Scalar(0, 255, 0), cv::FILLED);
    rectangle(image, Point(0, 0), Point(WIDTH+ 10, 10), Scalar(0, 255, 0), cv::FILLED);
    rectangle(image, Point(WIDTH - 10, 0), Point(WIDTH, HEIGHT), Scalar(0, 255, 0), cv::FILLED);
    circle(image, Point(ball_X, ball_Y), 4, Scalar(0, 0, 255), cv::FILLED, 8);

    rectangle(image, Point(iksz, HEIGHT - 10), Point(iksz + 80, HEIGHT), Scalar(255, 0, 0), cv::FILLED);      // bgr
    imshow("Display window", image);                                                                    // Show our image inside it.
}

void MouseCallBackFunc(int event, int x, int y, int flags, void* userdata)
{
    if (event == EVENT_LBUTTONDOWN)
    {
        iksz = x;
        redraw();
    }

}

int main(int argc, char** argv)
{
    image = Mat::zeros(HEIGHT, WIDTH, CV_8UC3);                          // CV_8UC3 - 8 bit - UNSIGNED - C3 (3 csatorna)
    iksz = WIDTH / 2 - 40;
    ipszilon = -100.0;
    // ball_X = 300;
    // ball_Y = 400;
    namedWindow("Display window", WINDOW_AUTOSIZE);                     // Create a window for display.
    setMouseCallback("Display window", MouseCallBackFunc, NULL);

    imshow("Display window", image);                                    // Show our image inside it.

    int key;
    while (true)
    {
        key = waitKey(100);
        ball_X = -10;
        ball_Y = -10;
        if (key == 27) break;

        switch (key)
        {
            case 'w':
                ipszilon -= 10;
                break;
            case 'a':
                if (20 <= iksz)
                {
                    iksz-=10;
                }
                break;
            case 's':
                ipszilon += 10;
                break;
            case 'd':
                if (iksz <= 700)
                {
                    iksz += 10;
                }
                break;     
        }
        redraw();
    }


    return 0;
}