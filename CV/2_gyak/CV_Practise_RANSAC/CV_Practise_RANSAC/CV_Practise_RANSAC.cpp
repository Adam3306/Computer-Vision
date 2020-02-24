// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <time.h>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>

using namespace cv;
using namespace std;

void GenerateData(vector<Point2d> &points, double noise, int pointNumber, int outlierNumber, Size size);
void DrawPoints(vector<Point2d> &points, Mat image);
void FitLineRANSAC(const vector<Point2d> * const points, vector<int> &inliers, Mat &line, double threshold, int iteration_number, Mat image);
void FitLineLSQ(const vector<Point2d> * const points, vector<int> &inliers, Mat &line);


int _tmain(int argc, _TCHAR* argv[])
{
	vector<Point2d> points;
	Mat image = Mat::zeros(600, 600, CV_8UC3);

	GenerateData(points, 5, 100, 50, Size(image.cols, image.rows));

	DrawPoints(points, image);

	imshow("Image", image);
	waitKey(0);

	vector<int> inliers;
	Mat bestLine;
	FitLineRANSAC(&points, inliers, bestLine, 2.0, 1000, image);

	for (const int& idx : inliers)
	{
		cv::circle(image, points[idx], 2, cv::Scalar(0, 255, 0), -1);
	}

	FitLineLSQ(&points, inliers, bestLine);

	imshow("Final result", image);
	waitKey(0);

	return 0;
}

// Draw points to the image
void DrawPoints(vector<Point2d> &points, Mat image)
{
	for (int i = 0; i < points.size(); ++i)
	{
		circle(image, points[i], 2, Scalar(255, 255, 255));
	}
}

// Generate a synthetic line and sample that. Then add outliers to the data.
void GenerateData(vector<Point2d> &points,  double noise, int pointNumber, int outlierNumber, Size size)
{
	srand(time(NULL));
	// Generate random line by its normal direction and a center
	Point2d center;
	center.x = (static_cast<double>(rand()) / RAND_MAX) * size.width;
	center.y = (static_cast<double>(rand()) / RAND_MAX) * size.height;

	Point2d tangent;
	tangent.x = (static_cast<double>(rand()) / RAND_MAX);
	tangent.y = (static_cast<double>(rand()) / RAND_MAX);
	tangent = tangent / cv::norm(tangent);

	// Generate random points on that line
	points.resize(pointNumber + outlierNumber);
	double t;

	for (int i = 0; i < pointNumber; ++i)
	{
		t = (static_cast<double>(rand()) / RAND_MAX)* size.width;
		points[i] = center + t * tangent;

		points[i].x += (static_cast<double>(rand()) / RAND_MAX) * noise - noise / 2.0;
		points[i].y -= (static_cast<double>(rand()) / RAND_MAX) * noise - noise / 2.0;
	}


	// Add outliers
	for (int i = 0; i < outlierNumber; ++i)
	{
		points[i + pointNumber].x = (static_cast<double>(rand()) / RAND_MAX)* size.width;
		points[i + pointNumber].y = (static_cast<double>(rand()) / RAND_MAX)* size.height;
	}
}

// Apply RANSAC to fit points to a 2D line
void FitLineRANSAC(const vector<Point2d> * const points, vector<int> &inliers, Mat &line, double threshold, int iteration_number, Mat image)
{
	int it = 0;
	int bestInlierNumber = 0;
	vector<int> bestInliers;
	Mat bestLine;
	Point2d bestPt1, bestPt2;

	// Select MSS
	vector<int> indices(points->size());
	for (int i = 0; i < indices.size(); ++i)
	{
		indices[i] = i;
	}

	while (it++ < iteration_number)
	{
		Mat img = image.clone();

		vector<int> mss(2);
		for (int i = 0; i < mss.size(); ++i)
		{
			size_t idx = (static_cast<double>(rand()) / RAND_MAX) * (indices.size() - 1);
			mss[i] = indices[idx];
			indices.erase(indices.begin() + idx);
		}


		// Calculate line parameters
		double a = 0, b = 0, c = 0;
		
		Point2d o = points->at(mss[0]);
		Point2d v = points->at(mss[1]) - points->at(mss[0]);

		// Find the normal
		Point2d n;
		n.x = -v.y;
		n.y = v.x;
		n = n / cv::norm(n);

		a = n.x;
		b = n.y;
		c = -o.x * a - o.y * b;

		// Get the inliers
		int inlierNumber = 0;
		vector<int> inliers;
		for (int i = 0; i < points->size(); ++i)
		{
			const Point2d& point = points->at(i);
			// Calc the distance
			const double signed_distance = a * point.x + b * point.y + c;
			const double distance = std::abs(signed_distance);

			if (distance < threshold)
			{
				++inlierNumber;
				inliers.emplace_back(i);
			}
		}

		/*
		Point pt1 = points->at(mss[0]);
		Point pt2 = points->at(mss[1]);

		cv::line(img, pt1, pt2, Scalar(0, 0, 255), 1);
		if (bestInlierNumber > 0)
			cv::line(img, bestPt1, bestPt2, Scalar(0, 255, 0), 1);
			*/
		//imshow("Current line", img);
		//waitKey(200);

		// Store the best model
		if (inlierNumber > bestInlierNumber)
		{
			bestInlierNumber = inlierNumber;
			bestInliers = inliers;
			bestLine = (Mat_<double>(3, 1) << a, b, c);

			/*
			bestPt1 = pt1;
			bestPt2 = pt2;*/
		}

		indices.emplace_back(mss[0]);
		indices.emplace_back(mss[1]);
	}

	inliers = bestInliers;
	line = bestLine;
}

// Apply Least-Squares line fitting (PCL).
void FitLineLSQ(const vector<Point2d> * const points, vector<int> &inliers, Mat &line)
{

}