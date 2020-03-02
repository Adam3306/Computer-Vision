// CV_Practise_RANSAC.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"

#include <iostream>
#include <opencv2\core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <vector>

// A function to generate synthetic data
void generateData(
	std::vector<cv::Point2d> &points_, // The vector where the generated points should be stored
	const double &noise_, // The noise parameter added to the point coordinates
	const size_t &number_of_inliers_, // The number of inliers (i.e. points on the line) to generate
	const size_t &number_of_outliers_, // The number of outliers (i.e. random points) to generate
	const cv::Size &image_size_); // The size of the image 

// A function to draw points into an image
void drawPoints(
	const std::vector<cv::Point2d> &points_, // The points to be drawn
	cv::Mat &image_, // The image where the points are supposed to be drawn
 	const cv::Scalar &color_, // The color used for the drawing
	const double &size_ = 3.0, // The radius of circles drawn as points
	const std::vector<size_t> * inliers_ = nullptr); // A subset of the points

// The function fitting a 2D line by applying RANSAC
void fitLineRANSAC(
	const std::vector<cv::Point2d> &points_, // The points used for the line fitting
	std::vector<size_t> &inliers_, // The inliers of the obtained line
	cv::Mat &found_line_, // The parameters of the obtained line
	const double &threshold_, // The inlier-outlier threshold used for determining which points are inliers
	const size_t& confidence); // The required iteration number 

// Draw a 2D line to an image
void draw2DLine(
	cv::Mat &image_, // The image where the line is supposed to be drawn
	const cv::Mat &line_, // The line parameters
	const cv::Scalar &color_, // The color of the drawing
	const double &size_); // The line weight

// Return a random number in-between 0 and 1.
double getRandomNumber();

// Fit a 2D line to a set of 2D points by least-squares fitting
void fitLineLSQ(
	const std::vector<cv::Point2d> * const points_, // All points
	const std::vector<size_t> &inliers_, // The subset of points which are used for the fitting
	cv::Mat &line_); // The estimated line parameters

size_t getIterationNumber(			// Number of inliers of the current best model
	const size_t& inlier_number_,	// The required confidence in the result
	const double& confidence_,		// The number of points
	const size_t& point_number_,	
	const size_t& sample_size_);

int _tmain(int argc, _TCHAR* argv[])
{
	std::vector<cv::Point2d> points; // The vector where the generated points are stored
	cv::Mat image = cv::Mat::zeros(600, 600, CV_8UC3); // The generated image
	const double noise = 10., // The noise (in pixels) added to the point coordinates
		threshold = 25.; // The inlier-outlier threshold for RANSAC
	const size_t number_of_inliers = 100, // The number inliers to be generated
		number_of_outliers = 100; // The number of outlier to be generated

	// Generating a synthetic scene to have points on which RANSAC
	// can be tested.
	generateData(points, // Generated 2D points
		noise, // Noise added to the point coordinates
		number_of_inliers, // Number of inliers
		number_of_outliers, // Number of outliers
		cv::Size(image.cols, image.rows)); // Size of the image

	// Draw the points to the image
	drawPoints(points,  // Input 2D points
		image,// The image to draw
		cv::Scalar(255,255,255)); // Color of the points

	// Show the image with the points
	cv::imshow("Input image", image);

	std::vector<size_t> inliers; // The found inliers
	cv::Mat found_line; // The found line parameters
	// Find a line by RANSAC
	fitLineRANSAC(points, // Input 2D points
		inliers, // Obtained inliers
		found_line, // Obtained line
		threshold, // Threshold
		0.99); // The image



	cv::Mat polished_line; // The polished line parameters
	// Re-calculate the line parameters by applying least-squared fitting to all found inliers
	fitLineLSQ(&points,  // Input 2D points
		inliers,  // The found inliers
		polished_line); // The refined model parameters

	// Draw the inliers and the found line
	drawPoints(points,  // Input 2D points
		image, // The image to draw
		cv::Scalar(0, 255, 0), // Color of the points
		3, // Size of the drawn points
		&inliers); // Inliers

	// Draw the found line
	draw2DLine(image,
		found_line,
		cv::Scalar(0, 0, 255),
		2);

	// Draw the polished line
	draw2DLine(image,
		polished_line,
		cv::Scalar(255, 0, 0),
		2);

	// Show the image with the points
	cv::imshow("Output image", image);
	// Wait for keypress
	cv::waitKey(0);

	return 0;
}

void draw2DLine(
	cv::Mat &image_, // The image where the line is supposed to be drawn
	const cv::Mat &line_, // The line parameters
	const cv::Scalar &color_, // The color of the drawing
	const double &size_) // The line weight
{
	double a = line_.at<double>(0);
	double b = line_.at<double>(1);
	double c = line_.at<double>(2);

	double x1 = 0;
	double y1 = (-x1 * a - c) / b;

	double x2 = image_.cols;
	double y2 = (-x2 * a - c) / b;

	cv::line(image_,
		cv::Point2d(x1, y1),
		cv::Point2d(x2, y2),
		color_,
		size_);
}

// Draw points to the image
void drawPoints(
	const std::vector<cv::Point2d> &points_, // The points to be drawn
	cv::Mat &image_, // The image where the points are supposed to be drawn
	const cv::Scalar &color_, // The color used for the drawing
	const double &size_, // The radius of circles drawn as points
	const std::vector<size_t> * inliers_) // A subset of the points
{
	if (inliers_ == nullptr)
		for (const auto &point : points_)
			circle(image_, point, size_, color_, -1);
	else
		for (const auto &point_idx : *inliers_)
			circle(image_, points_[point_idx], size_, color_, -1);
}

// Generate a synthetic line and sample that. Then add outliers to the data.
void generateData(
	std::vector<cv::Point2d> &points_, // The vector where the generated points should be stored
	const double &noise_, // The noise parameter added to the point coordinates
	const size_t &number_of_inliers_, // The number of inliers (i.e. points on the line) to generate
	const size_t &number_of_outliers_, // The number of outliers (i.e. random points) to generate
	const cv::Size &image_size_) // The size of the image 
{
	// Generate random line by its normal direction and a center point
	cv::Point2d center(getRandomNumber() * image_size_.width,
		getRandomNumber() * image_size_.height); // A point of the line

	double a, b, c;
	const double alpha = getRandomNumber() * 3.14; // A random angle determining the line direction
	a = sin(alpha); // The x coordinate of the line normal
	b = cos(alpha); // The y coordinate of the line normal
	c = -a * center.x - b * center.y; // The offset of the line coming from equation "a x + b y + c = 0"

	// Generate random points on that line
	double x, y;
	points_.reserve(number_of_inliers_ + number_of_outliers_);
	for (auto i = 0; i < number_of_inliers_; ++i)
	{
		x = getRandomNumber() * image_size_.width; // Generate a random x coordinate in the window
		y = -(a * x + c) / b; // Calculate the corresponding y coordinate

		// Add the point to the vector after adding random noise
		points_.emplace_back(
			cv::Point2d(x + noise_ * getRandomNumber(), y + noise_ * getRandomNumber()));
	}

	// Add outliers
	for (auto i = 0; i < number_of_outliers_; ++i)
	{
		x = getRandomNumber() * image_size_.width; // Generate a random x coordinate in the window
		y = getRandomNumber() * image_size_.height; // Generate a random y coordinate in the window

		// Add outliers, i.e., random points in the image
		points_.emplace_back(cv::Point2d(x, y));
	}
}

double getRandomNumber()
{
	return static_cast<double>(rand()) / RAND_MAX;
}

// Apply RANSAC to fit points to a 2D line
void fitLineRANSAC(
	const std::vector<cv::Point2d> &points_, // The points used for the line fitting
	std::vector<size_t> &inliers_, // The inliers of the obtained line
	cv::Mat &found_line_, // The parameters of the obtained line
	const double &threshold_, // The inlier-outlier threshold used for determining which points are inliers
	const size_t &confidence) // The required iteration number
{
	constexpr size_t sample_size = 2; // Sample size
	size_t maximum_iteration_number = std::numeric_limits<size_t>::max();

	const size_t point_number = points_.size(); // The number of points
	size_t * const sample = new size_t[sample_size];

	// The inliers of the current model
	std::vector<size_t> tmp_inliers;
	// Occupy the maximum memory required early
	tmp_inliers.reserve(point_number);
	found_line_.create(3, 1, CV_64F);
	
	for (size_t iteration = 0; iteration < maximum_iteration_number; ++iteration)
	{
		// Select a random sample of size two
		for (size_t sample_idx = 0; sample_idx < sample_size; ++sample_idx)
		{
			// Select a points via its index randomly
			size_t idx = round(getRandomNumber() * (point_number - 1));
			sample[sample_idx] = idx;

			// Check if the selected index has been already selected
			for (size_t prev_sample_idx = 0; prev_sample_idx < sample_idx; ++prev_sample_idx)
			{
				if (sample[prev_sample_idx] == sample[sample_idx])
				{
					--sample_idx;
					continue;
				}
			}
		}

		// Fit a line to the selected points
		cv::Point2d pt1 = points_.at(sample[0]); // The first point of the line
		cv::Point2d pt2 = points_.at(sample[1]); // The second point of the line
 		cv::Point2d v = pt2 - pt1;  // The direction of the line
		v = v / norm(v); // Normalize the direction since the length does not matter
		cv::Point2d n(-v.y, v.x); // The normal of the line, i.e., the direction rotated by 90°.
		const double &a = n.x, // The x coordinate of the normal
			&b = n.y; // The y coordinate of the normal
		double c = 
			-a * pt1.x - b * pt1.y; // The offset coming from equation "a x + b y + c = 0"
		
		// Iterate through all the points and count the inliers
		tmp_inliers.resize(0);
		for (size_t point_idx = 0; point_idx < point_number; ++point_idx)
		{
			const double &x = points_[point_idx].x,
				&y = points_[point_idx].y;

			// Calculate the point-to-model distance
			const double signed_distance = a * x + b * y + c;
			const double distance = abs(signed_distance);

			// If the point is closer than the threshold add it to the
			// set of inliers.
			if (distance < threshold_)
				tmp_inliers.emplace_back(point_idx);
		}

		// If the current line has more inliers than the previous so-far-the-best, update
		// the best parameters
		if (tmp_inliers.size() > inliers_.size())
		{
			// Swap the inliers with that of the previous so-far-the-best model
			tmp_inliers.swap(inliers_);
			// Save the model parameters
			found_line_.at<double>(0) = a;
			found_line_.at<double>(1) = b;
			found_line_.at<double>(2) = c;

			maximum_iteration_number = getIterationNumber(inliers_.size(), confidence, point_number, sample_size);

		}
	}

	// Clean up the memory
	delete[] sample;
}

// Apply Least-Squares line fitting
void fitLineLSQ(
	const std::vector<cv::Point2d>* const points_, // All points
	const std::vector<size_t>& inliers_, // The subset of points which are used for the fitting
	cv::Mat& line_) // The estimated line parameters
{
	// The number of inliers, i.e, the number of rows in the coefficent matrix
	const size_t& inlier_number = inliers_.size();

	cv::Mat A(inlier_number, 3, CV_64F);
	double* A_ptr = reinterpret_cast<double*>(A.data);

	for (const size_t& inlier_idx : inliers_)
	{
		// The coordinates of the current inlier
		const double& x = points_->at(inlier_idx).x;
		const double& y = points_->at(inlier_idx).y;

		// The constraint in A coming from: a*x + b*y + c = 0
		*(A_ptr++) = x;
		*(A_ptr++) = y;
		*(A_ptr++) = 1.0;
	}

	// Calc matrix A^T * A
	cv::Mat AtA = A.t() * A;	// The matrix to be decomposed
	cv::Mat eigen_values;		// The eigen values of matrix
	cv::Mat eigen_vectors;		// The eigen vectors of matrix

	cv::eigen(AtA, eigen_values, eigen_vectors);

	// The best line is the eigen vector corresponding to the lowest eigen value
	line_ = eigen_vectors.row(2);
}

size_t getIterationNumber(const size_t& inlier_number_, const double& confidence_, const size_t& point_number_, const size_t& sample_size_)
{
	// If all points are selected return 0
	if (inlier_number_ == point_number_)
	{
		return 0;
	}

	const double inlier_ratio = static_cast<double>(inlier_number_) / point_number_;

	const double probability_of_not_selecting_good_sample = 1.0 - pow(inlier_ratio, sample_size_);

	const double log_probability_of_not_selecting_good_sample = log(probability_of_not_selecting_good_sample);

	if (abs(log_probability_of_not_selecting_good_sample) < std::numeric_limits<double>::epsilon())
	{
		return std::numeric_limits<size_t>::max();
	}

	const size_t iteration_number = log(1.0 - confidence_) / log_probability_of_not_selecting_good_sample;
	
	return iteration_number;
}