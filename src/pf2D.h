#ifndef __PF2D
#define __PF2D

#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <time.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>


class my_gmm
{
	public:
		my_gmm();
		~my_gmm();
		void loadGaussian(cv::Mat mean, cv::Mat sigma, double weight);
		std::vector<cv::Mat> mean;
		std::vector<cv::Mat> sigma_i;
		std::vector<double> det_s;
		std::vector<double> weight;
		int N;
};

class ParticleFilter
{
	public:
		ParticleFilter(int numParticles, int numDims, bool side1);
		ParticleFilter();
		~ParticleFilter();		
		void predict();
		void update(cv::Mat measurement);
		cv::Mat getEstimator();
		my_gmm gmm;
	protected:

		void resample();
		double mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma);
		double gmmmvnpdf(cv::Mat x_u, cv::Mat sigma_i);
		double eyemvnpdf(cv::Mat x_u, double sigma);
		double maxWeight();
	//	cv::Mat closestMeasurement(cv::Mat measurements, cv::Mat particle);
		int N;
		int d;
		bool side;
		cv::Mat particles;
		std::vector<double> weights;
		int im_width, im_height;
		
		
};



#endif
