#ifndef __PF2DRAO
#define __PF2DRAO

#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class my_gmm
{
	public:
		my_gmm();
		~my_gmm();
		void loadGaussian(cv::Mat mean, cv::Mat sigma, double weight,double gamma);
		void resetTracker();
		std::vector<cv::Mat> mean;
		std::vector<double> weight;
		std::vector<cv::KalmanFilter> KFtracker;
		std::vector<double> KFweight;
		cv::Mat Sigma_a;
};

class ParticleFilter
{
	public:
		ParticleFilter(int states);
		~ParticleFilter();		
		void update(cv::Mat measurement);
		cv::Mat getEstimator();
		my_gmm gmm;
	protected:
		double mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma);
		double wsum;
};



#endif
