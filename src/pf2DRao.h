#ifndef __PF2DRAO
#define __PF2DRAO

#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

struct state_params
{
	cv::Mat state;
	cv::Mat cov;
	double weight;
};

class KF_model
{
	public:
		KF_model();
		~KF_model();
		cv::Mat Q, R, F, B, H;
		void predict(cv::Mat &state, cv::Mat &cov);
		void update(cv::Mat measurement, cv::Mat &state, cv::Mat &cov);
};

class my_gmm
{
	public:
		my_gmm();
		~my_gmm();
		void loadGaussian(cv::Mat mean, cv::Mat sigma, double weight);
		void resetTracker(int d);
		std::vector<cv::Mat> mean;
		std::vector<double> weight;
		std::vector<KF_model> KFtracker;
		std::vector<state_params> tracks;
		cv::Mat Sigma_a;
		int nParticles;
};

class ParticleFilter
{
	public:
		ParticleFilter(int states, int nParticles);
		~ParticleFilter();		
		void update(cv::Mat measurement);
		cv::Mat getEstimator();
		my_gmm gmm;
	protected:
		double mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma);
		double wsum;
		std::vector<int> resample(std::vector<double> weights, int N);
		double maxWeight(std::vector<double> weights);
};



#endif
