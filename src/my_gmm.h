#ifndef __MYGMM
#define __MYGMM

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "KF_model.h"

class state_params
{
	public:
		state_params();
		~state_params();
		state_params(const state_params& other);
		cv::Mat state;
		cv::Mat cov;
		double weight;
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
		int nParticles;
};

#endif
