#ifndef __MYGMM
#define __MYGMM

#include <iostream>
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
		cv::Mat measurement;
		int bins;
		cv::Mat cov;
		double weight;
};

class my_gmm
{
	public:
		my_gmm();
		~my_gmm();
		void loadGaussian(cv::Mat u, cv::Mat s, cv::Mat &H, cv::Mat &m, double w, double g);
		void resetTracker(int d);
		std::vector<cv::Mat> mean;
		std::vector<double> weight;
		std::vector<KF_model> KFtracker;
		std::vector<state_params> tracks;
		int nParticles;
};

#endif
