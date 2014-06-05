#ifndef __PF2DRAO
#define __PF2DRAO

#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "my_gmm.h"
#include "KF_model.h"
#include <fstream>

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
		std::vector<int> resampleStratified(std::vector<double> weights, int N);
		std::vector<int> resampleStratifiedFernhead(std::vector<double> weights, int N,std::vector<double> &new_weights);
		double maxWeight(std::vector<double> weights);
};



#endif
