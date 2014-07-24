#ifndef __KFMODEL
#define __KFMODEL

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class KF_model
{
	public:
		KF_model();
		~KF_model();
		cv::Mat Q, R, F, B, H, BH;
		void predict(cv::Mat &state, cv::Mat &cov);
		void update(cv::Mat measurement, cv::Mat &state, cv::Mat &cov);
};

#endif
