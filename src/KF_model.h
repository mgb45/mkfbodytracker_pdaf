#ifndef __KFMODEL
#define __KFMODEL

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

class KF_model
{
	public:
		KF_model();
		~KF_model();
		cv::Mat Q, R, F, B, H;
		void predict(cv::Mat state_in, cv::Mat cov_in, cv::Mat &state_out, cv::Mat &cov_out);
		void update(cv::Mat measurement, cv::Mat state_in, cv::Mat cov_in, cv::Mat &state_out, cv::Mat &cov_out);
};

#endif
