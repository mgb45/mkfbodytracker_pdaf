#include "KF_model.h"

KF_model::KF_model()
{
}

KF_model::~KF_model()
{
}

void KF_model::predict(cv::Mat state_in, cv::Mat cov_in, cv::Mat &state_out, cv::Mat &cov_out)
{
	state_out = F*state_in + B;
	cov_out = F*cov_in*F.t() + Q;
}

void KF_model::update(cv::Mat measurement, cv::Mat state_in, cv::Mat cov_in, cv::Mat &state_out, cv::Mat &cov_out)
{
	cv::Mat y = measurement - H*state_in;
	cv::Mat S = H*cov_in*H.t() + R;
	cv::Mat K = cov_in*H.t()*S.inv();
	
	state_out = state_in + K*y;
	cov_out = (cv::Mat::eye(cov_in.rows,cov_in.cols,cov_in.type()) - K*H)*cov_in;
}
