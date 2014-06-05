#include "KF_model.h"

KF_model::KF_model()
{
}

KF_model::~KF_model()
{
}

void KF_model::predict(cv::Mat &state, cv::Mat &cov)
{
	state = F*state + B;
	cov = F*cov*F.t() + Q;
}

void KF_model::update(cv::Mat measurement, cv::Mat &state, cv::Mat &cov)
{
	cv::Mat y = measurement - H*state;
	cv::Mat S = H*cov*H.t() + R;
	cv::Mat K = cov*H.t()*S.inv();
	
	state = state + K*y;
	cov = (cv::Mat::eye(cov.rows,cov.cols,cov.type()) - K*H)*cov;
}
