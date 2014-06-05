#include "my_gmm.h"

state_params::state_params()
{
}

state_params::~state_params()
{
}

state_params::state_params(const state_params& other)
{
	state = other.state.clone();
	cov = other.cov.clone();
	weight = other.weight;
}

// GMM storage class
my_gmm::my_gmm()
{
}
		
my_gmm::~my_gmm()
{
	mean.clear();
	weight.clear();
}

//Re-initialise tracker with large uncertainty and random state
void my_gmm::resetTracker(int d)
{
	tracks.clear();
	for (int i = 0; i < nParticles; i++)
	{
		state_params temp;
		temp.state = cv::Mat::zeros(d,1,CV_64F);
		randu(temp.state,1,480);
		for (int j = 2; j < d-6; j+=2)
		{
			randn(temp.state.row(j),3,5);
		}
		for (int j = d-6; j < d; j++)
		{
			randn(temp.state.row(j),0,5);
		}
		temp.cov = cv::Mat::zeros(d,d,CV_64F);
		temp.weight = 1.0/(double)nParticles;
		cv::setIdentity(temp.cov, cv::Scalar::all(4500));
		tracks.push_back(temp);
	}
}

// Load a gaussian for gmm with mean, sigma and weight		
void my_gmm::loadGaussian(cv::Mat u, cv::Mat s, double w)
{
	mean.push_back(u);
	cv::Mat temp;
	cv::invert(s,temp,cv::DECOMP_CHOLESKY);
	weight.push_back(w);
	
	KF_model tracker;
		
	cv::invert(Sigma_a.inv() + temp, tracker.Q, cv::DECOMP_LU);
	tracker.R = 5*cv::Mat::eye(6,6, CV_64F);
	
	tracker.F = tracker.Q*Sigma_a.inv();
	
	tracker.B = tracker.Q*temp*u.t();

	tracker.H = cv::Mat::zeros(6,s.cols, CV_64F);
	tracker.H.at<double>(0,9) = 1;
	tracker.H.at<double>(1,10) = 1;
	tracker.H.at<double>(2,0) = 1;
	tracker.H.at<double>(3,1) = 1;
	tracker.H.at<double>(4,12) = 1;
	tracker.H.at<double>(5,13) = 1;
	
	KFtracker.push_back(tracker);
}
