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
		temp.cov = cv::Mat::zeros(d,d,CV_64F);
		temp.weight = 1.0/(double)nParticles;
		setIdentity(temp.cov, cv::Scalar::all(1000));
		tracks.push_back(temp);
	}
}

// Load a gaussian for gmm with mean, sigma and weight		
void my_gmm::loadGaussian(cv::Mat u, cv::Mat s, cv::Mat &H, cv::Mat &m, double w, double g)
{
	mean.push_back(u);
	weight.push_back(w);
	
	KF_model tracker;
	
	tracker.Q = (1-g*g)*s;
	tracker.R = 5*cv::Mat::eye(6,6, CV_64F);
	
	tracker.F = g*cv::Mat::eye(s.cols,s.cols, CV_64F);
		
	tracker.B = (1.0 - g)*u.t();

	cv::Mat H1 = cv::Mat::zeros(6,m.cols, CV_64F);
	H1.at<double>(0,9) = 1;
	H1.at<double>(1,10) = 1;
	H1.at<double>(2,0) = 1;
	H1.at<double>(3,1) = 1;
	H1.at<double>(4,12) = 1;
	H1.at<double>(5,13) = 1;
	
	tracker.H = H1*H.t();
	tracker.BH = H1*m.t();
	
	KFtracker.push_back(tracker);
}
