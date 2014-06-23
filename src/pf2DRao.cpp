/*
 * pf2DRao.cpp
 * 
 * Copyright 2013 Michael Burke <mgb45@chillip>
 * 
 * 
 */
#include "pf2DRao.h"

using namespace cv;
using namespace std;

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
void my_gmm::resetTracker()
{
	for (int i = 0; i < (int)KFtracker.size(); i++)
	{
		KFtracker[i].statePre = mean[i].t();
		KFtracker[i].statePost = mean[i].t();
		KFweight[i] = 0.0;
		setIdentity(KFtracker[i].errorCovPost, Scalar::all(4500));
		setIdentity(KFtracker[i].errorCovPre, Scalar::all(4500));
	}
}

// Load a gaussian for gmm with mean, sigma and weight		
void my_gmm::loadGaussian(cv::Mat u, cv::Mat s, double w, double gamma)
{
	mean.push_back(u);
	//cv::Mat temp;
	//invert(s,temp,DECOMP_CHOLESKY);
	weight.push_back(w);
	
	cv::KalmanFilter tracker;
	tracker.init(s.cols,6,s.cols,CV_64F);
	
	tracker.statePre = u.t();
	tracker.statePost = u.t();
	
	setIdentity(tracker.errorCovPost, Scalar::all(500));
	setIdentity(tracker.errorCovPre, Scalar::all(500));
	
	//cv::invert(Sigma_a.inv() + temp, tracker.processNoiseCov, DECOMP_LU);
	tracker.processNoiseCov = (1-gamma*gamma)*s;
	setIdentity(tracker.measurementNoiseCov, Scalar::all(5));
	
	//tracker.transitionMatrix = tracker.processNoiseCov*Sigma_a.inv();
	setIdentity(tracker.transitionMatrix, Scalar::all(gamma));
	
	//cv::invert((Sigma_a.inv() + temp), tracker.controlMatrix, DECOMP_LU);
	tracker.controlMatrix = (1 - gamma)*cv::Mat::eye(s.cols, s.cols, CV_64F);//tracker.controlMatrix*temp;

	tracker.measurementMatrix = cv::Mat::zeros(6,s.cols, CV_64F);
	tracker.measurementMatrix.at<double>(0,9) = 1;
	tracker.measurementMatrix.at<double>(1,10) = 1;
	tracker.measurementMatrix.at<double>(2,0) = 1;
	tracker.measurementMatrix.at<double>(3,1) = 1;
	tracker.measurementMatrix.at<double>(4,12) = 1;
	tracker.measurementMatrix.at<double>(5,13) = 1;
	
	KFtracker.push_back(tracker);
	KFweight.push_back(0.0);
}

ParticleFilter::ParticleFilter(int states)
{
	cv::Mat Sigma_a = Mat::zeros(states, states, CV_64F);
	setIdentity(Sigma_a, Scalar::all(25));
	Sigma_a.at<double>(0,0) = 560;
	Sigma_a.at<double>(1,1) = 560;
	Sigma_a.at<double>(2,2) = 1;
	Sigma_a.at<double>(3,3) = 560;
	Sigma_a.at<double>(4,4) = 560;
	Sigma_a.at<double>(5,5) = 0.1;
	Sigma_a.at<double>(8,8) = 0.1;
	Sigma_a.at<double>(11,11) = 0.1;
	Sigma_a.at<double>(14,14) = 0.1;
	Sigma_a.at<double>(15,15) = 0.09;
	Sigma_a.at<double>(16,16) = 0.09;
	Sigma_a.at<double>(17,17) = 0.09;
	Sigma_a.at<double>(18,18) = 0.01;
	Sigma_a.at<double>(19,19) = 0.01;
	Sigma_a.at<double>(20,20) = 0.01;
	gmm.Sigma_a = Sigma_a;
}

ParticleFilter::~ParticleFilter()
{
}
		
// Weighted	average pose estimate
cv::Mat ParticleFilter::getEstimator()
{
	cv::Mat estimate;
	for (int i = 0;  i < (int)gmm.KFtracker.size(); i++)
	{
		gmm.KFweight[i] = gmm.KFweight[i]/wsum;
		estimate = estimate + gmm.KFweight[i]*gmm.KFtracker[i].statePost;
	}
	wsum = 1.0;
	return estimate;
}

// Evaulate multivariate gaussian - measurement model
double ParticleFilter::mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma)
{
	cv::Mat sigma_i;
	invert(sigma,sigma_i,DECOMP_CHOLESKY);
	cv::Mat x_u(x.size(),x.type());
	x_u = x - u;
	cv::Mat temp = -0.5*x_u.t()*sigma_i*x_u;
	return 1.0/(pow(2.0*M_PI,sigma.rows/2.0)*sqrt(cv::determinant(sigma)))*exp(temp.at<double>(0,0));
}

// Update stage
void ParticleFilter::update(cv::Mat measurement)
{
	// Add eps (5e-2) and renormalise
	for (int i = 0; i < (int)gmm.KFtracker.size(); i++)
	{
		gmm.KFweight[i] = (gmm.KFweight[i] + 5e-2)/(wsum+(int)gmm.KFweight.size()*5e-2);
	}
	
	wsum = 0;
	for (int i = 0; i < (int)gmm.KFtracker.size(); i++)
	{
		gmm.KFtracker[i].predict(gmm.mean[i].t());
		gmm.KFweight[i] = gmm.KFweight[i]*gmm.weight[i]*mvnpdf(measurement,gmm.KFtracker[i].measurementMatrix*gmm.KFtracker[i].statePre,gmm.KFtracker[i].measurementMatrix*gmm.KFtracker[i].errorCovPre*gmm.KFtracker[i].measurementMatrix.t()+gmm.KFtracker[i].measurementNoiseCov);
		wsum = wsum + gmm.KFweight[i];
		gmm.KFtracker[i].correct(measurement);
	}
}
