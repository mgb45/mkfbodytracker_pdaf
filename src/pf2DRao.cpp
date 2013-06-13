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
	N = 0;
}
		
my_gmm::~my_gmm()
{
	mean.clear();
	sigma.clear();
	weight.clear();
	N = 0;
}

// Load a gaussian for gmm with mean, sigma and weight		
void my_gmm::loadGaussian(cv::Mat u, cv::Mat s, double w)
{
	N++;
	mean.push_back(u);
	cv::Mat temp;
	invert(s,temp,DECOMP_CHOLESKY);
	sigma.push_back(s);
	weight.push_back(w);
	
	cv::KalmanFilter tracker;
	tracker.init(8,4,8,CV_64F);
	
	randu(tracker.statePre, Scalar(0), Scalar(480));
	randu(tracker.statePost, Scalar(0), Scalar(480));
	
	setIdentity(tracker.errorCovPost, Scalar::all(4500));
	setIdentity(tracker.errorCovPre, Scalar::all(4500));
	
	cv::invert(Sigma_a.inv() + temp, tracker.processNoiseCov, DECOMP_LU);
	setIdentity(tracker.measurementNoiseCov, Scalar::all(25));
	
	tracker.transitionMatrix = tracker.processNoiseCov*Sigma_a.inv();
	
	cv::invert((Sigma_a.inv() + temp), tracker.controlMatrix, DECOMP_LU);
	tracker.controlMatrix = tracker.controlMatrix*temp;

	tracker.measurementMatrix = cv::Mat::zeros(4,8, CV_64F);
	tracker.measurementMatrix.at<double>(0,6) = 1;
	tracker.measurementMatrix.at<double>(1,7) = 1;
	tracker.measurementMatrix.at<double>(2,0) = 1;
	tracker.measurementMatrix.at<double>(3,1) = 1;
	
	KFtracker.push_back(tracker);
	KFweight.push_back(0.0);
}

ParticleFilter::ParticleFilter()
{
	Sigma_a = Mat::zeros(8, 8, CV_64F);
	setIdentity(Sigma_a, Scalar::all(125));
	Sigma_a.at<double>(4,4) = 25;
	Sigma_a.at<double>(5,5) = 25;
	Sigma_a.at<double>(6,6) = 25;
	Sigma_a.at<double>(7,7) = 25;
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
	//~ cout << estimate << std::endl;
	return estimate;
}

// Evaulate multivariate gaussian - measurement model
double ParticleFilter::mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma)
{
//	cout << x << std::endl;
	cv::Mat sigma_i;
	invert(sigma,sigma_i,DECOMP_CHOLESKY);
	cv::Mat x_u(x.size(),x.type());
	x_u = x - u;
	cv::Mat temp = -0.5*x_u.t()*sigma_i*x_u;
	//cout << 1.0/(pow(2.0*M_PI,sigma.rows/2.0)*sqrt(cv::determinant(sigma)))*exp(temp.at<double>(0,0)) << std::endl;	
	return 1.0/(pow(2.0*M_PI,sigma.rows/2.0)*sqrt(cv::determinant(sigma)))*exp(temp.at<double>(0,0));
}

// Update stage
void ParticleFilter::update(cv::Mat measurement)
{
	wsum = 0;
	for (int i = 0; i < (int)gmm.KFtracker.size(); i++)
	{
		gmm.KFtracker[i].predict(gmm.mean[i].t());
		gmm.KFweight[i] = gmm.weight[i]*mvnpdf(gmm.KFtracker[i].statePost,gmm.mean[i].t(),gmm.sigma[i]+Sigma_a)*mvnpdf(measurement,gmm.KFtracker[i].measurementMatrix*gmm.KFtracker[i].statePre,gmm.KFtracker[i].measurementMatrix*gmm.KFtracker[i].errorCovPre*gmm.KFtracker[i].measurementMatrix.t()+gmm.KFtracker[i].measurementNoiseCov);
		wsum = wsum + gmm.KFweight[i];
		gmm.KFtracker[i].correct(measurement);
	}
}

