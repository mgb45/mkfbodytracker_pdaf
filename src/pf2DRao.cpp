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

ParticleFilter::ParticleFilter(int states, int nParticles)
{
	cv::Mat Sigma_a = Mat::zeros(states, states, CV_64F);
	setIdentity(Sigma_a, Scalar::all(5));
	Sigma_a.at<double>(0,0) = 50;
	Sigma_a.at<double>(1,1) = 50;
	Sigma_a.at<double>(2,2) = 1;
	Sigma_a.at<double>(3,3) = 5;
	Sigma_a.at<double>(4,4) = 5;
	Sigma_a.at<double>(5,5) = 1;
	Sigma_a.at<double>(8,8) = 1;
	Sigma_a.at<double>(11,11) = 1;
	Sigma_a.at<double>(14,14) = 1;
	Sigma_a.at<double>(15,15) = 0.5;
	Sigma_a.at<double>(16,16) = 0.5;
	Sigma_a.at<double>(17,17) = 0.5;
	Sigma_a.at<double>(18,18) = 0.01;
	Sigma_a.at<double>(19,19) = 0.01;
	Sigma_a.at<double>(20,20) = 0.01;
	gmm.Sigma_a = Sigma_a;
	gmm.nParticles = nParticles;
	gmm.resetTracker(states);
}

ParticleFilter::~ParticleFilter()
{
}
		
// Weighted	average pose estimate
cv::Mat ParticleFilter::getEstimator()
{
	cv::Mat estimate;
	for (int i = 0;  i < gmm.nParticles; i++)
	{
		estimate = estimate + 1.0/(double)gmm.nParticles*gmm.tracks[i].state;
	}
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
	// Propose indicators
	
	std::vector<int> indicators = resample(gmm.weight, gmm.nParticles);
	std::vector<double> weights;
	wsum = 0;
	int i;
	std::vector<state_params> temp;
	for (int j = 0; j < gmm.nParticles; j++) //update KF for each track using indicator samples
	{
		i = indicators[j];
		gmm.KFtracker[i].predict(gmm.tracks[j].state,gmm.tracks[j].cov);
		weights.push_back(mvnpdf(measurement,gmm.KFtracker[i].H*gmm.tracks[j].state,gmm.KFtracker[i].H*gmm.tracks[j].cov*gmm.KFtracker[i].H.t()+gmm.KFtracker[i].R));
		wsum = wsum + weights[j];
		gmm.KFtracker[i].update(measurement,gmm.tracks[j].state,gmm.tracks[j].cov);
		temp.push_back(gmm.tracks[j]);
	}
		
	for (int i = 0; i < (int)gmm.tracks.size(); i++)
	{
		weights[i] = weights[i]/wsum;
	}
		
	// Re-sample tracks
	indicators.clear();
	indicators = resample(weights, gmm.nParticles);
	for (int j = 0; j < gmm.nParticles; j++) //update KF for each track using indicator samples
	{
		gmm.tracks[j] = temp[indicators[j]];
	}
	wsum = 1.0;
}

// Return best weight
double ParticleFilter::maxWeight(std::vector<double> weights)
{
	double mw = 0;
	for (int i = 0; i < (int)weights.size(); i++)
	{
		if (weights[i] > mw)
		{
			mw = weights[i];
		}
	}
	return mw;
}

// Resample according to weights
std::vector<int> ParticleFilter::resample(std::vector<double> weights, int N)
{
	std::vector<int> indicators;
	int idx = rand() % N;
	double beta = 0.0;
	double mw = maxWeight(weights);
	if (mw == 0)
	{
		weights.clear();
		for (int i = 0; i < N; i++)
		{
			indicators.push_back(i);
			weights.push_back(1.0/(double)N);
		}
	}
	else
	{
		idx = 0;
		double step = 1.0 / (double)N;
		beta = ((double)rand()/RAND_MAX)*step;
		for (int i = 0; i < N; i++)
		{
			while (beta > weights[idx])
			{
				beta -= weights[idx];
				idx = (idx + 1) % N;
			}
			beta += step;
			indicators.push_back(idx);
		}
	}
	return indicators;
}
