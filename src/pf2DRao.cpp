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
	gmm.nParticles = nParticles;
	gmm.resetTracker(states);
	
	idx_v = cv::Mat::zeros(60*80,2,CV_64F);
	for (int j = 0; j < 60; j++)
	{
		for (int i = 0; i < 80; i++)
		{
			idx_v.at<double>(80*j + i,0) = (double)i;
			idx_v.at<double>(80*j + i,1) = (double)j;
		}
	}
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
	cv::Mat x_u = (x - repeat(u,1,x.cols)).t()*sigma_i;
	cv::Mat temp;
	cv::log(sigma_i.diag(0),temp);
	double logSqrtDetSigma = cv::sum(temp)[0];
	cv::pow(x_u,2,temp);
	double quadform = cv::sum(x_u)[0];
	return (exp(-0.5*quadform - logSqrtDetSigma - x.rows*log(2*M_PI)/2));
}

cv::Mat ParticleFilter::logmvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma)
{
	cv::Mat sigma_i;
	invert(sigma,sigma_i,DECOMP_CHOLESKY);
	cv::Mat x_u = (x - repeat(u,1,x.cols)).t()*sigma_i;
	cv::Mat temp;
	cv::log(sigma_i.diag(0),temp);
	double logSqrtDetSigma;
	logSqrtDetSigma = cv::sum(temp)[0];
	cv::pow(x_u,2,temp);
	cv::Mat quadform;
	cv::reduce(temp,quadform,1,CV_REDUCE_SUM,-1);
	cv::Mat output;
	exp(-0.5*quadform - logSqrtDetSigma - x.rows*log(2*M_PI)/2,output);
	return output;
}


cv::Mat ParticleFilter::getProbMap(cv::Mat H, cv::Mat M)
{
	cv::Mat output = cv::Mat::zeros(60,80,CV_8UC3);
	cv::Mat Im;
	std::vector<cv::Mat> Im_arr;
	std::vector<cv::Mat> Im_arr_2;
	for (int i = 0; i < 5; i++)
	{
		Im = cv::Mat::zeros(60*80,1,CV_64F);
		Im_arr.push_back(cv::Mat::zeros(60,80,CV_8UC1));
		for (int k = 0; k < gmm.nParticles; k++)
		{
			cv::Mat state = (H*gmm.tracks[k].state + M)/8.0;
			cv::Mat cov = H*gmm.tracks[k].cov/8.0*H.t();
			Im = Im + gmm.tracks[k].weight*logmvnpdf(idx_v.t(), state.rowRange(Range(3*i,3*i+2)),cov(Range(3*i,3*i+2),Range(3*i,3*i+2)));
		}
		cv::normalize(Im.reshape(1,60), Im_arr[i], 0, 255, NORM_MINMAX, CV_8UC1);
	}
	Im_arr_2.push_back(Im_arr[0]);
	Im_arr_2.push_back(Im_arr[1]);
	Im_arr_2.push_back(Im_arr[2]+Im_arr[3]+Im_arr[4]);
	cv::merge(Im_arr_2,output);
	return output;
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
		weights.push_back(mvnpdf(measurement,gmm.KFtracker[i].H*gmm.tracks[j].state+gmm.KFtracker[i].BH,gmm.KFtracker[i].H*gmm.tracks[j].cov*gmm.KFtracker[i].H.t()+gmm.KFtracker[i].R));
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
	int L = weights.size();
	std::vector<int> indicators;
	int idx = rand() % L;
	double beta = 0.0;
	double mw = maxWeight(weights);
	cv::RNG rng(cv::getTickCount());
	if (mw == 0)
	{
		weights.clear();
		for (int i = 0; i < N; i++)
		{
			indicators.push_back(rng.uniform(0, L));
			weights.push_back(1.0/(double)N);
		}
	}
	else
	{
		idx = 0;
		double step = 1.0 / (double)N;
		beta = rng.uniform(0.0, 1.0)*step;
		for (int i = 0; i < N; i++)
		{
			while (beta > weights[idx])
			{
				beta -= weights[idx];
				idx = (idx + 1) % L;
			}
			beta += step;
			indicators.push_back(idx);
		}
	}
	return indicators;
}

