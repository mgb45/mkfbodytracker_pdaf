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

ParticleFilter::ParticleFilter(int nParticles)
{
	gmm.nParticles = nParticles;
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


cv::Mat ParticleFilter::chol(cv::Mat in)
{
	cv::Mat sigma_i = in.clone();
	if (Cholesky(sigma_i.ptr<double>(), sigma_i.step, sigma_i.cols, 0, 0, 0))
	{
		cv::Mat diagElem = sigma_i.diag();
		for (int e = 0; e < diagElem.rows; ++e)
		{
			double elem = diagElem.at<double>(e);
			sigma_i.row(e) *= elem;
			sigma_i.at<double>(e,e) = 1.0 / elem;
			if (e > 0)
			{
				cv::Mat z = cv::Mat::zeros(sigma_i.rows-e,1,CV_64F);
				z.copyTo(sigma_i.diag(-e));
			}
		}
	}
	return sigma_i;
}

// Evaulate multivariate gaussian - measurement model
double ParticleFilter::mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma)
{
	cv::Mat R = chol(sigma);
	cv::Mat x_u = (x - u).t()*R.inv();
	cv::Mat temp;
	cv::log(R.diag(0),temp);
	double logSqrtDetSigma = cv::sum(temp)[0];
	cv::pow(x_u,2,temp);
	reduce(temp,temp,1,CV_REDUCE_SUM,-1);
	double quadform = temp.at<double>(0,0);
	return (exp(-0.5*quadform - logSqrtDetSigma - x.rows*log(2.0*M_PI)/2.0));
}

cv::Mat ParticleFilter::logmvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma)
{
	cv::Mat sigma_i = chol(sigma);
	cv::Mat x_u = (x - repeat(u,1,x.cols)).t()*sigma_i.inv();
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

cv::Mat ParticleFilter::getSamples(cv::Mat H, cv::Mat M, int N,double scale)
{
	RNG rng;
	cv::Mat output = cv::Mat::zeros(2,N,CV_64F);
	cv::Mat temp = cv::Mat::zeros(1,1,CV_64FC2);
	
	cv::Mat C = 0.55*scale*cv::Mat::eye(2,2,CV_64F);
	
	// Approximate posterior over hands with one gaussian
	cv::Mat full_state = getEstimator();
	Mat_<Vec2d> &U = reinterpret_cast<Mat_<Vec2d>&>(temp);
	for (int k = 0; k < N; k++)
	{
		cv::Mat state = (H*full_state + M);
		randn(temp,state.rowRange(Range(0,2)),C);
		
		output.at<double>(0,k) = U(0,0)[0];
		output.at<double>(1,k) = U(0,0)[1];
	}
	
	return output;
}

void ParticleFilter::getSampleProb(cv::Mat H, cv::Mat M, cv::Mat input1, cv::Mat input2, std::vector<double> &weight1, std::vector<double> &weight2,double scale)
{
	weight1.resize(input1.cols,0);
	weight2.resize(input2.cols,0);
	
	// Aproximate posterior as Gaussian	
	cv::Mat full_state = getEstimator();
		
	cv::Mat state = (H*full_state + M);
	cv::Mat cov = 0.55*scale*cv::Mat::eye(2,2,CV_64F);
		
	cv::Mat R = chol(cov(Range(0,2),Range(0,2)));
	cv::Mat Rinv = R.inv();
	
	cv::Mat temp;
	cv::log(R.diag(0),temp);
	double logSqrtDetSigma = cv::sum(temp)[0];
	for (int k = 0; k < input1.cols; k++)
	{
		cv::Mat x_u = (input1.col(k) - state.rowRange(Range(0,2))).t()*Rinv;
		cv::pow(x_u,2,temp);
		reduce(temp,temp,1,CV_REDUCE_SUM,-1);
		double quadform = temp.at<double>(0,0);
		weight1[k] = weight1[k] + (exp(-0.5*quadform - logSqrtDetSigma - 2*log(2.0*M_PI)/2.0));
			
		x_u = (input2.col(k) - state.rowRange(Range(0,2))).t()*Rinv;
		cv::pow(x_u,2,temp);
		reduce(temp,temp,1,CV_REDUCE_SUM,-1);
		quadform = temp.at<double>(0,0);
		weight2[k] = weight2[k] + (exp(-0.5*quadform - logSqrtDetSigma - 2*log(2.0*M_PI)/2.0));
	}
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
		weights.push_back(mvnpdf(measurement.col(j),gmm.KFtracker[i].H*gmm.tracks[j].state+gmm.KFtracker[i].BH,gmm.KFtracker[i].H*gmm.tracks[j].cov*gmm.KFtracker[i].H.t()+gmm.KFtracker[i].R));
		wsum = wsum + weights[j];
		gmm.KFtracker[i].update(measurement.col(j),gmm.tracks[j].state,gmm.tracks[j].cov);
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
	cv::RNG rng(cv::getTickCount());
	int idx = rng.uniform(0, L);
	double beta = 0.0;
	double mw = maxWeight(weights);
	
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

