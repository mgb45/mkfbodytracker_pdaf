/*
 * pf2D.cpp
 * 
 * Copyright 2013 Michael Burke <mgb45@chillip>
 * 
 * 
 */
#include "pf2D.h"

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
	sigma_i.clear();
	det_s.clear();
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
	sigma_i.push_back(temp);
	det_s.push_back(1.0/(pow(2.0*M_PI,u.cols/2.0)*sqrt(cv::determinant(s))));
	weight.push_back(w);
}

ParticleFilter::ParticleFilter()
{
}

//PF constructor
ParticleFilter::ParticleFilter(int numParticles, int numDims, bool side1)
{
	N = numParticles;
	d = numDims;
	side = side1;
	im_height = 480;
	im_width = 640;
	particles = cv::Mat(N,d,CV_64F);
	// Uniform weights
	for (int i = 0; i < N; i++)
	{
		weights.push_back(1.0/(double)N);
	}
	
	// Randomise particles across image
	for (int i = 0; i < d; i++)
	{
		
		if (i == 6)
		{
			cv::randu(particles.col(i),Scalar(im_width/2.0*side +1),Scalar(im_width/2.0 + im_width/2.0*side));
		}
		else
		{
			cv::randu(particles.col(i),Scalar(1),Scalar(((i%2)==0)*im_width +(((i+1)%2)==0)*im_height));
		}
	}
}

ParticleFilter::~ParticleFilter()
{
	weights.clear();
}
		
// Weighted	average pose estimate
cv::Mat ParticleFilter::getEstimator()
{
	cv::Mat estimate;
	for (int i = 0; i < N; i++)
	{
		estimate = estimate + weights[i]*particles.row(i);
	}
	return estimate;
}

// Random walk motion model
void ParticleFilter::predict()
{
	cv::Mat temp(1,d,CV_64F);
	//particles = particles + temp;
	for (int i = 0; i < N; i++)
	{
		cv::randn(temp.colRange(2,6),0,5);
		cv::randn(temp.colRange(6,8),0,5);
		cv::randn(temp.colRange(0,2),0,5);
		//cout << temp;
		particles.row(i) = particles.row(i) + temp;
	}
}

// Evaluate gmm likelihood
double ParticleFilter::gmmmvnpdf(cv::Mat x_u, cv::Mat sigma_i)
{
	cv::Mat temp = -0.5*x_u*sigma_i*x_u.t();
	return expf(float(temp.at<double>(0,0)));
}

// Evaulate multivariate gaussian - measurement model
double ParticleFilter::mvnpdf(cv::Mat x, cv::Mat u, cv::Mat sigma)
{
//	cout << x << std::endl;
	cv::Mat sigma_i;
	invert(sigma,sigma_i,DECOMP_CHOLESKY);
	cv::Mat x_u(x.size(),x.type());
	x_u = x - u;
	cv::Mat temp = -0.5*x_u*sigma_i*x_u.t();
	return 1.0/(pow(2.0*M_PI,sigma.rows/2.0)*sqrt(cv::determinant(sigma)))*exp(temp.at<double>(0,0));
}

// Evaulate multivariate gaussian with identity sigma - measurement model
double ParticleFilter::eyemvnpdf(cv::Mat x_u, double scale)
{
	cv::Mat temp = -0.5*x_u*1.0/scale*cv::Mat::eye(2,2,CV_64F)*x_u.t();
	return 1.0/(pow(2.0*M_PI*scale,x_u.cols/2.0))*exp(temp.at<double>(0,0));
}

/*cv::Mat ParticleFilter::closestMeasurement(cv::Mat measurements, cv::Mat particle)
{
	double dmin = 10000,temp = 0;
	int minidx = -1;
	for (int i = 0; i < measurements.rows; i++)
	{
		temp = norm(measurements.row(i) - particle);
		
		if (temp <= dmin)
		{
			dmin = temp;
			minidx = i;
		}
	}
	return measurements.row(minidx);
}*/

// Update stage
void ParticleFilter::update(cv::Mat measurement)
{
	// Pick best measurement for each particle!
	time_t  start_time = clock(), end_time;
	float time1;
	double prior[N];
	double likelihood[N];
	double weightSum = 0;

	for (int i = 0; i < N; i++)
	{
		prior[i] = 0;
		for (int j = 0; j < gmm.N; j++)
		{
			//prior[i] = prior[i] + gmm.weight[j]*mvnpdf(particles.row(i),gmm.mean[j],gmm.sigma[j]);
			prior[i] = prior[i] + gmm.weight[j]*gmm.det_s[j]*gmmmvnpdf(particles.row(i) - gmm.mean[j], gmm.sigma_i[j]);
		}
		//closestMeasurement(measurement,particles(Range(i,i+1),Range(0,2)))
		//likelihood[i] *= mvnpdf(particles(Range(i,i+1),Range(6,8)),measurement.row(0),15*cv::Mat::eye(2,2,CV_64F));
		//likelihood[i] *= mvnpdf(particles(Range(i,i+1),Range(0,2)),measurement.row(1),15*cv::Mat::eye(2,2,CV_64F));
		likelihood[i] = eyemvnpdf(particles(Range(i,i+1),Range(6,8))-measurement.row(0),15)*eyemvnpdf(particles(Range(i,i+1),Range(0,2))-measurement.row(1),15);
		// Arm order!
		weights[i] = prior[i]*likelihood[i];
		weightSum += weights[i];
	}

	for (int i = 0; i < N; i++)
	{
		weights[i] = weights[i]/weightSum;
	}
	
	end_time = clock();
	time1 = (float) (end_time - start_time) / CLOCKS_PER_SEC; 
	start_time = end_time;
	printf("Update: %f seconds\n", time1);
		
	resample(); // resample according to weights
	end_time = clock();
	time1 = (float) (end_time - start_time) / CLOCKS_PER_SEC; 
	start_time = end_time;
	printf("Resample: %f seconds\n", time1);
	
	// Plot weights
	//cv::Mat weightImage =cv::Mat::zeros(480,640,CV_8UC3);
	/*for (int i = 1; i < N; i++)
	{
		line(weightImage, Point(480*(double)i/(double)N, 640 - cvRound(639*weights[i-1]+1)), Point(480*(double)(i+1)/(double)N, 640 - cvRound(639*weights[i]+1)), Scalar(255, 0, 0), 2, 8, 0);
		circle(weightImage, Point(particles.at<double>(i,0),particles.at<double>(i,1)), 1, Scalar( 0, 0, 255), 1, 8);
		circle(weightImage, Point(particles.at<double>(i,6),particles.at<double>(i,7)), 1, Scalar( 0, 255, 0), 1, 8);
		circle(weightImage, Point(particles.at<double>(i,2),particles.at<double>(i,3)), 1, Scalar( 255, 0, 0), 1, 8);
		circle(weightImage, Point(particles.at<double>(i,4),particles.at<double>(i,5)), 1, Scalar( 255, 255, 0), 1, 8);
	}
	char fname[20];
	sprintf (fname,"win%d",side);
	imshow (fname, weightImage);                   // Show our image inside it.
	waitKey(50);*/
	
	predict(); // predict new pos
	end_time = clock();
	time1 = (float) (end_time - start_time) / CLOCKS_PER_SEC; 
	start_time = end_time;
	printf("Predict: %f seconds\n", time1);
}
// Return best weight
double ParticleFilter::maxWeight()
{
	double mw = 0;
	for (int i = 0; i < N; i++)
	{
		if (weights[i] > mw)
		{
			mw = weights[i];
		}
	}
	return mw;
}
// Resample according to weights
void ParticleFilter::resample()
{
	cv::Mat old_particles = particles.clone();
	int idx = rand() % N;
	double beta = 0.0;
	double mw = maxWeight();
	cout << "Max weight: " << mw << std::endl;
	if (mw == 0)
	{
		for (int i = 0; i < d; i++)
		{
			if (i == 6)
			{
				cv::randu(particles.col(i),Scalar(im_width/2.0*side +1),Scalar(im_width/2.0 + im_width/2.0*side));
			}
			else
			{
				cv::randu(particles.col(i),Scalar(1),Scalar(((i%2)==0)*im_width +(((i+1)%2)==0)*im_height));
			}
		}
		weights.clear();
		for (int i = 0; i < N; i++)
		{
			weights.push_back(1.0/(double)N);
		}
	}
	else
	{
		idx = 0;
		double step = 1.0 / (double)N;
		beta = ((double)rand()/RAND_MAX)*step;
		particles = cv::Mat::zeros(N, d, CV_64F);
		for (int i = 0; i < N; i++)
		{
			while (beta > weights[idx])
			{
				beta -= weights[idx];
				idx = (idx + 1) % N;
			}
			beta += step;
			particles.row(i) = particles.row(i) + old_particles.row(idx);
		}
	}
}
