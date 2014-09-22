#ifndef __PFTRACKER
#define __PFTRACKER

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <opencv/cv.h>
#include <ros/ros.h>
#include <cv_bridge/cv_bridge.h>
#include "opencv2/video/tracking.hpp"
#include <opencv2/features2d/features2d.hpp>
#include <image_transport/image_transport.h>
#include <sensor_msgs/image_encodings.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/RegionOfInterest.h"
#include "geometry_msgs/Point.h"
#include <message_filters/subscriber.h>
#include <message_filters/time_synchronizer.h>
#include "pf2DRao.h"
#include <sstream>
#include <string>
#include <ros/package.h>
#include "opencv2/ml/ml.hpp"
#include "measurementproposals/HFPose2D.h"
#include "measurementproposals/HFPose2DArray.h"
#include <tf/transform_broadcaster.h>

class PFTracker
{
	public:
		PFTracker();
		~PFTracker();
				
	private:
		ros::NodeHandle nh;
		image_transport::Publisher pub;
		image_transport::Publisher edge_pub;
		image_transport::Publisher prob_pub;
		ros::Publisher hand_pub;
		
		ParticleFilter *pf1;
		ParticleFilter *pf2;
		
		cv::Mat m1_pca, m2_pca, h1_pca, h2_pca;
				
		void callback(const sensor_msgs::ImageConstPtr& immsg, const measurementproposals::HFPose2DArrayConstPtr& msg);
		
		message_filters::TimeSynchronizer<sensor_msgs::Image, measurementproposals::HFPose2DArray>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<measurementproposals::HFPose2DArray> pose_sub;	
		
		cv::Mat rpy(double roll, double pitch, double yaw);
		cv::Mat get3Dpose(cv::Mat estimate);
		cv::Mat associateHands(const measurementproposals::HFPose2DArrayConstPtr& msg);
		
		void getProbImage(cv::Mat e1, cv::Mat e2);
		void publishTFtree(cv::Mat e1, cv::Mat e2);
		void publish2Dpos(cv::Mat e1, cv::Mat e2, const measurementproposals::HFPose2DArrayConstPtr& msg);
		void update(const measurementproposals::HFPose2DArrayConstPtr& msg, cv::Mat image);
		cv::Mat knnProb(cv::Mat m1, cv::Mat m2, cv::Mat measurements,int K);
		
		
		int d, numParticles;
};

#endif
