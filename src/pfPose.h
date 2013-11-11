#ifndef __PFTRACKER
#define __PFTRACKER

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/objdetect/objdetect.hpp"
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
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>
//~ #include "pf2D.h"
#include "pf2DRao.h"
#include <sstream>
#include <string>
#include <ros/package.h>
#include "opencv2/ml/ml.hpp"
#include "handBlobTracker/HFPose2D.h"
#include "handBlobTracker/HFPose2DArray.h"
#include <tf/transform_broadcaster.h>

class PFTracker
{
	public:
		PFTracker();
		~PFTracker();
				
	private:
		ParticleFilter *pf1;
		ros::NodeHandle nh;
		image_transport::Publisher pub;
		ros::Publisher hand_pub;
		ParticleFilter *pf2;
		
		void callback(const sensor_msgs::ImageConstPtr& immsg, const handBlobTracker::HFPose2DArrayConstPtr& msg); // Detected face array/ image callback
		typedef message_filters::sync_policies::ApproximateTime<sensor_msgs::Image, handBlobTracker::HFPose2DArray> MySyncPolicy; // synchronising image and face array
		message_filters::Synchronizer<MySyncPolicy>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<handBlobTracker::HFPose2DArray> pose_sub;	
		cv::Mat means1, weights1;
		cv::Mat means2, weights2;
		cv::Mat covs1;
		cv::Mat covs2;
		
		cv::Mat rpy(double roll, double pitch, double yaw);
		cv::Mat get3Dpose(cv::Mat estimate);
};

#endif
