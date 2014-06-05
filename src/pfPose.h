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
#include <message_filters/time_synchronizer.h>
#include "pf2DRao.h"
#include <sstream>
#include <string>
#include <ros/package.h>
#include "opencv2/ml/ml.hpp"
#include "handBlobTracker/HFPose2D.h"
#include "handBlobTracker/HFPose2DArray.h"
#include <tf/transform_broadcaster.h>
#include "my_gmm.h"

class PFTracker
{
	public:
		PFTracker();
		~PFTracker();
				
	private:
		ros::NodeHandle nh;
		image_transport::Publisher pub;
		image_transport::Publisher edge_pub;
		ros::Publisher hand_pub;
		
		ParticleFilter *pf1;
		ParticleFilter *pf2;
				
		void callback(const sensor_msgs::ImageConstPtr& immsg, const handBlobTracker::HFPose2DArrayConstPtr& msg);
		
		message_filters::TimeSynchronizer<sensor_msgs::Image, handBlobTracker::HFPose2DArray>* sync;
		message_filters::Subscriber<sensor_msgs::Image> image_sub;
		message_filters::Subscriber<handBlobTracker::HFPose2DArray> pose_sub;	
		
		cv::Mat rpy(double roll, double pitch, double yaw);
		cv::Mat get3Dpose(cv::Mat estimate);
		cv::Mat associateHands(const handBlobTracker::HFPose2DArrayConstPtr& msg);
		bool edgePoseCorrection(cv::Mat image4, handBlobTracker::HFPose2DArray pfPose, cv::Mat image3);
		
		double edge_heuristic;
		bool swap;
		int d;
};

#endif
