#include "pfPose.h"

using namespace cv;
using namespace std;
using namespace message_filters;

PFTracker::PFTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/poseImage",10); //ROS
	arm_pub = it.advertise("/armImage",10); //ROS
	hand_pub = nh.advertise<handBlobTracker::HFPose2DArray>("/correctedFaceHandPose", 10);
		
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	pose_sub.subscribe(nh, "/faceHandPose", 1); // requires face array input
	
	//sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(100),image_sub, pose_sub);
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, handBlobTracker::HFPose2DArray>(image_sub,pose_sub,10);
	sync->registerCallback(boost::bind(&PFTracker::callback, this, _1, _2));
	
	// Load Kinect GMM priors
	std::stringstream ss1;
	ss1 << ros::package::getPath("handBlobTracker") << "/data13D.yml";//"/kinectData/data13D.yml";
	std::stringstream ss2;
	ss2 << ros::package::getPath("handBlobTracker") << "/data23D.yml";//"/kinectData/data23D.yml";
	cv::FileStorage fs1(ss1.str(), FileStorage::READ);
	cv::FileStorage fs2(ss2.str(), FileStorage::READ);
    fs1["means"] >> means1;
    fs2["means"] >> means2;
  	fs1["covs"] >> covs1;
	fs2["covs"] >> covs2;
	fs1["weights"] >> weights1;
    fs2["weights"] >> weights2;
    fs1.release();
    fs2.release();
    
	//int N = 2500;
	pf1 = new ParticleFilter(covs1.cols);//(N,8,0); // left arm pf
	pf2 = new ParticleFilter(covs2.cols);//(N,8,1); // right arm pf
   
	for (int i = 0; i < means1.rows; i++)
	{
		pf2->gmm.loadGaussian(means1.row(i),covs1(Range(covs1.cols*i,covs1.cols*(i+1)),Range(0,covs1.cols)),weights1.at<double>(0,i));
		pf1->gmm.loadGaussian(means2.row(i),covs2(Range(covs2.cols*i,covs2.cols*(i+1)),Range(0,covs2.cols)),weights2.at<double>(0,i));
	}
	
	trackCount=0;
}

PFTracker::~PFTracker()
{
	delete sync;
	delete pf1;
	delete pf2;
}

// Create Rotation matrix from roll, pitch and yaw
cv::Mat PFTracker::rpy(double roll, double pitch, double yaw)
{
	cv::Mat R1 = (Mat_<double>(3, 3) << 1, 0, 0, 0, cos(roll), -sin(roll), 0, sin(roll), cos(roll));
	cv::Mat R2 = (Mat_<double>(3, 3) << cos(pitch), 0, sin(pitch), 0, 1, 0, -sin(pitch), 0, cos(pitch));
	cv::Mat R3 = (Mat_<double>(3, 3) << cos(yaw), -sin(yaw), 0, sin(yaw), cos(yaw), 0, 0, 0, 1);

	return  R3*R2*R1;
}

cv::Mat PFTracker::get3Dpose(cv::Mat estimate)
{
	cv::Mat est;
	estimate.copyTo(est);
	// Webcam
	//~ cv::Mat K  = (Mat_<double>(3, 3) << 660.326889, 0.0, 318.705890, 0.0, 660.857176, 240.784699, 0.0, 0.0, 1.0);
	
	// Kinect
	cv::Mat K  = (Mat_<double>(3, 3) << 525.0, 0.0, 319.5, 0.0, 525.0, 239.5, 0.0, 0.0, 1.0);
	
		// h       e     s       h       n        r  p  y   tx ty tz 
		// 0 1 2  3 4 5  6 7 8 9 10 11  12 13 14  15 16 17  18 19 20
	cv::Mat R = rpy(est.at<double>(0,16),est.at<double>(0,17),est.at<double>(0,15));
	cv::Mat t = (Mat_<double>(3, 1) << est.at<double>(0,18), est.at<double>(0,19), est.at<double>(0,20));
	
	cv::Mat P,T;
	hconcat(R, t, T);
	P = K*T;
	
	cv::Mat im_points(5,3,CV_64FC1);
	cv::Mat temp;
	for (int k = 0; k < 5; k++)
	{
		est.at<double>(0,3*k) = est.at<double>(0,3*k)*est.at<double>(0,3*k+2);
		est.at<double>(0,3*k+1) = est.at<double>(0,3*k+1)*est.at<double>(0,3*k+2);
		temp = (est.rowRange(Range(3*k,3*k+3))).t();
		temp.copyTo(im_points(Range(k,k+1),Range(0,3)));
	}
	
	cv::Mat P_I;
	invert(P(Range(0,3),Range(0,3)),P_I,DECOMP_LU);
	
	cv::Mat pos3D = P_I*(im_points-repeat(P.col(3).t(),5,1)).t();
	//cout << std::endl << pos3D << std::endl;
	return pos3D;
}



cv::Mat PFTracker::associateHands(const handBlobTracker::HFPose2DArrayConstPtr& msg)
{
	cv::Mat pt1 = (cv::Mat_<double>(2,1) << msg->measurements[0].x, msg->measurements[0].y);
	cv::Mat pt2 = (cv::Mat_<double>(2,1) << msg->measurements[1].x, msg->measurements[1].y);
	double p1 = pf1->getHandLikelihood(pt1)*pf2->getHandLikelihood(pt2);
	double p2 = pf1->getHandLikelihood(pt2)*pf2->getHandLikelihood(pt1);
	
	ROS_INFO("Probs %f %f",p1,p2);
	
	cv::Mat pt(4,1,CV_64F);	
	if (p1 > p2)
	{
		 pt1.copyTo(pt.rowRange(Range(0,2)));
		 pt2.copyTo(pt.rowRange(Range(2,4)));
	}
	else
	{
		pt2.copyTo(pt.rowRange(Range(0,2)));
		pt1.copyTo(pt.rowRange(Range(2,4)));
	}
	return pt;
}


// TODO: Fix tracking bug to reset particle filter if lost for a while

// perform particle filter update to estimate upper body joint positions
void PFTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const handBlobTracker::HFPose2DArrayConstPtr& msg)
{
	cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
	cv::Mat or_image;
	image.copyTo(or_image);
		
	cv::Mat measurement1(6,1,CV_64F);
	cv::Mat measurement2(6,1,CV_64F);
		
	if ((msg->valid[1])&&(msg->valid[0]))
	{
	
		trackCount++;
		
		cv::Mat hands = associateHands(msg);
		
		measurement1.at<double>(0,0) = msg->measurements[2].x;
		measurement1.at<double>(1,0) = msg->measurements[2].y;
		measurement1.at<double>(2,0) = hands.at<double>(0,0);
		measurement1.at<double>(3,0) = hands.at<double>(1,0);
		measurement1.at<double>(4,0) = msg->measurements[3].x;
		measurement1.at<double>(5,0) = msg->measurements[3].y;
		pf1->update(measurement1); // particle filter measurement left arm
	
		measurement2.at<double>(0,0) = msg->measurements[2].x;
		measurement2.at<double>(1,0) = msg->measurements[2].y;
		measurement2.at<double>(2,0) = hands.at<double>(2,0);
		measurement2.at<double>(3,0) = hands.at<double>(3,0);
		measurement2.at<double>(4,0) = msg->measurements[3].x;
		measurement2.at<double>(5,0) = msg->measurements[3].y;
		pf2->update(measurement2); // particle filter measurement right arm
	
		if (trackCount > 3)
		{
			cv::Mat e1 = pf1->getEstimator(); // Weighted average pose estimate
			cv::Mat e2 = pf2->getEstimator();
		
			cv::Mat p3D1 = get3Dpose(e1);
			cv::Mat p3D2 = get3Dpose(e2);
			
			std::string strArr2[] = {"Left_Hand", "Left_Elbow", "Left_Shoulder"};
			std::string strArr1[] = {"Right_Hand", "Right_Elbow", "Right_Shoulder"};
			static tf::TransformBroadcaster br;
			tf::Transform transform;
			for (int k = 0; k < 2; k++)
			{
				
				transform.setOrigin(tf::Vector3(p3D1.at<double>(0,k) - p3D1.at<double>(0,k+1), p3D1.at<double>(2,k) - p3D1.at<double>(2,k+1), -p3D1.at<double>(1,k)+p3D1.at<double>(1,k+1)));
				transform.setRotation(tf::Quaternion(0, 0, 0));
				br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), strArr1[k+1].c_str(), strArr1[k].c_str()));
				transform.setOrigin(tf::Vector3(p3D2.at<double>(0,k) - p3D2.at<double>(0,k+1), p3D2.at<double>(2,k) - p3D2.at<double>(2,k+1), -p3D2.at<double>(1,k)+p3D2.at<double>(1,k+1)));
				transform.setRotation(tf::Quaternion(0, 0, 0));
				br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), strArr2[k+1].c_str(), strArr2[k].c_str()));
			}
			double neck_x = (p3D1.at<double>(0,4) + p3D2.at<double>(0,4))/2.0;
			double neck_y = (p3D1.at<double>(1,4) + p3D2.at<double>(1,4))/2.0;
			double neck_z = (p3D1.at<double>(2,4) + p3D2.at<double>(2,4))/2.0;
			double head_x = (p3D1.at<double>(0,3) + p3D2.at<double>(0,3))/2.0;
			double head_y = (p3D1.at<double>(1,3) + p3D2.at<double>(1,3))/2.0;
			double head_z = (p3D1.at<double>(2,3) + p3D2.at<double>(2,3))/2.0;
			transform.setOrigin(tf::Vector3(p3D1.at<double>(0,2) - neck_x, p3D1.at<double>(2,2) - neck_z, -p3D1.at<double>(1,2) + neck_y));
			transform.setRotation(tf::Quaternion(0, 0, 0));
			br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "Neck", strArr1[2].c_str()));
			transform.setOrigin(tf::Vector3(p3D2.at<double>(0,2) - neck_x, p3D2.at<double>(2,2) - neck_z, -p3D2.at<double>(1,2) + neck_y));
			transform.setRotation(tf::Quaternion(0, 0, 0));
			br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "Neck", strArr2[2].c_str()));
			transform.setOrigin(tf::Vector3(neck_x - head_x, neck_z - head_z, -neck_y + head_y));
			transform.setRotation(tf::Quaternion(0, 0, 0));
			br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "Head", "Neck"));
			transform.setOrigin(tf::Vector3(head_x, head_z, -head_y));
			//~ transform.setOrigin(tf::Vector3(0, 0, 0));
			//~ transform.setRotation(tf::Quaternion(0, -M_PI/2.0, -M_PI));
			transform.setRotation(tf::Quaternion(0, 0, 0));
			//~ br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "head_1", "Head"));
			br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "Head"));
			
			transform.setOrigin(tf::Vector3(-e1.at<double>(0,18), -e1.at<double>(0,20), e1.at<double>(0,19)));
			transform.setRotation(tf::Quaternion(-e1.at<double>(0,16),-e1.at<double>(0,17),-e1.at<double>(0,15)));
			br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cam"));
			
			cv::Mat roiMask = cv::Mat::zeros(image.rows,image.cols,CV_8UC3);
			int i;
			int col[3] = {0, 125, 255};
			for (i = 0; i < 2; i++)
			{
				line(image, Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1)), Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)), Scalar(col[i], 255, col[2-i]), 5, 8,0);
				double ang = atan2(e1.at<double>(0,3*(i+1)+1)-e1.at<double>(0,3*i+1),e1.at<double>(0,3*(i+1))-e1.at<double>(0,3*i))*180.0/M_PI;
				double d = sqrt(pow(e1.at<double>(0,3*(i+1)+1)-e1.at<double>(0,3*i+1),2) + pow(e1.at<double>(0,3*(i+1))-e1.at<double>(0,3*i),2));
				cv::RotatedRect ell =cv::RotatedRect((Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1))+Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)))*0.5,cv::Size(d,d/3.0),ang);
				try
				{
					ellipse(roiMask, ell, Scalar(255,255,255), -1, 8);
					ellipse(image, ell, Scalar(col[i], 255, col[2-i]), 1, 8);
				}
				catch( cv::Exception& e )
				{
					const char* err_msg = e.what();
					ROS_ERROR("%s",err_msg);
				}	
				
				line(image, Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1)), Point(e2.at<double>(0,3*(i+1)),e2.at<double>(0,3*(i+1)+1)), Scalar(col[i], 255, col[2-i]), 5, 8,0);
				ang = atan2(e2.at<double>(0,3*(i+1)+1)-e2.at<double>(0,3*i+1),e2.at<double>(0,3*(i+1))-e2.at<double>(0,3*i))*180.0/M_PI;
				d = sqrt(pow(e2.at<double>(0,3*(i+1)+1)-e2.at<double>(0,3*i+1),2) + pow(e2.at<double>(0,3*(i+1))-e2.at<double>(0,3*i),2));
				ell =cv::RotatedRect((Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1))+Point(e2.at<double>(0,3*(i+1)),e2.at<double>(0,3*(i+1)+1)))*0.5,cv::Size(d,d/3.0),ang);
				try
				{
					ellipse(roiMask, ell, Scalar(255,255,255), -1, 8);
					ellipse(image, ell, Scalar(col[i], 255, col[2-i]), 1, 8);
				}
				catch( cv::Exception& e )
				{
					const char* err_msg = e.what();
					ROS_ERROR("%s",err_msg);
				}	
			}
			line(image, Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1)), Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1)), Scalar(0, 255, 0), 5, 8,0);
			line(image, Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)), Point(e1.at<double>(0,3*(i+2)),e1.at<double>(0,3*(i+2)+1)), Scalar(255, 0, 0), 5, 8,0);
			
				
			// h       e     s       h       n
			// 0 1 2  3 4 5  6 7 8 9 10 11  12 13 14
		
			handBlobTracker::HFPose2D rosHands;
			handBlobTracker::HFPose2DArray rosHandsArr;
			rosHands.x = e1.at<double>(0,0);
			rosHands.y = e1.at<double>(0,1);
			rosHandsArr.names.push_back("Left Hand");
			rosHandsArr.measurements.push_back(rosHands); //hand1
			rosHands.x = e2.at<double>(0,0);
			rosHands.y = e2.at<double>(0,1);
			rosHandsArr.names.push_back("Right Hand");
			rosHandsArr.measurements.push_back(rosHands); //hand2
			rosHands.x = 0.5*(e1.at<double>(0,9) + e2.at<double>(0,9));
			rosHands.y = 0.5*(e1.at<double>(0,10) + e2.at<double>(0,10));
			rosHandsArr.names.push_back("Head");
			rosHandsArr.measurements.push_back(rosHands); //head
			rosHands.x = 0.5*(e1.at<double>(0,12) + e2.at<double>(0,12));
			rosHands.y = 0.5*(e1.at<double>(0,13) + e2.at<double>(0,13));
			rosHandsArr.names.push_back("Neck");
			rosHandsArr.measurements.push_back(rosHands); //Neck
			rosHands.x = e1.at<double>(0,3);
			rosHands.y = e1.at<double>(0,4);
			rosHandsArr.names.push_back("Left Elbow");
			rosHandsArr.measurements.push_back(rosHands); //elbow1
			rosHands.x = e2.at<double>(0,3);
			rosHands.y = e2.at<double>(0,4);
			rosHandsArr.names.push_back("Right Elbow");
			rosHandsArr.measurements.push_back(rosHands); //elbow2
			rosHands.x = e1.at<double>(0,6);
			rosHands.y = e1.at<double>(0,7);
			rosHandsArr.names.push_back("Left Shoulder");
			rosHandsArr.measurements.push_back(rosHands); //Shoulder1
			rosHands.x = e2.at<double>(0,6);
			rosHands.y = e2.at<double>(0,7);
			rosHandsArr.names.push_back("Right Shoulder");
			rosHandsArr.measurements.push_back(rosHands); //Shoulder2
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.valid.push_back(true);
			rosHandsArr.header = msg->header;
			rosHandsArr.id = msg->id;
			hand_pub.publish(rosHandsArr);
			
					
			cv_bridge::CvImage img_arm;
			img_arm.header = immsg->header;
			img_arm.encoding = "mono8";
			
			Canny(or_image, img_arm.image, 50, 100, 3 );
			arm_pub.publish(img_arm.toImageMsg()); // publish result image	
		}
		else
		{
			handBlobTracker::HFPose2D rosHands;
			handBlobTracker::HFPose2DArray rosHandsArr;
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Left Hand");
			rosHandsArr.measurements.push_back(rosHands); //hand1
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Right Hand");
			rosHandsArr.measurements.push_back(rosHands); //hand2
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Head");
			rosHandsArr.measurements.push_back(rosHands); //head
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Neck");
			rosHandsArr.measurements.push_back(rosHands); //Neck
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Left Elbow");
			rosHandsArr.measurements.push_back(rosHands); //elbow1
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Right Elbow");
			rosHandsArr.measurements.push_back(rosHands); //elbow2
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Left Shoulder");
			rosHandsArr.measurements.push_back(rosHands); //Shoulder1
			rosHands.x = 0;
			rosHands.y = 0;
			rosHandsArr.names.push_back("Right Shoulder");
			rosHandsArr.measurements.push_back(rosHands); //Shoulder2
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.valid.push_back(false);
			rosHandsArr.header = msg->header;
			rosHandsArr.id = "0";
			hand_pub.publish(rosHandsArr);
		}
	}
	else
	{
		ROS_DEBUG("Resetting trackers");
		//TODO: reset trackers in class
		delete pf1;
		delete pf2;
		pf1 = new ParticleFilter(covs1.cols);//(N,8,0); // left arm pf
		pf2 = new ParticleFilter(covs2.cols);//(N,8,1); // right arm pf
	   
		for (int i = 0; i < means1.rows; i++)
		{
			pf2->gmm.loadGaussian(means1.row(i),covs1(Range(covs1.cols*i,covs1.cols*(i+1)),Range(0,covs1.cols)),weights1.at<double>(0,i));
			pf1->gmm.loadGaussian(means2.row(i),covs2(Range(covs2.cols*i,covs2.cols*(i+1)),Range(0,covs2.cols)),weights2.at<double>(0,i));
		}
		trackCount = 0;
		
		handBlobTracker::HFPose2D rosHands;
		handBlobTracker::HFPose2DArray rosHandsArr;
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Left Hand");
		rosHandsArr.measurements.push_back(rosHands); //hand1
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Right Hand");
		rosHandsArr.measurements.push_back(rosHands); //hand2
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Head");
		rosHandsArr.measurements.push_back(rosHands); //head
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Neck");
		rosHandsArr.measurements.push_back(rosHands); //Neck
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Left Elbow");
		rosHandsArr.measurements.push_back(rosHands); //elbow1
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Right Elbow");
		rosHandsArr.measurements.push_back(rosHands); //elbow2
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Left Shoulder");
		rosHandsArr.measurements.push_back(rosHands); //Shoulder1
		rosHands.x = 0;
		rosHands.y = 0;
		rosHandsArr.names.push_back("Right Shoulder");
		rosHandsArr.measurements.push_back(rosHands); //Shoulder2
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.valid.push_back(false);
		rosHandsArr.header = msg->header;
		rosHandsArr.id = "0";
		hand_pub.publish(rosHandsArr);
	}
	
	cv_bridge::CvImage img2;
	img2.header = immsg->header;
	img2.encoding = "rgb8";
	img2.image = image;			
	pub.publish(img2.toImageMsg()); // publish result image
		
}
