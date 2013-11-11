#include "pfPose.h"

using namespace cv;
using namespace std;

PFTracker::PFTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/poseImage",1); //ROS
	hand_pub = nh.advertise<handBlobTracker::HFPose2DArray>("/correctedFaceHandPose", 1000);
		
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	pose_sub.subscribe(nh, "/faceHandPose", 1); // requires face array input
	
	sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, pose_sub);
	sync->registerCallback(boost::bind(&PFTracker::callback, this, _1, _2));
	
	// Load Kinect GMM priors
	std::stringstream ss1;
	ss1 << ros::package::getPath("handBlobTracker") << "/data13D.yml";
	std::stringstream ss2;
	ss2 << ros::package::getPath("handBlobTracker") << "/data23D.yml";
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
	cv::Mat K  = (Mat_<double>(3, 3) << 660.326889, 0.0, 318.705890, 0.0, 660.857176, 240.784699, 0.0, 0.0, 1.0);
	
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

// TODO: Fix tracking bug to reset particle filter if lost for a while

// perform particle filter update to estimate upper body joint positions
void PFTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const handBlobTracker::HFPose2DArrayConstPtr& msg)
{
	cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
		
	cv::Mat measurement1(6,1,CV_64F);
	cv::Mat measurement2(6,1,CV_64F);
	double scale = 1.0;
	
	if ((msg->valid[1])&&(msg->valid[0]))
	{
	
		measurement1.at<double>(0,0) = msg->measurements[2].x/scale;
		measurement1.at<double>(1,0) = msg->measurements[2].y/scale;
		measurement1.at<double>(2,0) = msg->measurements[0].x/scale;
		measurement1.at<double>(3,0) = msg->measurements[0].y/scale;
		measurement1.at<double>(4,0) = msg->measurements[3].x/scale;
		measurement1.at<double>(5,0) = msg->measurements[3].y/scale;
		pf1->update(measurement1); // particle filter measurement left arm
	
		measurement2.at<double>(0,0) = msg->measurements[2].x/scale;
		measurement2.at<double>(1,0) = msg->measurements[2].y/scale;
		measurement2.at<double>(2,0) = msg->measurements[1].x/scale;
		measurement2.at<double>(3,0) = msg->measurements[1].y/scale;
		measurement2.at<double>(4,0) = msg->measurements[3].x/scale;
		measurement2.at<double>(5,0) = msg->measurements[3].y/scale;
		pf2->update(measurement2); // particle filter measurement right arm
	

		cv::Mat e1 = scale*pf1->getEstimator(); // Weighted average pose estimate
		cv::Mat e2 = scale*pf2->getEstimator();
	
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
		transform.setRotation(tf::Quaternion(0, 0, 0));
		br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "Head"));
		transform.setOrigin(tf::Vector3(-e1.at<double>(0,18), -e1.at<double>(0,20), e1.at<double>(0,19)));
		transform.setRotation(tf::Quaternion(e1.at<double>(0,16),e1.at<double>(0,17),e1.at<double>(0,15)));
		br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cam"));
		
	//~ // Draw body lines
	//~ if (msg->valid[0])
	//~ {
		//~ line(image, Point(e1.at<double>(0,0),e1.at<double>(0,1)), Point(e1.at<double>(0,2),e1.at<double>(0,3)), Scalar(255, 0, 255), 5, 8,0);
		//~ line(image, Point(e1.at<double>(0,2),e1.at<double>(0,3)), Point(e1.at<double>(0,4),e1.at<double>(0,5)), Scalar(0, 255, 255), 5, 8,0);
	//~ }
	//~ 
	//~ if (msg->valid[1])
	//~ {
		//~ //h-e
		//~ line(image, Point(e2.at<double>(0,0),e2.at<double>(0,1)), Point(e2.at<double>(0,2),e2.at<double>(0,3)), Scalar(255, 0, 255), 5, 8,0);
		//~ //E -S
		//~ line(image, Point(e2.at<double>(0,2),e2.at<double>(0,3)), Point(e2.at<double>(0,4),e2.at<double>(0,5)), Scalar(0, 255, 255), 5, 8,0);
		//~ //S-S
		//~ // S -H
	//~ }
	//~ 
	//~ if ((msg->valid[1])&&(msg->valid[0]))
	//~ {
		//~ line(image, Point(e1.at<double>(0,4),e1.at<double>(0,5)), Point(e2.at<double>(0,4),e2.at<double>(0,5)), Scalar(255, 255, 0), 5, 8,0);
		//~ line(image, Point((e2.at<double>(0,4) +e1.at<double>(0,4))/2,(e2.at<double>(0,5) +e1.at<double>(0,5))/2), Point(msg->measurements[2].x,msg->measurements[2].y), Scalar(255, 255,0), 5, 8,0);
	//~ }
	
	
		//~ cout << "Estimate:" << std::endl << e1;
		//~ cout << std::endl << e2;
		int i;
		int col[3] = {0, 125, 255};
		for (i = 0; i < 2; i++)
		{
			line(image, Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1)), Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)), Scalar(col[i], 255, col[2-i]), 5, 8,0);
			line(image, Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1)), Point(e2.at<double>(0,3*(i+1)),e2.at<double>(0,3*(i+1)+1)), Scalar(col[i], 255, col[2-i]), 5, 8,0);
		}
		line(image, Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1)), Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1)), Scalar(0, 255, 0), 5, 8,0);
		line(image, Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)), Point(e1.at<double>(0,3*(i+2)),e1.at<double>(0,3*(i+2)+1)), Scalar(255, 0, 0), 5, 8,0);
		
		ROS_INFO("Publishing corrections");
		
		// h       e     s       h       n
		// 0 1 2  3 4 5  6 7 8 9 10 11  12 13 14
		handBlobTracker::HFPose2D rosHands;
		handBlobTracker::HFPose2DArray rosHandsArr;
		rosHands.x = e1.at<double>(0,0);
		rosHands.y = e1.at<double>(0,1);
		rosHandsArr.measurements.push_back(rosHands);
		rosHands.x = e2.at<double>(0,0);
		rosHands.y = e2.at<double>(0,1);
		rosHandsArr.measurements.push_back(rosHands);
		rosHands.x = e1.at<double>(0,9);
		rosHands.y = e1.at<double>(0,10);;
		rosHandsArr.measurements.push_back(rosHands);
		rosHands.x = e1.at<double>(0,12);
		rosHands.y = e1.at<double>(0,13);
		rosHandsArr.measurements.push_back(rosHands); //Neck
		rosHandsArr.valid.push_back(true);
		rosHandsArr.valid.push_back(true);
		rosHandsArr.valid.push_back(true);
		rosHandsArr.valid.push_back(true);
		rosHandsArr.header = msg->header;
		rosHandsArr.id = msg->id;
		hand_pub.publish(rosHandsArr);
		

	}
	else
	{
		ROS_INFO("Resetting trackers");
		delete pf1;
		delete pf2;
		pf1 = new ParticleFilter(covs1.cols);//(N,8,0); // left arm pf
		pf2 = new ParticleFilter(covs2.cols);//(N,8,1); // right arm pf
	   
		for (int i = 0; i < means1.rows; i++)
		{
			pf2->gmm.loadGaussian(means1.row(i),covs1(Range(covs1.cols*i,covs1.cols*(i+1)),Range(0,covs1.cols)),weights1.at<double>(0,i));
			pf1->gmm.loadGaussian(means2.row(i),covs2(Range(covs2.cols*i,covs2.cols*(i+1)),Range(0,covs2.cols)),weights2.at<double>(0,i));
		}
	}
	
	cv_bridge::CvImage img2;
	img2.encoding = "rgb8";
	img2.image = image;			
	pub.publish(img2.toImageMsg()); // publish result image
		
}
