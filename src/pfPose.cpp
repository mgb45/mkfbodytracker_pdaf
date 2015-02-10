#include "pfPose.h"

using namespace cv;
using namespace std;
using namespace message_filters;

PFTracker::PFTracker()
{
	image_transport::ImageTransport it(nh);
	
	pub = it.advertise("/poseImage",1);
	prob_pub = it.advertise("/probImage",1);
	hand_pub = nh.advertise<measurementproposals::HFPose2DArray>("/correctedFaceHandPose", 10);
		
	image_sub.subscribe(nh, "/rgb/image_raw", 5);
	likelihood_sub.subscribe(nh, "/likelihood", 5);
	pose_sub.subscribe(nh, "/faceROIs", 10); 
		
	sync = new message_filters::TimeSynchronizer<sensor_msgs::Image, sensor_msgs::Image, faceTracking::ROIArray>(image_sub,likelihood_sub,pose_sub,20);
	sync->registerCallback(boost::bind(&PFTracker::callback, this, _1, _2, _3));
	
	// Load Kinect GMM priors
	std::stringstream ss1;
	std::string left_arm_training;
	ros::param::param<std::string>("left_arm_training", left_arm_training, "/data13D_PCA_100000_25_10.yml");
	ss1 << ros::package::getPath("mkfbodytracker_pdaf") << left_arm_training;
	std::stringstream ss2;
	std::string right_arm_training;
	ros::param::param<std::string>("right_arm_training", right_arm_training, "/data23D_PCA_100000_25_10.yml");
	ss2 << ros::package::getPath("mkfbodytracker_pdaf") << right_arm_training;
	ROS_INFO("Getting data from %s",ss1.str().c_str());
	ROS_INFO("Getting data from %s",ss2.str().c_str());
	cv::FileStorage fs1(ss1.str(), FileStorage::READ);
	cv::FileStorage fs2(ss2.str(), FileStorage::READ);
	
	cv::Mat means1, weights1, means2, weights2, covs1, covs2, g1, g2;
    fs1["means"] >> means1;
    fs2["means"] >> means2;
  	fs1["covs"] >> covs1;
	fs2["covs"] >> covs2;
	fs1["weights"] >> weights1;
    fs2["weights"] >> weights2;
    fs1["pca_proj"] >> h1_pca;
    h1_pca.convertTo(h1_pca, CV_64F);
    fs2["pca_proj"] >> h2_pca;
    h2_pca.convertTo(h2_pca, CV_64F);
    fs1["pca_mean"] >> m1_pca;
    m1_pca.convertTo(m1_pca, CV_64F);
    fs2["pca_mean"] >> m2_pca;
    m2_pca.convertTo(m2_pca, CV_64F);
    fs2["gamma"] >> g1;
    fs2["gamma"] >> g2;
    fs1.release();
    fs2.release();
    
    numParticles = 100;
    d = h1_pca.rows;
	pf1 = new ParticleFilter(d,numParticles); // left arm pf
	pf2 = new ParticleFilter(d,numParticles); // right arm pf

	for (int i = 0; i < means1.rows; i++)
	{
		pf2->gmm.loadGaussian(means1.row(i),covs1(Range(covs1.cols*i,covs1.cols*(i+1)),Range(0,covs1.cols)), h1_pca, m1_pca, weights1.at<double>(0,i), g1.at<double>(0,i));
		pf1->gmm.loadGaussian(means2.row(i),covs2(Range(covs2.cols*i,covs2.cols*(i+1)),Range(0,covs2.cols)), h2_pca, m2_pca, weights2.at<double>(0,i), g2.at<double>(0,i));
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
	return pos3D;
}

void PFTracker::getProbImage(cv::Mat e1, cv::Mat e2)
{
	cv_bridge::CvImage prob;
	prob.encoding = "rgb8";
	prob.image = pf1->getProbMap(h2_pca.t(), m2_pca.t());//edge_image;			
	int i;
	int col[3] = {0, 125, 255};
	for (i = 0; i < 2; i++)
	{
		line(prob.image, Point(e1.at<double>(0,3*i)/8.0,e1.at<double>(0,3*i+1)/8.0), Point(e1.at<double>(0,3*(i+1))/8.0,e1.at<double>(0,3*(i+1)+1)/8.0), Scalar(col[i], 255, col[2-i]), 1, 8,0);
		line(prob.image, Point(e2.at<double>(0,3*i)/8.0,e2.at<double>(0,3*i+1)/8.0), Point(e2.at<double>(0,3*(i+1))/8.0,e2.at<double>(0,3*(i+1)+1)/8.0), Scalar(col[i], 255, col[2-i]), 1, 8,0);
	}
	line(prob.image, Point(e1.at<double>(0,3*i)/8.0,e1.at<double>(0,3*i+1)/8.0), Point(e2.at<double>(0,3*i)/8.0,e2.at<double>(0,3*i+1)/8.0), Scalar(0, 255, 0), 1, 8,0);
	line(prob.image, Point(e1.at<double>(0,3*(i+1))/8.0,e1.at<double>(0,3*(i+1)+1)/8.0), Point(e1.at<double>(0,3*(i+2))/8.0,e1.at<double>(0,3*(i+2)+1)/8.0), Scalar(255, 0, 0), 1, 8,0);	
	prob_pub.publish(prob.toImageMsg()); // publish result image

}

void PFTracker::publishTFtree(cv::Mat e1, cv::Mat e2)
{
	cv::Mat p3D1 = get3Dpose(e1);
	cv::Mat p3D2 = get3Dpose(e2);

	// Publish tf tree for rviz
	std::string strArr2[] = {"Left_Hand", "Left_Elbow", "Left_Shoulder"};
	std::string strArr1[] = {"Right_Hand", "Right_Elbow", "Right_Shoulder"};
	static tf::TransformBroadcaster br;
	tf::Transform transform;
	tf::Quaternion no_rot_quat;
	no_rot_quat.setEuler(0.0,0.0,0.0);
	transform.setRotation(no_rot_quat);
	for (int k = 0; k < 2; k++)
	{
		transform.setOrigin(tf::Vector3(p3D1.at<double>(0,k) - p3D1.at<double>(0,k+1), p3D1.at<double>(2,k) - p3D1.at<double>(2,k+1), -p3D1.at<double>(1,k)+p3D1.at<double>(1,k+1)));
		br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), strArr1[k+1].c_str(), strArr1[k].c_str()));
		transform.setOrigin(tf::Vector3(p3D2.at<double>(0,k) - p3D2.at<double>(0,k+1), p3D2.at<double>(2,k) - p3D2.at<double>(2,k+1), -p3D2.at<double>(1,k)+p3D2.at<double>(1,k+1)));
		br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), strArr2[k+1].c_str(), strArr2[k].c_str()));
	}

	double neck_x = (p3D1.at<double>(0,4) + p3D2.at<double>(0,4))/2.0;
	double neck_y = (p3D1.at<double>(1,4) + p3D2.at<double>(1,4))/2.0;
	double neck_z = (p3D1.at<double>(2,4) + p3D2.at<double>(2,4))/2.0;
	double head_x = (p3D1.at<double>(0,3) + p3D2.at<double>(0,3))/2.0;
	double head_y = (p3D1.at<double>(1,3) + p3D2.at<double>(1,3))/2.0;
	double head_z = (p3D1.at<double>(2,3) + p3D2.at<double>(2,3))/2.0;

	transform.setOrigin(tf::Vector3(p3D1.at<double>(0,2) - neck_x, p3D1.at<double>(2,2) - neck_z, -p3D1.at<double>(1,2) + neck_y));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "Neck", strArr1[2].c_str()));
	transform.setOrigin(tf::Vector3(p3D2.at<double>(0,2) - neck_x, p3D2.at<double>(2,2) - neck_z, -p3D2.at<double>(1,2) + neck_y));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "Neck", strArr2[2].c_str()));
	transform.setOrigin(tf::Vector3(neck_x - head_x, neck_z - head_z, -neck_y + head_y));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "Head", "Neck"));
	transform.setOrigin(tf::Vector3(head_x, head_z, -head_y));
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "Head"));

	transform.setOrigin(tf::Vector3(-e1.at<double>(0,18), -e1.at<double>(0,20), e1.at<double>(0,19)));
	tf::Quaternion cam_rot;
	cam_rot.setEuler(-e1.at<double>(0,16),-e1.at<double>(0,17),-e1.at<double>(0,15));
	transform.setRotation(cam_rot);
	br.sendTransform(tf::StampedTransform(transform, ros::Time::now(), "world", "cam"));
}

void PFTracker::publish2Dpos(cv::Mat e1, cv::Mat e2,const faceTracking::ROIArrayConstPtr& msg)
{
	// Create result message
	measurementproposals::HFPose2D rosHands;
	measurementproposals::HFPose2DArray rosHandsArr;
	rosHands.x = e1.at<double>(0,0);
	rosHands.y = e1.at<double>(0,1);
	rosHandsArr.measurements.push_back(rosHands); //hand1
	rosHands.x = e2.at<double>(0,0);
	rosHands.y = e2.at<double>(0,1);
	rosHandsArr.measurements.push_back(rosHands); //hand2
	rosHands.x = 0.5*(e1.at<double>(0,9) + e2.at<double>(0,9));
	rosHands.y = 0.5*(e1.at<double>(0,10) + e2.at<double>(0,10));
	rosHandsArr.measurements.push_back(rosHands); //head
	rosHands.x = 0.5*(e1.at<double>(0,12) + e2.at<double>(0,12));
	rosHands.y = 0.5*(e1.at<double>(0,13) + e2.at<double>(0,13));
	rosHandsArr.measurements.push_back(rosHands); //Neck
	rosHands.x = e1.at<double>(0,3);
	rosHands.y = e1.at<double>(0,4);
	rosHandsArr.measurements.push_back(rosHands); //elbow1
	rosHands.x = e2.at<double>(0,3);
	rosHands.y = e2.at<double>(0,4);
	rosHandsArr.measurements.push_back(rosHands); //elbow2
	rosHands.x = e1.at<double>(0,6);
	rosHands.y = e1.at<double>(0,7);
	rosHandsArr.measurements.push_back(rosHands); //Shoulder1
	rosHands.x = e2.at<double>(0,6);
	rosHands.y = e2.at<double>(0,7);
	rosHandsArr.measurements.push_back(rosHands); //Shoulder2
		
	rosHandsArr.header = msg->header;
	rosHandsArr.id = msg->ids[0];
	hand_pub.publish(rosHandsArr);
}

void PFTracker::update(const std::vector<int> binsL, const std::vector<int> binsR, const faceTracking::ROIArrayConstPtr& msg, cv::Mat image)
{
	// Package new measurements into matrix
		
	cv::Mat measurement1(6,numParticles,CV_64F);
	cv::Mat measurement2(6,numParticles,CV_64F);
	
	for (int i = 0; i < numParticles; i++)
	{
		measurement1.at<double>(0,i) = msg->ROIs[0].x_offset + msg->ROIs[0].width/2.0;
		measurement1.at<double>(1,i) = msg->ROIs[0].y_offset + msg->ROIs[0].height/2.0;
		measurement1.at<double>(2,i) = 8*(binsL[i]%80);
		measurement1.at<double>(3,i) = 8*(binsL[i]/80);
		measurement1.at<double>(4,i) = msg->ROIs[0].x_offset + msg->ROIs[0].width/2.0;
		measurement1.at<double>(5,i) = msg->ROIs[0].y_offset + 3.65/2.0*msg->ROIs[0].height;
		circle(image, cv::Point(measurement1.at<double>(2,i),measurement1.at<double>(3,i)), 2, Scalar(255,0,0), -1, 8);
				
		measurement2.at<double>(0,i) = msg->ROIs[0].x_offset + msg->ROIs[0].width/2.0;
		measurement2.at<double>(1,i) = msg->ROIs[0].y_offset + msg->ROIs[0].height/2.0;
		measurement2.at<double>(2,i) = 8*(binsR[i]%80);
		measurement2.at<double>(3,i) = 8*(binsR[i]/80);
		measurement2.at<double>(4,i) = msg->ROIs[0].x_offset + msg->ROIs[0].width/2.0;
		measurement2.at<double>(5,i) = msg->ROIs[0].y_offset + 3.65/2.0*msg->ROIs[0].height;
		circle(image, cv::Point(measurement2.at<double>(2,i),measurement2.at<double>(3,i)), 2, Scalar(0,0,255), -1, 8);
	}
				
	pf1->update(measurement1); // particle filter measurement left arm
	pf2->update(measurement2); // particle filter measurement right arm
}	

cv::Mat PFTracker::getMeasurementProposal(cv::Mat likelihood)
{
	cv::Mat prob_image_L = pf1->getProbMap(h2_pca.t(), m2_pca.t());
	cv::Mat prob_image_R = pf2->getProbMap(h1_pca.t(), m1_pca.t());
	cv::Mat output = cv::Mat::zeros(prob_image_L.rows,prob_image_L.cols,CV_8UC3);
	
	GaussianBlur(prob_image_L, prob_image_L, cv::Size(15,15), 3, 3, BORDER_DEFAULT);
	GaussianBlur(prob_image_R, prob_image_R, cv::Size(15,15), 3, 3, BORDER_DEFAULT);
	
	cv::Mat mini_likelihood;
	resize(likelihood,mini_likelihood,prob_image_L.size(),0,0,INTER_LINEAR);
		
	prob_image_L.convertTo(prob_image_L, CV_32FC1);
    prob_image_R.convertTo(prob_image_R, CV_32FC1);
    mini_likelihood.convertTo(mini_likelihood, CV_32FC1);
    clutter.convertTo(clutter, CV_32FC1);
    
    clutter = 5*cv::Mat::ones(prob_image_L.rows,prob_image_L.cols,CV_32FC1);
			
	cv::Mat L = mini_likelihood.mul(prob_image_L);//*0.9 + prob_image_R*0.05 + 0.05*clutter);
	cv::Mat R = mini_likelihood.mul(prob_image_R);//*0.9 + prob_image_L*0.05 + 0.05*clutter);
	clutter = mini_likelihood.mul(clutter);//*0.8 + prob_image_R*0.1  + prob_image_L*0.1);
	cv::normalize(0.2*L, L, 0, 255, NORM_MINMAX, CV_8UC1);
	cv::normalize(0.2*R, R, 0, 255, NORM_MINMAX, CV_8UC1);
	cv::normalize(0.6*clutter, clutter, 0, 255, NORM_MINMAX, CV_8UC1);
			
	std::vector<cv::Mat> Im_arr;
	Im_arr.push_back(L);
	Im_arr.push_back(clutter);
	Im_arr.push_back(R);
	cv::merge(Im_arr,output);
	
	return output;
}
	
// perform particle filter update to estimate upper body joint positions
void PFTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const sensor_msgs::ImageConstPtr& like_msg, const faceTracking::ROIArrayConstPtr& msg)
{
	cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; 
	cv::Mat likelihood = (cv_bridge::toCvCopy(like_msg, sensor_msgs::image_encodings::MONO8))->image; 
	
	if (msg->ROIs.size() > 0)
	{
		cv_bridge::CvImage prob;
		prob.encoding = "rgb8";
		prob.image = getMeasurementProposal(likelihood);			
		prob_pub.publish(prob.toImageMsg()); // publish result image
		
		std::vector<cv::Mat> Im_arr;
		split(prob.image,Im_arr);
		
		Mat vecL = Im_arr[0].reshape(0,1);
		vecL.convertTo(vecL, CV_32FC1);
		
		Mat vecC = Im_arr[1].reshape(0,1);
		vecC.convertTo(vecC, CV_32FC1);
		
		Mat vecR = Im_arr[2].reshape(0,1);
		vecR.convertTo(vecR, CV_32FC1);
		
		double sum_P = sum(vecR)[0] + sum(vecL)[0] + sum(vecC)[0];
		vecR = vecR/sum_P;
		vecL = vecL/sum_P;
				
		std::vector<double> weightsL(vecL);
		std::vector<double> weightsR(vecR);
		std::vector<int> binsL = pf1->resample(weightsL, numParticles);
		std::vector<int> binsR = pf2->resample(weightsR, numParticles);
		
		update(binsL,binsR,msg,image);
	
		cv::Mat e1 = h2_pca.t()*pf1->getEstimator() + m2_pca.t(); // Weighted average pose estimate
		cv::Mat e2 = h1_pca.t()*pf2->getEstimator() + m1_pca.t();
	
		publishTFtree(e1,e2);
		publish2Dpos(e1,e2,msg);
		
		//Draw stick man on result image
		//Result vector arrangment	
		//h       e     s       h       n
		//0 1 2  3 4 5  6 7 8 9 10 11  12 13 14
		int i;
		int col[3] = {0, 125, 255};
		for (i = 0; i < 2; i++)
		{
			line(image, Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1)), Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)), Scalar(col[i], 255, col[2-i]), 5, 8,0);
			line(image, Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1)), Point(e2.at<double>(0,3*(i+1)),e2.at<double>(0,3*(i+1)+1)), Scalar(col[i], 255, col[2-i]), 5, 8,0);
		}
		line(image, Point(e1.at<double>(0,3*i),e1.at<double>(0,3*i+1)), Point(e2.at<double>(0,3*i),e2.at<double>(0,3*i+1)), Scalar(0, 255, 0), 5, 8,0);
		line(image, Point(e1.at<double>(0,3*(i+1)),e1.at<double>(0,3*(i+1)+1)), Point(e1.at<double>(0,3*(i+2)),e1.at<double>(0,3*(i+2)+1)), Scalar(255, 0, 0), 5, 8,0);
	}		
		
	cv_bridge::CvImage img2;
	img2.header = immsg->header;
	img2.encoding = "rgb8";
	img2.image = image;			
	pub.publish(img2.toImageMsg()); // publish result image	
}
