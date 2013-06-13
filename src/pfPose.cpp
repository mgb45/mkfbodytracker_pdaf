#include "pfPose.h"

using namespace cv;
using namespace std;

PFTracker::PFTracker()
{
	image_transport::ImageTransport it(nh); //ROS
	
	pub = it.advertise("/poseImage",1); //ROS
		
	image_sub.subscribe(nh, "/rgb/image_color", 1); // requires camera stream input
	pose_sub.subscribe(nh, "/faceHandPose", 1); // requires face array input
	
	sync = new message_filters::Synchronizer<MySyncPolicy>(MySyncPolicy(10),image_sub, pose_sub);
	sync->registerCallback(boost::bind(&PFTracker::callback, this, _1, _2));
	
	int N = 2500;
	pf1 = new ParticleFilter();//(N,8,0); // left arm pf
	pf2 = new ParticleFilter();//(N,8,1); // right arm pf
	
	// Load Kinect GMM priors
	std::stringstream ss1;
	ss1 << ros::package::getPath("handBlobTracker") << "/data1.yml";
	std::stringstream ss2;
	ss2 << ros::package::getPath("handBlobTracker") << "/data2.yml";
	cv::FileStorage fs1(ss1.str(), FileStorage::READ);
	cv::FileStorage fs2(ss2.str(), FileStorage::READ);
    cv::Mat means1, weights1;
    cv::Mat means2, weights2;
    cv::Mat covs1;
    cv::Mat covs2;
    fs1["means"] >> means1;
    fs2["means"] >> means2;
  	fs1["covs"] >> covs1;
	fs2["covs"] >> covs2;
	fs1["weights"] >> weights1;
    fs2["weights"] >> weights2;
    fs1.release();
    fs2.release();
   
	for (int i = 0; i < 8; i++)
	{
		pf2->gmm.loadGaussian(means1.row(i),covs1(Range(8*i,8*(i+1)),Range(0,8)),weights1.at<double>(0,i));
		pf1->gmm.loadGaussian(means2.row(i),covs2(Range(8*i,8*(i+1)),Range(0,8)),weights2.at<double>(0,i));
	}
	
}

PFTracker::~PFTracker()
{
	delete sync;
}

// perform particle filter update to estimate upper body joint positions
void PFTracker::callback(const sensor_msgs::ImageConstPtr& immsg, const handBlobTracker::HFPose2DArrayConstPtr& msg)
{
	cv::Mat image = (cv_bridge::toCvCopy(immsg, sensor_msgs::image_encodings::RGB8))->image; //ROS
		
	cv::Mat measurement1(4,1,CV_64F);
	
	double scale = 1.0;
	
	measurement1.at<double>(0,0) = msg->measurements[2].x/scale;
	measurement1.at<double>(1,0) = msg->measurements[2].y/scale;
	measurement1.at<double>(2,0) = msg->measurements[0].x/scale;
	measurement1.at<double>(3,0) = msg->measurements[0].y/scale;
	pf1->update(measurement1); // particle filter measurement left arm

	cv::Mat measurement2(4,1,CV_64F);
	measurement2.at<double>(0,0) = msg->measurements[2].x/scale;
	measurement2.at<double>(1,0) = msg->measurements[2].y/scale;
	measurement2.at<double>(2,0) = msg->measurements[1].x/scale;
	measurement2.at<double>(3,0) = msg->measurements[1].y/scale;
	pf2->update(measurement2); // particle filter measurement right arm

	cv::Mat e1 = scale*pf1->getEstimator(); // Weighted average pose estimate
	cv::Mat e2 = scale*pf2->getEstimator();
	
	// Draw body lines
	//h-e
	line(image, Point(e1.at<double>(0,0),e1.at<double>(0,1)), Point(e1.at<double>(0,2),e1.at<double>(0,3)), Scalar(255, 0, 255), 5, 8,0);
	line(image, Point(e2.at<double>(0,0),e2.at<double>(0,1)), Point(e2.at<double>(0,2),e2.at<double>(0,3)), Scalar(255, 0, 255), 5, 8,0);
	//E -S
	line(image, Point(e1.at<double>(0,2),e1.at<double>(0,3)), Point(e1.at<double>(0,4),e1.at<double>(0,5)), Scalar(0, 255, 255), 5, 8,0);
	line(image, Point(e2.at<double>(0,2),e2.at<double>(0,3)), Point(e2.at<double>(0,4),e2.at<double>(0,5)), Scalar(0, 255, 255), 5, 8,0);
	//S-S
	line(image, Point(e1.at<double>(0,4),e1.at<double>(0,5)), Point(e2.at<double>(0,4),e2.at<double>(0,5)), Scalar(255, 255, 0), 5, 8,0);
	// S -H
	line(image, Point((e2.at<double>(0,4) +e1.at<double>(0,4))/2,(e2.at<double>(0,5) +e1.at<double>(0,5))/2), Point(msg->measurements[2].x,msg->measurements[2].y), Scalar(255, 255,0), 5, 8,0);
	
	cv_bridge::CvImage img2;
	img2.encoding = "rgb8";
	img2.image = image;			
	pub.publish(img2.toImageMsg()); // publish result image
	
}
