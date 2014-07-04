#include <ros/ros.h>
#include <iostream>
#include <fstream>
#include <ros/package.h>
#include <tf/transform_listener.h>

int main(int argc, char** argv)
{
	ros::init(argc, argv, "my_tf_listener");

	std::string left_arm_training;
	ros::param::param<std::string>("left_arm_training", left_arm_training, "/data13D.yml");
	
	std::ofstream myfile1, myfile2;
	time_t current_time = time(0);
	struct tm * now = localtime( & current_time );
	char buffer [80];
	strftime (buffer,80,"%Y-%m-%d-%H-%M-%S",now);
	std::stringstream path_out1;
	std::stringstream path_out2;
	path_out1 << ros::package::getPath("mkfbodytracker") << "/kinectdata_" << buffer << ".txt";
	std::string path1 = path_out1.str();
	path_out2 << ros::package::getPath("mkfbodytracker") << "/camdata_" << buffer << ".txt";
	std::string path2 = path_out2.str();
	
	std::cout << "Saving data to: " << path1.c_str() << std::endl;
	std::cout << "Saving data to: " << path2.c_str() << std::endl;

	myfile1.open (path1.c_str());
	myfile2.open (path2.c_str());
	myfile1 << "%" << left_arm_training << std::endl; 
	myfile2 << "%" << left_arm_training << std::endl; 
	
	ros::NodeHandle node;

	tf::TransformListener listener;

	ros::Rate rate(30);
	while (node.ok())
	{
		tf::StampedTransform transform;
		geometry_msgs::TransformStamped Head, RHand, LHand, RElbow, LElbow, RShoulder, LShoulder, Torso;
		try
		{
			listener.lookupTransform("/Head", "/head_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, Head);
			listener.lookupTransform("/Head", "/left_hand_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, LHand);
			listener.lookupTransform("/Head", "/right_hand_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, RHand);
			listener.lookupTransform("/Head", "/left_elbow_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, LElbow);
			listener.lookupTransform("/Head", "/right_elbow_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, RElbow);
			listener.lookupTransform("/Head", "/left_shoulder_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, LShoulder);
			listener.lookupTransform("/Head", "/right_shoulder_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, RShoulder);
			listener.lookupTransform("/Head", "/neck_1", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, Torso);
			
			myfile1 << " " << Head.transform.translation.x << " " << Head.transform.translation.y << " " << Head.transform.translation.z;
			myfile1 << " " << LHand.transform.translation.x << " " << LHand.transform.translation.y << " " << LHand.transform.translation.z;
			myfile1 << " " << RHand.transform.translation.x << " " << RHand.transform.translation.y << " " << RHand.transform.translation.z;
			myfile1 << " " << LElbow.transform.translation.x << " " << LElbow.transform.translation.y << " " << LElbow.transform.translation.z;
			myfile1 << " " << RElbow.transform.translation.x << " " << RElbow.transform.translation.y << " " << RElbow.transform.translation.z;
			myfile1 << " " << RShoulder.transform.translation.x << " " << RShoulder.transform.translation.y << " " << RShoulder.transform.translation.z;
			myfile1 << " " << LShoulder.transform.translation.x << " " << LShoulder.transform.translation.y << " " << LShoulder.transform.translation.z;
			myfile1 << " " << Torso.transform.translation.x << " " << Torso.transform.translation.y << " " << Torso.transform.translation.z;
			myfile1 << std::endl;
			
			listener.lookupTransform("/Head", "/Head", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, Head);
			listener.lookupTransform("/Head", "/Left_Hand", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, LHand);
			listener.lookupTransform("/Head", "/Right_Hand", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, RHand);
			listener.lookupTransform("/Head", "/Left_Elbow", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, LElbow);
			listener.lookupTransform("/Head", "/Right_Elbow", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, RElbow);
			listener.lookupTransform("/Head", "/Left_Shoulder", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, LShoulder);
			listener.lookupTransform("/Head", "/Right_Shoulder", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, RShoulder);
			listener.lookupTransform("/Head", "/Neck", ros::Time(0), transform);
			tf::transformStampedTFToMsg(transform, Torso);
			
			myfile2 << " " << Head.transform.translation.x << " " << Head.transform.translation.y << " " << Head.transform.translation.z;
			myfile2 << " " << LHand.transform.translation.x << " " << LHand.transform.translation.y << " " << LHand.transform.translation.z;
			myfile2 << " " << RHand.transform.translation.x << " " << RHand.transform.translation.y << " " << RHand.transform.translation.z;
			myfile2 << " " << LElbow.transform.translation.x << " " << LElbow.transform.translation.y << " " << LElbow.transform.translation.z;
			myfile2 << " " << RElbow.transform.translation.x << " " << RElbow.transform.translation.y << " " << RElbow.transform.translation.z;
			myfile2 << " " << RShoulder.transform.translation.x << " " << RShoulder.transform.translation.y << " " << RShoulder.transform.translation.z;
			myfile2 << " " << LShoulder.transform.translation.x << " " << LShoulder.transform.translation.y << " " << LShoulder.transform.translation.z;
			myfile2 << " " << Torso.transform.translation.x << " " << Torso.transform.translation.y << " " << Torso.transform.translation.z;
			myfile2 << std::endl;
		}
		catch (tf::TransformException ex)
		{
			//ROS_ERROR("%s",ex.what());
		}
		ros::spinOnce();
		rate.sleep();
	}
	myfile1.close();
	myfile2.close();
	return 0;
};
