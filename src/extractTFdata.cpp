#include <ros/ros.h>
#include <iostream>
#include <fstream>

#include <tf/transform_listener.h>

int main(int argc, char** argv){
  ros::init(argc, argv, "my_tf_listener");

  std::ofstream myfile1, myfile2;
  myfile1.open ("kinectdata.txt");
  myfile2.open ("camdata.txt");
  
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
      ROS_ERROR("%s",ex.what());
    }
    ros::spinOnce();
    rate.sleep();
  }
  myfile1.close();
  myfile2.close();
  return 0;
};
