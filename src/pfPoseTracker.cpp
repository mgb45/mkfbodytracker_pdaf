/* Body tracking using a particle filter and kinect upper body priors */
/* M. Burke*/
#include "pfPose.h"

int main( int argc, char** argv )
{
	ros::init(argc, argv, "poseTracking");
	
	PFTracker *tracker = new PFTracker();
	
	ros::spin();
	delete tracker;	
	return 0;
}
