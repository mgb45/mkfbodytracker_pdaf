This is very much research code, but the following should get a test case up and running.

1.) Install ROS: http://wiki.ros.org/kinetic/Installation/Ubuntu

2.) 
	mkdir pose_estimation
	cd pose_estimation/
	git clone https://github.com/mgb45/handblobtracker.git
	git clone -b KF_proposals https://github.com/mgb45/facetracking.git
	git clone https://github.com/mgb45/mkfbodytracker_pdaf.git
	git clone https://github.com/mgb45/measurementproposals.git
	rosws init . /opt/ros/kinetic
	source setup.bash
	rosws set handblobtracker/
	rosws set facetracking/
	rosws set mkfbodytracker_pdaf/
	rosws set measurementproposals
	source setup.bash
	rosmake facetracking
	rosmake handblobtracker
	rosmake measurementproposals
        rosmake mkfbodytracker_pdaf
	
3.) ...
