# Installation and MWE
This is very much research code, but the following should get a test case up and running. On Ubuntu 16.06.

* Install [ROS Kinetic](http://wiki.ros.org/kinetic/Installation/Ubuntu)
* Install opencv
```
sudo apt-get install libopencv-dev
```
* Clone and compile code
```
mkdir pose_estimation
cd pose_estimation/

git clone -b KF_proposals https://github.com/mgb45/handblobtracker.git
git clone https://github.com/mgb45/facetracking.git
git clone -b pca_sampling https://github.com/mgb45/mkfbodytracker_pdaf.git

rosws init . /opt/ros/kinetic
source setup.bash
rosws set handblobtracker/
rosws set facetracking/
rosws set mkfbodytracker_pdaf/
source setup.bash

rosmake facetracking
rosmake handblobtracker
rosmake mkfbodytracker_pdaf
```
* Run demo launch file
```
roslaunch mkfbodytracker_pdaf bodytrackingBag.launch
```

# Caveats
* The face and head detector are very much tuned for the test sequence. More accurate detection of head and hands is required for anything useful. If you can get other joint measurements, the final pose estimate will be even better, but code will need to be modified to accomodate this.
* The motion model has only been trained on predominantly front-on, simple hand motions, if you want to train a new model use [gmm_training]https://github.com/mgb45/

