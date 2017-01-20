# Fast 3D pose estimation from single camera 2D measurements

This package performs 3D pose estimation from 2D measurements of head and hands using a mixture of mean reverting random walks motion model. More detail on the model is available in 

> Burke, M. G. (2015). Fast upper body pose estimation for human-robot interaction (doctoral thesis). https://doi.org/10.17863/CAM.203

> Burke, M. and Lasenby, J., Fast upper body joint tracking using kinect pose priors, International Conference on Articulated Motion and Deformable Objects (Best paper award), 94-105, 2014. https://doi.org/10.1007/978-3-319-08849-5_10

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
* Download test [bag file](https://drive.google.com/file/d/0BxI1ZklhNyhhdV9Fa0tDVW5pTDg/view?usp=sharing) and save in `mkfbodytracker_pdaf`
* Run demo launch file
```
roslaunch mkfbodytracker_pdaf bodyTrackingBag.launch
```

# Caveats
* The image input in the MWE can be replaced with a suitable ros image topic, say from a webcam streamer like http://wiki.ros.org/gscam, but the face and head detector (from other packages) are hand tuned for the test sequence accompanying the package. More accurate detection of head and hands is required for anything useful. If you can get other joint measurements, the final pose estimate will be even better, but code will need to be modified to accomodate this.
* The motion model has only been trained on predominantly front-on, simple hand motions, if you want to train a new model use [gmm_training](https://github.com/mgb45/)
* The models provided have been trained with an xbox kinect camera, and use the camera calibration file in cal.yml for 3D reconstruction. If your camera differs wildy from this, re-training will be required.


