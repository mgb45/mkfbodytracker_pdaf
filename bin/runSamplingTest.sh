#!/bin/bash

source /opt/ros/groovy/setup.bash
source /home/mgb45/Documents/setup.bash

N=100000
K=27

for ((i=0; i<20; i=$i+1));
do
	roslaunch mkfbodytracker bodyTrackingBagCompare.launch "file1:=/data13D_PCA_${N}_${K}.yml" "file2:=/data23D_PCA_${N}_${K}.yml"
done
