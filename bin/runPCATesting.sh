#!/bin/bash

source /opt/ros/groovy/setup.bash
source /home/mgb45/Documents/setup.bash

N=100000
#K=(1 3 6 11 17 22 30 26 22 23 21 24 29 37 37 47 42 36 40 38)

for ((k=1; k<50; k=$k+1));
do
	for ((i=1; i<21; i=$i+1));
	do
		roslaunch mkfbodytracker bodyTrackingBagCompare.launch "file1:=/data13D_PCA_${N}_${k}_$i.yml" "file2:=/data23D_PCA_${N}_${k}_${i}.yml"
	done
done
