#!/bin/bash

N=100000
K=(1 3 6 11 17 22 30 26 22 23 21 24 29 37 37 47 42 36 40 38)

for ((i=0; i<20; i=$i+1));
do
	d=$((i+1))
	roslaunch mkfbodytracker bodyTrackingBagCompare.launch file1:=data13D_PCA_$N_${K[${i}]}_$d.yml file2:=data23D_PCA_$N_${K[${i}]}_$d.yml
done
