#!/bin/bash

for i in /opt/ChaLearn/validation* ; do
	for j in $i/Sample* ; do
		rm -fr $j/Sample*.bag
		roslaunch mkfbodytracker_pdaf ChaLearn.launch FILENAME:=$j/Sample${j:32:5}_color SAVE:=true
#		roslaunch mkfbodytracker_pdaf ChaLearn.launch FILENAME:=$j/Sample${j:52:5}_color SAVE:=false
	done
done
