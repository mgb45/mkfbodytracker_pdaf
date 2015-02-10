#!/bin/bash

for i in /media/mgb45/System/Data/ChaLearn/valid* ; do
	for j in $i/* ; do
		rm -fr $j/Sample${j:52:5}_color*.bag
		roslaunch mkfbodytracker_pdaf ChaLearn.launch FILENAME:=$j/Sample${j:52:5}_color SAVE:=true
#		roslaunch mkfbodytracker_pdaf ChaLearn.launch FILENAME:=$j/Sample${j:52:5}_color SAVE:=false
	done
done
