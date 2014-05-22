#!/bin/bash

for i in {0088..0400}
do
	
	eval montage -background black ./pose/left0$((10#$i+23)).jpg ./edgeVoting/left$i.jpg -geometry +4+4 test$((10#$i-88)).jpg
done

eval avconv -r 15 -i test%d.jpg -codec:v libx264 -r 30 -pix_fmt yuv420p out.mp4
