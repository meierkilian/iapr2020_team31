#!/bin/bash 
# Rotate video in 15-degree steps
# for i in {0..345..15}
for i in {0..15..15}
do
    echo "Rotation: "$i" degrees."
    ffmpeg -i ../data/robot_parcours_1.avi -c:v libx265 -crf 2 -vf "rotate="$i"*PI/180:ow='max(iw,ih)':oh='max(iw,ih)'" \
        ./robot_parcours_1_rotated_$i.avi
    python3 ./main.py ../data/robot_parcours_1_rotated_$i.avi ../data/output_rotated_$i.avi
done