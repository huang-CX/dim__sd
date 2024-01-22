#!/bin/bash
id=0
left_y=0.035
right_y=0.035
while(($id<=34))
do

    if (($id<=20))
    then
	interval=0.07  	
	floor=`expr $id / 7`
	left_y=$(echo "0.03+$floor*0.005"|bc)
	right_y=$(echo "$left_y-$interval"|bc)
    elif (($id<=27))
    then
        interval=0.08
	left_y=0.04
	right_y=$(echo "$left_y-$interval"|bc)
    elif (($id<=34))
    then
	interval=0.09
	left_y=0.045
	right_y=$(echo "$left_y-$interval"|bc)
    fi
    mod=`expr $id % 7`
    x=$(echo "-0.003+$mod*0.001"|bc)
    echo $id, $mod, $x, $left_y, $right_y
    sed -i '150c \      xyz="'$x' '$left_y' 0"' exhx5_w_b_16_$id.urdf
    sed -i '493c \      xyz="'$x' '$right_y' 0"' exhx5_w_b_16_$id.urdf
    sed -i 's/\/home\/zhou\/exhx5_pybullet/../g' exhx5_w_b_16_$id.urdf
    sed -i 's/package:\/\///g' exhx5_w_b_16_$id.urdf
    sed -i 's/exhx5_pybullet/../g' exhx5_w_b_16_$id.urdf
    let "id++"
done
