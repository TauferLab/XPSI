#!/bin/sh

for i in $(seq 2 20)
do
	echo "K= ${i}"
        python predict_angles.py $1 --knn-trials=10 --rf-trials=0 --knn-average=vector_mean_3_rad --k="$i" --trees=0
done
