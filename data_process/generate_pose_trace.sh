#!/bin/bash

#mkdir "../video/pose_trace"

for (( i = 9; i <= 17; i++ )); do
    echo "=======> process video${i} <======="
    mkdir "../video/pose_trace/video_${i}"
    for (( j = 1; j <= 48; j++ )); do
        echo "---> process user${j} ..."
        let k=i-9
        python generate_pose_trace.py ../video/viewing_data/Experiment_2/video_${k}/viewer_${j}.csv ../video/pose_trace/video_${i}/pose_trace_${j}.json
    done
done