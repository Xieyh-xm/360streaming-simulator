#!/bin/bash

for (( id = 0; id <= 8; id++ )); do
  mkdir ../video/viewing_data/video_${id}
  for f in `seq 48`
  do
      cp ../Experiment_2/${f}/video_${id}.csv ../video/viewing_data/video_${id}/viewer_${f}.csv
  done
done
#python  generate_cu_navigation_graph.py ../video/viewing_data/video_0/viewer_{?,1?,2?,3?,4[0-2]}.csv