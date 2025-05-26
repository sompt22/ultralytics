# Perform object tracking on a video from the command line
# You can specify different sources like webcam (0) or RTSP streams
yolo track \
  classes=[0] \
  source=/data_dir/SOMPT22/images/test/SOMPT22-01/SOMPT22-01.mp4 \
  save=True \
  save_frames=True \
  show_labels=False \
  show_conf=True \
  name=exp1 \
  stream=True \
  project=/tmp