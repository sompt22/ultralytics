# Mount a local directory into the container
docker run -it --ipc=host --gpus all -v $PWD:/tmp -v /home/fatih/phd/mot_dataset:/data_dir ultralytics/ultralytics:latest bash