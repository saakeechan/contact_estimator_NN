container_name=$1

xhost +local:
docker run -it --net=host \
  --user=$(id -u) \
  -e DISPLAY=$DISPLAY \
  -e QT_GRAPHICSSYSTEM=native \
  -e XAUTHORITY \
  -e USER=$USER \
  --workdir=/home/$USER/Research/deep-contact-estimator \
  -v "/tmp/.X11-unix:/tmp/.X11-unix:rw" \
  -v "/etc/passwd:/etc/passwd:rw" \
  -e "TERM=xterm-256color" \
  -v "/home/saakee/Research:/home/$USER/Research" \
  --device=/dev/dri:/dev/dri \
  --name=${container_name} \
  --security-opt seccomp=unconfined \
  contact_estimator_pytorch:latest
