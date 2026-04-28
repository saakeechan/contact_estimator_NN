FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

LABEL version="0.6"

USER root

# scipy, tensorboard, pandas, onnx, tqdm, pyyaml, onnxscript
RUN pip install --break-system-packages scipy
RUN pip install --break-system-packages tensorboard
RUN pip install --break-system-packages -U scikit-learn
RUN pip install --break-system-packages pandas
RUN pip install --break-system-packages onnx
RUN pip install --break-system-packages onnxscript
RUN pip install --break-system-packages tqdm
RUN pip install --break-system-packages pyyaml

# build essential, cmake, vim, git
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get -y install cmake
RUN apt-get update && apt-get install -y vim
RUN apt-get install -y build-essential
RUN apt-get update && apt-get install -y git-all

# python3-dev
RUN apt-get update && apt-get install -y python3-dev


# lcm 
RUN apt-get install -y libglib2.0-dev
RUN mkdir -p /home/root/tmp/
RUN cd /home/root/tmp/ \
    && git clone https://github.com/lcm-proj/lcm.git \
    && cd lcm \
    && git checkout v1.5.1
RUN cd /home/root/tmp/lcm/ \
    && mkdir build \
    && cd build \
    && cmake -DLCM_ENABLE_PYTHON=ON .. && make -j && make install
