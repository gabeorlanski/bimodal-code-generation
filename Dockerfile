FROM python:3.8-slim-buster
RUN apt-get update -y \
	&& apt-get upgrade -y \
    && apt-get install -y --no-install-recommends build-essential \
    && apt-get install -y git

ENV LC_ALL=C.UTF-8
ENV LANG=C.UTF-8
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64


ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
LABEL com.nvidia.volumes.needed="nvidia_driver"

#RUN apt-get install -y python3.7 python3-pip
#RUN ln -s /usr/bin/python3 /usr/bin/python

RUN pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 torchaudio==0.10.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
# Install dependencies:
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN git clone https://github.com/gabeorlanski/taskio.git
RUN cd taskio && pip install -r requirements.txt && pip install -e .

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY . .