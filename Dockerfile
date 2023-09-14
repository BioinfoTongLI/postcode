FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

RUN apt-get -y update && apt-get -y install --no-install-recommends procps python3 python3-dev python3-pip python-is-python3 libgomp1 ffmpeg libsm6 libxext6 git

COPY requirements.txt /tmp/requirements.txt

RUN pip install -r /tmp/requirements.txt --no-cache-dir && pip install git+https://github.com/BioinfoTongLI/postcode.git@master --no-cache-dir