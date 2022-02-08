# Setting base image
FROM ubuntu:16.04


# Adding current maintainer
MAINTAINER pfagan@stanford.edu


# Basic installations
RUN apt-get update -q \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y \
    curl \
    git \
    libgl1-mesa-dev \
    libgl1-mesa-glx \
    libglew-dev \
    libosmesa6-dev \
    software-properties-common \
    net-tools \
    unzip \
    vim \
    virtualenv \
    wget \
    xpra \
    xserver-xorg-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Python 3.6 installation and environment setup
RUN DEBIAN_FRONTEND=noninteractive add-apt-repository --yes ppa:deadsnakes/ppa && apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get install --yes python3.6-dev python3.6 python3-pip
RUN virtualenv --python=python3.6 env

RUN rm /usr/bin/python
RUN ln -s /env/bin/python3.6 /usr/bin/python
RUN ln -s /env/bin/pip3.6 /usr/bin/pip
RUN ln -s /env/bin/pytest /usr/bin/pytest

RUN curl -o /usr/local/bin/patchelf https://s3-us-west-2.amazonaws.com/openai-sci-artifacts/manual-builds/patchelf_0.9_amd64.elf \
    && chmod +x /usr/local/bin/patchelf

ENV LANG C.UTF-8


# Install MuJoCo and copy license key
RUN mkdir -p /root/.mujoco \
    && wget https://www.roboti.us/download/mujoco200_linux.zip -O mujoco.zip \
    && unzip mujoco.zip -d /root/.mujoco \
    && mv /root/.mujoco/mujoco200_linux /root/.mujoco/mujoco200 \
    && rm mujoco.zip
COPY ./mjkey.txt /root/.mujoco/
ENV LD_LIBRARY_PATH /root/.mujoco/mujoco200/bin:${LD_LIBRARY_PATH}


# Install further dependencies (we could move all pip install details to this file and delete )
RUN pip install --no-cache-dir glfw>=1.4.0 \
                numpy>=1.11 \
                Cython>=0.27.2 \
                cffi>=1.10 \
                imageio>=2.1.2 \
                fasteners~=0.15 \
                imagehash>=3.4 \
                Pillow>=4.0.0 \
                pycparser>=2.17.0 \
                pytest>=3.0.5 \
                pytest-instafail==0.3.0 \
                mujoco-py \
                torch \
                matplotlib \
                gym \
                tensorboard \
                pyyaml


# Switch working directory to assignment src mount location
WORKDIR /home/scpdxcs/A3
