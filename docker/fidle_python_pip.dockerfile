#
#
ARG PYTHON_VERSION=3.7
ARG docker_image_base=python:${PYTHON_VERSION}-slim
FROM ${docker_image_base}

MAINTAINER soraya.arias@inria.fr
LABEL maintainer=soraya.arias@inria.fr

# Ensure a sane environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing && \
    apt install -y --no-install-recommends apt-utils && \
    apt install wget curl git \
        python3-venv python3-pip && \
    apt -y dist-upgrade && \
    curl -fsSL https://deb.nodesource.com/setup_lts.x |  bash - && \
    apt install -y nodejs && \
    apt clean && \
    rm -fr /var/lib/apt/lists/*

# Get Python requirement packages list
COPY requirements.txt /root/requirements.txt

# Add & Update Python tools and install requirements packages
RUN pip install --upgrade pip && \
    pip install -I --upgrade setuptools && \
    pip install -r /root/requirements.txt 

# Get Fidle datasets
RUN mkdir /data && \
    wget -c https://fidle.cnrs.fr/fidle-datasets.tar && tar -xf fidle-datasets.tar -C /data/ && \
    rm fidle-datasets.tar

# Notebooks as a volume
RUN mkdir /notebooks/; cd /notebooks && \
    git clone https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle.git

# Add Jupyter configuration (no browser, listen all interfaces, ...)
COPY jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py
COPY fidle_env_test.py /root/fidle_env_test.py

# Jupyter notebook uses 8888 
EXPOSE 8888

VOLUME /notebooks
WORKDIR /notebooks

# Set a folder in the volume as Python Path
ENV PYTHONPATH=/notebooks/fidle/:$PYTHONPATH

# Force bash as the default shell (useful in the notebooks)
ENV SHELL=/bin/bash

# Set Fidle dataset directory variable
ENV FIDLE_DATASETS_DIR=/data/fidle-datasets

# Run a notebook by default
CMD ["jupyter", "lab"]
