#
#
ARG PYTHON_VERSION=3.8
ARG docker_image_base=python:${PYTHON_VERSION}-slim
FROM ${docker_image_base}

LABEL maintainer=soraya.arias@inria.fr

# Ensure a sane environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing && \
    apt install -y --no-install-recommends apt-utils && \
        python3-venv python3-pip && \
    apt -y dist-upgrade && \
    curl -fsSL https://deb.nodesource.com/setup_lts.x |  bash - && \
    apt install -y nodejs && \
    apt clean && \
    rm -fr /var/lib/apt/lists/*

# Get Python requirement packages list
COPY requirements-cpu.txt /root/requirements.txt

# Add & Update Python tools and install requirements packages
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir -I --upgrade setuptools && \
    pip install --no-cache-dir -r /root/requirements.txt 

# Get Fidle datasets
# - datasets
RUN mkdir /data && \
    fid  install_datasets --quiet --install_dir /data
# - notebooks
RUN mkdir /notebooks/ && \
    fid install_notebooks --quiet --install_dir /notebooks

# Add Jupyter configuration (no browser, listen all interfaces, ...)
COPY jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py
COPY notebook.json /root/.jupyter/nbconfig/notebook.json

# Jupyter notebook uses 8888 
EXPOSE 8888

VOLUME /notebooks
WORKDIR /notebooks

# Set a folder in the volume as Python Path
ENV PYTHONPATH=/notebooks/fidle-master/:$PYTHONPATH

# Force bash as the default shell (useful in the notebooks)
ENV SHELL=/bin/bash

# Set Fidle dataset directory variable
ENV FIDLE_DATASETS_DIR=/data/datasets-fidle

# Run a notebook by default
CMD ["jupyter", "lab"]
