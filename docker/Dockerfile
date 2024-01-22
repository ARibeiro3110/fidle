# base image
ARG PYTHON_VERSION=3.9
ARG docker_image_base=python:${PYTHON_VERSION}-slim
FROM ${docker_image_base}

# maintainers
LABEL maintainer1=soraya.arias@inria.fr maintainer2=jean-luc.parouty@simap.grenoble-inp.fr

ARG ARCH_VERSION=cpu

# Ensure a sane environment
ENV TZ=Europe/Paris LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update --fix-missing && \
    apt install -y --no-install-recommends apt-utils  \
        python3-venv  \
        python3-pip && \
    apt -y dist-upgrade && \
    apt clean && \
    rm -fr /var/lib/apt/lists/*  
   
# copy Python requirement packages list in docker image
COPY requirements_packages_${ARCH_VERSION}.txt /root/requirements_packages_${ARCH_VERSION}.txt

# Update Python tools and install requirements packages
RUN python3 -m pip install --upgrade pip && \
    pip3 install --no-cache-dir --upgrade -r /root/requirements_packages_${ARCH_VERSION}.txt 

# Special case to deal with keras_cv and keras3 !
RUN pip3 install --no-cache-dir --upgrade keras-cv tensorflow && \
    pip3 install --no-cache-dir --upgrade keras

# Install tensorboard & update jupyter
RUN pip3 install --no-cache-dir --upgrade tensorboard tensorboardX jupyter ipywidgets

# Move default logo python
RUN bin/rm /usr/local/share/jupyter/kernels/python3/logo*      
    
# Change default logo and name kernels
COPY images/env-fidle.png /usr/local/share/jupyter/kernels/python3/logo-64x64.png
COPY images/env-fidle.svg /usr/local/share/jupyter/kernels/python3/logo-svg.svg

# Get Fidle datasets
RUN mkdir /data && \
    fid  install_datasets --quiet --install_dir /data

# Get Fidle notebooks and create link
RUN mkdir /notebooks/ && \
    fid install_notebooks --notebooks fidle-pre-master --quiet --install_dir /notebooks && \
    ln -s /notebooks/fidle-pre-master-3.0.0 /notebooks/fidle-master

# Add Jupyter configuration (no browser, listen all interfaces, ...)
COPY jupyter_lab_config.py /root/.jupyter/jupyter_lab_config.py
COPY notebook.json /root/.jupyter/nbconfig/notebook.json

# Jupyter notebook uses 8888 
EXPOSE 8888

# tensorboard uses 6006 
EXPOSE 6006

VOLUME /notebooks
WORKDIR /notebooks

# Set a folder in the volume as Python Path
ENV KERAS_BACKEND torch
ENV PYTHONPATH=/notebooks/fidle-master/:$PYTHONPATH

# Force bash as the default shell (useful in the notebooks)
ENV SHELL=/bin/bash

# Set Fidle dataset directory variable
ENV FIDLE_DATASETS_DIR=/data/datasets-fidle

# Run a notebook by default
CMD ["jupyter", "lab"]