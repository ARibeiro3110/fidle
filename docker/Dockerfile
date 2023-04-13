# base image
ARG PYTHON_VERSION=3.7
ARG docker_image_base=python:${PYTHON_VERSION}-slim
FROM ${docker_image_base}

# maintainers
LABEL maintainer1=soraya.arias@inria.fr maintainer2=achille.mbogol-touye@grenoble-inp.fr

# Ensure a sane environment
ENV TZ=Europe/Paris LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update --fix-missing && \
    apt install -y --no-install-recommends apt-utils vim \
        python3-venv  \
        python3-pip && \
    apt -y dist-upgrade && \
    apt clean && \
    rm -fr /var/lib/apt/lists/*  
   
# copy Python requirement packages list in docker image
COPY requirements_packages.txt /root/requirements_packages.txt

# Update Python tools and install requirements packages
RUN python3 -m pip install --upgrade pip && \
    pip install --no-cache-dir -r /root/requirements_packages.txt

 
RUN bin/rm /usr/local/share/jupyter/kernels/python3/logo*      
    
# Change default logo and name kernels
COPY images/env-fidle.png /usr/local/share/jupyter/kernels/python3/logo-64x64.png
COPY images/env-fidle.svg /usr/local/share/jupyter/kernels/python3/logo-svg.svg


# Get Fidle datasets
RUN mkdir /data && \
    fid  install_datasets --datasets datasets-fidle-v2.0 --install_dir /data

# Get Fidle notebooks
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
