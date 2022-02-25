FROM python:3.7-slim 
LABEL maintainer=soraya.arias@inria.fr

# Ensure a sane environment
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8 DEBIAN_FRONTEND=noninteractive

RUN apt update --fix-missing && \
    apt install -y apt-utils wget git \
        python3-venv python3-pip && \
    apt -y dist-upgrade && \
    apt clean && \
    rm -fr /var/lib/apt/lists/*

COPY requirements.txt /root/requirements.txt

# Add & Update Python tools and install requirements packages
RUN pip install --upgrade pip && \
    pip install -I --upgrade setuptools && \
    pip install -r /root/requirements.txt 

# Add Jupyter configuration (no browser, listen all interfaces, ...)
COPY jupyter_notebook_config.py /root/.jupyter/jupyter_notebook_config.py
COPY notebook.json /root/.jupyter/nbconfig/notebook.json

# Notebooks as a volume
RUN mkdir  /notebooks/; cd /notebooks && \
   git clone https://gricad-gitlab.univ-grenoble-alpes.fr/talks/fidle.git 

VOLUME /notebooks
WORKDIR /notebooks

#COPY data/fidle-datasets /data

# Set a folder in the volume as Python Path
ENV PYTHONPATH=/notebooks/python_path:$PYTHONPATH

# Force bash as the default shell (useful in the notebooks)
ENV SHELL=/bin/bash

# Set Fidle dataset directory variable
ENV FIDLE_DATASETS_DIR=/data

# Run a notebook by default
CMD ["jupyter", "notebook", "--port=8888", "--ip=0.0.0.0"]
