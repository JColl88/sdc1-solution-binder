FROM jupyter/base-notebook:ubuntu-18.04

USER root
RUN apt-get update && apt-get install -y graphviz git

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get --yes install --no-install-recommends \
    bison \
    build-essential \
    cmake \
    eog \
    flex \
    g++ \
    gcc \
    gettext-base \
    gfortran \
    git \
    libarmadillo-dev \
    libblas-dev \
    libcfitsio-dev \
    libfftw3-dev \
    libgsl-dev \
    libgtkmm-3.0-dev \
    libhdf5-serial-dev \
    liblapacke-dev \
    liblog4cplus-1.1-9 \
    liblog4cplus-dev \
    libncurses5-dev \
    libpng-dev \
    libpython3-dev \
    libreadline-dev \
    libxml2-dev \
    openssh-server \
    python3-pip \
    python3-tk \
    python3-setuptools \
    python3-dev \
    subversion \
    vim \
    wcslib-dev \
    wget \
    screen

# Boost
RUN conda install -c conda-forge boost

# Montage
WORKDIR /
RUN git clone https://github.com/Caltech-IPAC/Montage.git \
    && cd Montage \
    && make \
    && cd .. 
ENV PATH="/Montage/bin:${PATH}"

# Python3 requirements
COPY ./ska/sdc1/requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /tmp/requirements.txt

# Aladin lite viewer requirements
RUN python3 -m pip install ipywidgets
RUN python3 -m pip install ipyaladin
RUN python3 -m jupyter nbextension enable --py widgetsnbextension --sys-prefix
RUN python3 -m jupyter nbextension enable --py --sys-prefix ipyaladin

# Add tests/source code/sample images
COPY ./scripts /opt/scripts
COPY ./ska/notebook/example_notebooks /opt/example_notebooks
COPY ./data/sample_images /opt/data/sample_images
COPY ./ska /opt/ska
COPY ./tests /opt/tests

# Fix permissions
RUN fix-permissions /opt/scripts
RUN fix-permissions /opt/example_notebooks
RUN fix-permissions /opt/data
RUN fix-permissions /opt/ska
RUN fix-permissions /opt/tests

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/usr/local/lib
ENV PYTHONPATH=${PYTHONPATH}:/opt

USER $NB_UID

WORKDIR $HOME
