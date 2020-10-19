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

# boost
RUN conda install -c conda-forge boost

# montage
WORKDIR /
RUN git clone https://github.com/Caltech-IPAC/Montage.git \
    && cd Montage \
    && make \
    && cd .. 
ENV PATH="/Montage/bin:${PATH}"

# python3 requirements
COPY ./ska/sdc1/requirements.txt /tmp/
RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install -r /tmp/requirements.txt

# Add tests/source code
COPY ./scripts /opt/scripts
COPY ./ska /opt/ska
COPY ./tests /opt/tests

RUN fix-permissions /opt/scripts
RUN fix-permissions /opt/ska
RUN fix-permissions /opt/tests

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/usr/local/lib

USER $NB_UID

WORKDIR $HOME