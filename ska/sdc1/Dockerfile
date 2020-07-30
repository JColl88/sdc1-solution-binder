FROM ubuntu:18.04

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
    python3.6 \
    python3-pip \
    python3-tk \
    python3-setuptools \
    subversion \
    vim \
    wcslib-dev \
    wget \
    screen

# boost v1.63
RUN cd / \
    && wget https://dl.bintray.com/boostorg/release/1.63.0/source/boost_1_63_0.tar.bz2 \
    && tar xvf boost_1_63_0.tar.bz2 \
    && cd boost_1_63_0 \
    && ./bootstrap.sh \
    --with-python=/usr/bin/python3.6 \
    --with-libraries=python,date_time,filesystem,system,program_options,test \
    && ./b2 install \
    && cd / && rm -r boost_1_63_0*

# montage
RUN git clone https://github.com/Caltech-IPAC/Montage.git \
    && cd Montage \
    && make \
    && cd .. 
ENV PATH="/Montage/bin:${PATH}"

# python3 requirements
COPY ./requirements.txt /tmp/
RUN python3.6 -m pip install --upgrade pip
RUN python3.6 -m pip install -r /tmp/requirements.txt

RUN git clone https://gitlab.com/ska-telescope/sdc/ska-sdc.git ska_sdc
RUN sed -i 's/3.7/3.6/g' ska_sdc/setup.py
RUN python3.6 -m pip install ./ska_sdc

ENV LD_LIBRARY_PATH=${LD_LIBRARY_PATH}/usr/local/lib

WORKDIR /opt/

ENTRYPOINT /bin/bash
