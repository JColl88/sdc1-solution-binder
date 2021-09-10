FROM jupyter/scipy-notebook:ubuntu-18.04

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

# Install Rucio components - from https://hub.docker.com/r/rucio/jupyterlab/dockerfile
RUN conda update -n base conda
RUN conda install -y -c conda-forge python-gfal2 \
    && conda clean --all -f -y

RUN apt update -y \
    && apt install -y voms-clients-java \
    && apt clean -y \
    && rm /opt/conda/bin/voms-proxy-init \
    && ln -s /usr/bin/voms-proxy-init /opt/conda/bin/voms-proxy-init

RUN git clone https://github.com/rucio/jupyterlab-extension.git /rucio-jupyterlab
WORKDIR /rucio-jupyterlab

RUN fix-permissions /rucio-jupyterlab \
    && sed -i -e 's/\r$/\n/' /rucio-jupyterlab/docker/configure.sh

RUN pip install -e . \
    && jupyter serverextension enable --py rucio_jupyterlab.server --sys-prefix \
    && jlpm \
    && jlpm build \
    && jupyter labextension link . --dev-build=False --minimize=False \
    && jupyter lab clean -y \
    && npm cache clean --force \
    && rm -rf "/home/${NB_USER}/.cache/yarn" \
    && rm -rf "/home/${NB_USER}/.node-gyp"

ENV JUPYTER_ENABLE_LAB=yes

WORKDIR $HOME

# Add further Rucio components - from https://github.com/ESCAPE-WP2/Rucio-Client-Containers/blob/master/rucio-jupyterhub-container/Dockerfile
RUN apt-get update -y && apt-get install -y wget curl gnupg vim

# EGI trust anchors
RUN wget -q -O - \
    https://dist.eugridpma.info/distribution/igtf/current/GPG-KEY-EUGridPMA-RPM-3 \
    | apt-key add -

RUN echo "deb http://repository.egi.eu/sw/production/cas/1/current egi-igtf core" >> /etc/apt/sources.list 

RUN apt-get update -y && apt-get install -y ca-certificates ca-policy-egi-core && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* /var/cache/apt/archives/*

# ESCAPE VOMS setup
RUN mkdir -p /etc/vomses \
    && wget https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.vomses -O /etc/vomses/voms-escape.cloud.cnaf.infn.it.vomses
RUN mkdir -p /etc/grid-security/vomsdir/escape \
    && wget https://indigo-iam.github.io/escape-docs/voms-config/voms-escape.cloud.cnaf.infn.it.lsc -O /etc/grid-security/vomsdir/escape/voms-escape.cloud.cnaf.infn.it.lsc

RUN git clone https://github.com/ESCAPE-WP2/Rucio-Client-Containers.git /rucio-client-containers
RUN cp /rucio-client-containers/rucio-jupyterhub-container/etc/init.sh /
RUN chmod o+x /init.sh

RUN conda install j2cli && \
    conda clean --all -f -y

RUN cp /rucio-client-containers/rucio-jupyterhub-container/etc/rucio.cfg.j2 /
RUN mkdir -p /opt/rucio/etc/
RUN fix-permissions /opt/rucio/etc/
RUN cp /rucio-client-containers/rucio-jupyterhub-container/etc/createProxyFromP12.sh /opt/rucio/etc/
RUN rm -r /rucio-client-containers

USER $NB_UID
CMD /init.sh


