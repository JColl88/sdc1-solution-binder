FROM sdc1-dev:latest as sdc1

# montage
RUN git clone https://github.com/Caltech-IPAC/Montage.git \
    && cd Montage \
    && make \
    && cd .. 
ENV PATH="/Montage/bin:${PATH}"

# python3 requirements
COPY ./requirements.txt /tmp/
RUN python3.6 -m pip install -r /tmp/requirements.txt

RUN python3.6 -m pip install --upgrade bdsf

WORKDIR /opt/

ENTRYPOINT /bin/bash