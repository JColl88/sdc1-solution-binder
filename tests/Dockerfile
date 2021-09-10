FROM sdc1-dev:latest as sdc1

# python3 requirements
COPY ./requirements.txt /tmp/
RUN python3.6 -m pip install -r /tmp/requirements.txt

WORKDIR /opt/

ENTRYPOINT /bin/bash