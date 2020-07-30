#!/bin/bash

docker container stop sdc1-test >> /dev/null 2>&1
docker container rm sdc1-test >> /dev/null 2>&1

docker run --name sdc1-test -d -t \
    -v $SDC1_SOLUTION_ROOT/ska:/opt/ska \
    -v $SDC1_SOLUTION_ROOT/tests:/opt/tests \
    sdc1-test:latest
