#!/bin/bash

docker container stop sdc1-dev >> /dev/null 2>&1
docker container rm sdc1-dev >> /dev/null 2>&1

docker run --name sdc1-dev -d -t \
    -v $SDC1_SOLUTION_ROOT/data:/opt/data \
    -v $SDC1_SOLUTION_ROOT/docs:/opt/docs \
    -v $SDC1_SOLUTION_ROOT/scripts:/opt/scripts \
    -v $SDC1_SOLUTION_ROOT/ska:/opt/ska \
    -v $SDC1_SOLUTION_ROOT/tests:/opt/tests \
    sdc1-dev:latest
