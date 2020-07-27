#!/bin/bash

docker container stop sdc1-test >> /dev/null 2>&1
docker container rm sdc1-test >> /dev/null 2>&1

docker run --rm --name sdc1-test -t -d \
    -v $SDC1_SOLUTION_ROOT/ska:/opt/ska \
    -v $SDC1_SOLUTION_ROOT/tests:/opt/tests \
    sdc1-test:latest \
    && docker exec -it sdc1-test python3.6 -m pytest tests/unittests 

docker container stop sdc1-test >> /dev/null 2>&1
docker container rm sdc1-test >> /dev/null 2>&1
    

