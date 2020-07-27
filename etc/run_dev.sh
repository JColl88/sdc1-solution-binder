#!/bin/bash

docker container stop sdc1-test >> /dev/null 2>&1 \
    && docker container kill sdc1-test >> /dev/null 2>&1

docker run -it --rm -v $SDC1_SOLUTION_ROOT/ska:/opt/ska \
    -v /home/eng/ESCAP/156/sdc1-solution/tests:/opt/tests \
    sdc1-test:latest /bin/bash