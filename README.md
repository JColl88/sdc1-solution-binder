# Science Data Challenge 1 Solution Workflow

The SKA Science Data Challenge 1 (SDC1, https://astronomers.skatelescope.org/ska-science-data-challenge-1/) tasked participants with identifying and classifying sources in synthetic radio images.

Here we present an environment and workflow for producing a solution to this challenge that can easily be reproduced and developed further.

Instructions for setting up the (containerised) environment and running a simple workflow script using some Python helper modules are provided in this document.

## Environment setup via Docker

To install Docker, follow the general installation instructions on the [Docker](https://docs.docker.com/install/) site:

- [macOS](https://docs.docker.com/docker-for-mac/install/)
- [Ubuntu](https://docs.docker.com/install/linux/docker-ce/ubuntu/)
- [Windows](https://docs.docker.com/docker-for-windows/install/)

## Development and execution environment

Export your sdc1-solution base directory as the environment variable SDC1_SOLUTION_ROOT:

```bash
$ export SDC1_SOLUTION_ROOT="/home/eng/sdc1-solution/"
```

Source `etc/aliases` for shell auto-complete:

```bash
$ source etc/aliases
```

Make the docker image:

```bash
$ make dev
```

### Run dev container with mounts

```bash
$ sdc1-start-dev
```

### Exec a shell inside the container

```bash
$ sdc1-exec-dev
```

### Run an analysis pipeline

A script for running a simple analysis workflow is provided in `scripts/sdc1_solution.py`. Prior to running this, the data must be downloaded. This can be done by executing the script from the project root:

```bash
/bin/bash scripts/download_data.sh
```

The analysis pipeline can then be run by Python 3.6:

```bash
PYTHONPATH=./ python3.6 scripts/sdc1_solution.py
```

### Stop the container

```bash
$ sdc1-stop-dev
```

## Unit testing

Make the docker image:

```bash
$ make test
```

### Run unit tests

```bash
$ sdc1-run-unittests
```
