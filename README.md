[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/JColl88/sdc1-solution-binder/HEAD)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.5526844.svg)](https://doi.org/10.5281/zenodo.5526844)

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

While it is possible to perform all remaining steps in a containerised environment, it is more efficient to download the data on the host machine and mount this as a volume in the container.

Download the data (from the project root folder `${SDC1_SOLUTION_ROOT}`):

```bash
$ /bin/bash scripts/download_data.sh
```

Make the docker image:

```bash
$ make dev
```

Run dev container with mounts (this will throw an error if the data has been downloaded to a directory other than `${SDC1_SOLUTION_ROOT}/data`)

```bash
$ sdc1-start-dev
```

Exec a shell inside the container

```bash
$ sdc1-exec-dev
```

### Run an analysis pipeline

A script for running a simple analysis workflow is provided in `scripts/sdc1_solution.py`. This assumes the data has been downloaded as described above, and mounted in the development container (performed by default by the `sdc1-start-dev` alias).

The analysis pipeline can then be run by the container's Python 3.6 interpreter:

```bash
$ PYTHONPATH=./ python3.6 scripts/sdc1_solution.py
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

## Run with BinderHub

It is possible to build an interactive environment capable of running parts of the solution workflow (compute resources dependent) using the cloud service BinderHub. To view this on the public BinderHub deployment [mybinder](https://mybinder.org/), navigate to https://mybinder.org/v2/gh/JColl88/sdc1-solution-binder/HEAD.

### Getting started on BinderHub

After launching a BinderHub environment, some example data must be downloaded. The script `scripts/download_data.sh` is designed to download all of the requisite SDC1 data, however it is likely a BinderHub environment (such as mybinder) will not have the resources necessary to process or even store the full images.

For this situation, a small subsample of each image can be downloaded using the script `binder/download_sample_data.sh`.

After downloading the sample data an example workflow notebook is provided at `analyse_sample.ipynb`.
