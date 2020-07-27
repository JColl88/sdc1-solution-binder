# sdc1-solution

Workflow for a solution to the Square Kilometer Array (SKA) Science Data Challenge 1

Work in progress, but progress is documented on the SKA confluence page: https://confluence.skatelescope.org/display/SE/SP-1044%3A+Data+Challenge+1+Solution

## Development environment

Export your sdc1-solution base directory as the environment variable SDC1_SOLUTION_ROOT:

```bash
$ export SDC1_SOLUTION_ROOT="/home/eng/ESCAP/156/sdc1-solution/"
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

Exec a shell inside the container:

```bash
$ sdc1-exec-dev
```

Stop the container:
```bash
$ sdc1-stop-dev
```

### Unit testing

Make the docker image:

```bash
$ make test
```

```bash
$ sdc1-run-test
```



