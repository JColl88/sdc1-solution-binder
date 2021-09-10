.ONESHELL:

dev:
	docker build ./ska/sdc1/ -f ./ska/sdc1/Dockerfile --tag sdc1-dev:latest

test: dev
	docker build ./tests/ -f ./tests/Dockerfile --tag sdc1-test:latest

