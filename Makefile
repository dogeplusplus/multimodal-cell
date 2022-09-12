REPO_NAME := $(shell basename `git rev-parse --show-toplevel`)
DVC_REMOTE := ${GDRIVE_FOLDER}/${REPO_NAME}


.PHONY:test
test:
	poetry run python -m pytest

.PHONY:install-hooks
install-hooks:
	precommit install
