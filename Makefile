PORT ?= 8000
HOST ?= 0.0.0.0

export PORT HOST

.PHONY: help run smoke debug test

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

run: ## Start the server (make run PORT=9000 to change port)
	./run.sh

smoke: ## Load the model and run one in-process request, no server
	./run.sh --smoke

debug: ## Start the server with request/prompt/latency logging
	./run.sh --debug

test: ## Send one request per question type (server must be running)
	BASE_URL="http://127.0.0.1:$(PORT)" ./test.sh
