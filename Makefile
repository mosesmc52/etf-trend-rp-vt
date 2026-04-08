# ------- config -------
PROJECT_NAME ?= algo            # default service name for exec/logs
SERVICE      ?= $(PROJECT_NAME) # can override separately if needed

# Choose compose file by MODE:
#   MODE=local  -> docker-compose.local.yml
#   MODE=server -> docker-compose.server.yml
#   (unset)     -> docker-compose.yml
ifeq ($(MODE),local)
  COMPOSE_FILE ?= docker-compose.local.yml
else ifeq ($(MODE),server)
  COMPOSE_FILE ?= docker-compose.server.yml
else
  COMPOSE_FILE ?= docker-compose.yml
endif

# Pick the compose CLI (works for both old/new Docker)
DOCKER_COMPOSE ?= $(if $(shell command -v docker-compose 2>/dev/null),docker-compose,docker compose)

DROPLET_USER ?= root
DROPLET_LOG_FILE ?= /var/log/cloud-init-output.log
SPACES_ENDPOINT ?= https://sfo3.digitaloceanspaces.com

# ------- helpers -------
.PHONY: _assert_compose_file _assert_droplet_ip _assert_log_key _assert_spaces_bucket show-config help
_assert_compose_file:
	@test -f "$(COMPOSE_FILE)" || { echo "ERROR: compose file not found: $(COMPOSE_FILE)"; exit 1; }

_assert_droplet_ip:
	@test -n "$(DROPLET_IP)" || { echo "ERROR: DROPLET_IP is required"; exit 1; }

_assert_log_key:
	@test -n "$(LOG_KEY)" || { echo "ERROR: LOG_KEY is required"; exit 1; }

_assert_spaces_bucket:
	@test -n "$(SPACES_BUCKET)" || { echo "ERROR: SPACES_BUCKET is required"; exit 1; }

show-config:
	@echo "MODE          = $(MODE)"
	@echo "COMPOSE_FILE  = $(COMPOSE_FILE)"
	@echo "SERVICE       = $(SERVICE)"
	@echo "PROJECT_NAME  = $(PROJECT_NAME)"
	@echo "DOCKER_COMPOSE= $(DOCKER_COMPOSE)"

help:
	@echo "Usage: make <target> [MODE=local|server] [SERVICE=name]"
	@echo "Targets: build up upd shell logs restart stop down clean show-config"
	@echo "  do-droplet-log       Tail droplet log over SSH with DROPLET_IP=<ip>"
	@echo "  do-spaces-log        Print uploaded log from Spaces with LOG_KEY=<key>"

# ------- targets -------
.PHONY: build up upd shell logs restart stop down clean
build: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) build

up: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up

upd: _assert_compose_file ## Up in daemon mode
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) up -d

shell: _assert_compose_file
	# prefer compose exec so you don't need container IDs/names
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) exec $(SERVICE) /bin/bash || \
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) exec $(SERVICE) /bin/sh

logs: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) logs -f $(SERVICE)

restart: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) restart

stop: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) stop

down: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down

# Remove containers + named volumes created by the compose file, and local images it built
clean: _assert_compose_file
	$(DOCKER_COMPOSE) -f $(COMPOSE_FILE) down -v --rmi local

do-droplet-log: _assert_droplet_ip
	ssh $(DROPLET_USER)@$(DROPLET_IP) "sudo tail -f $(DROPLET_LOG_FILE)"

do-spaces-log: _assert_log_key _assert_spaces_bucket
	aws --endpoint-url $(SPACES_ENDPOINT) s3 cp s3://$(SPACES_BUCKET)/$(LOG_KEY) -
