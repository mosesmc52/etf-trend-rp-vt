PROJECT_NAME := algo
COMPOSE_FILE := docker-compose.yml

.PHONY: build up upd shell logs restart stop down clean

build:
	docker-compose -f $(COMPOSE_FILE) build

up:
	docker-compose -f $(COMPOSE_FILE) up

upd: ## Up in daemon mode
	docker-compose -f $(COMPOSE_FILE) up -d

shell:
	docker exec -it $(PROJECT_NAME) /bin/bash

logs:
	docker-compose logs $(PROJECT_NAME)

restart:
	docker-compose -f $(COMPOSE_FILE) restart

stop:
	docker-compose -f $(COMPOSE_FILE) stop

down:
	docker-compose -f $(COMPOSE_FILE) down

clean:
	docker-compose -f $(COMPOSE_FILE) rm -sfv pead-trend-algo
	docker image rm pead-drift-algo_pead-trend-algo || true
