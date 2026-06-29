# BackTalk gateway + node stack. `make help` lists commands.
.PHONY: help up up-node down logs ps restart migrate reset gateway-logs node-logs psql

COMPOSE = docker compose -f docker-compose.yml -f docker-compose.local.yml

help:
	@echo "BackTalk stack commands:"
	@echo "  make up         build + start gateway and db (detached)"
	@echo "  make up-node    optionally start the local Docker node profile"
	@echo "  make down       stop everything"
	@echo "  make logs       follow all logs"
	@echo "  make gateway-logs / node-logs   follow one service"
	@echo "  make ps         show service status"
	@echo "  make restart    recreate services"
	@echo "  make migrate    apply DB migrations"
	@echo "  make reset      DROP + recreate the DB schema (destructive)"
	@echo "  make psql       open a psql shell"

up:
	$(COMPOSE) up --build -d

up-node:
	$(COMPOSE) --profile local-node up --build -d node

down:
	$(COMPOSE) --profile local-node down

logs:
	$(COMPOSE) logs -f

gateway-logs:
	$(COMPOSE) logs -f gateway

node-logs:
	$(COMPOSE) --profile local-node logs -f node

ps:
	$(COMPOSE) ps

restart:
	$(COMPOSE) up -d --force-recreate

# The gateway also auto-migrates on boot; these are for manual control.
migrate:
	$(COMPOSE) exec gateway node dist/migrate.js up

reset:
	$(COMPOSE) exec gateway node dist/migrate.js reset

psql:
	$(COMPOSE) exec db psql -U $${POSTGRES_USER:-backtalk} -d $${POSTGRES_DB:-backtalk}
