# BackTalk gateway + node stack. `make help` lists commands.
.PHONY: help up down logs ps restart migrate reset gateway-logs psql

COMPOSE = docker compose -f docker-compose.yml -f docker-compose.local.yml

help:
	@echo "BackTalk stack commands:"
	@echo "  make up         build + start gateway and db (detached)"
	@echo "  make down       stop everything"
	@echo "  make logs       follow all logs"
	@echo "  make gateway-logs   follow the gateway service"
	@echo "  make ps         show service status"
	@echo "  make restart    recreate services"
	@echo "  make migrate    apply DB migrations"
	@echo "  make reset      DROP + recreate the DB schema (destructive)"
	@echo "  make psql       open a psql shell"

up:
	$(COMPOSE) up --build -d

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f

gateway-logs:
	$(COMPOSE) logs -f gateway

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
