# BackTalk architecture

## Runtime topology

```text
Users / API clients
        |
        v
Public server with static IP
  frontend/gateway <----> PostgreSQL
        ^
        | outbound HTTPS long polling
        |
Private model nodes
  backend-node + local checkpoints + CPU/CUDA/MPS
```

The gateway is the only public service. A node registers, sends heartbeats,
long-polls for queued jobs, streams real generation events back to the gateway,
and posts the final result. No inbound node connection, public hostname, port
forwarding, or static node IP is required.

## Components

### `frontend/gateway`

- Serves Ask, Stories, Docs, and Dashboard pages.
- Exposes web, native, and OpenAI-compatible APIs.
- Authenticates callers and nodes.
- Applies caller rate/concurrency limits.
- Routes work to online nodes.
- Stores requests, answers, votes, and JSONL exports.
- Queues pull jobs and forwards real token events as SSE.
- Applies forward-only PostgreSQL migrations at startup.

### `backend-node`

- Loads runtime checkpoints and the tokenizer.
- Selects CPU, CUDA, or MPS.
- Uses outbound polling to receive jobs.
- Streams actual model events while generating.
- Reports throughput and system metrics.
- Keeps checkpoints private to the compute machine.

### PostgreSQL

Stores durable application state and the node transport queue. Migration files
under `frontend/gateway/migrations/` are immutable deployment history and must
remain committed.

## Deployment

### Coolify gateway server

Deploy `docker-compose.yml` as a Coolify Docker Compose resource and route the
gateway domain to its internal port `8080`. The production Compose project
starts `db` and `gateway` only, publishes no host port, and has no dependency on
model files. Local development adds `docker-compose.local.yml` through the
Makefile.

### Private node

```bash
pip install -r backend-node/requirements.txt
GATEWAY_URL=https://gateway.example.com \
NODE_TOKEN=change-me-node \
NODE_NAME=my-node \
NODE_TRANSPORT=pull \
python3 backend-node/worker.py
```

The node is deliberately not part of Docker Compose. It runs natively on the
compute machine so MPS or CUDA is available and reads checkpoints from that
machine's local filesystem. Checkpoint distribution through object storage will
be designed separately.

## Repository boundaries

Committed source:

```text
backend-node/
frontend/gateway/
train/
scripts/
configs/
tokenizers/backtalk-tokenizer/
```

Local/generated state:

```text
checkpoints/
data/
dist/
logs/
```

Secrets belong only in `.env`, which is ignored by Git.
