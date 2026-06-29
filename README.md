# BackTalk

BackTalk is a distributed reverse-text language-model service:

- `frontend/gateway` is the public Node.js/TypeScript gateway and web UI.
- PostgreSQL stores users, requests, feedback, node state, and pull jobs.
- `backend-node` runs models on private machines using CPU, CUDA, or Apple MPS.
- Nodes make outbound long-polling requests; they need no static IP or open port.

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full design.

## Deploy the public gateway with Coolify

Create a Docker Compose resource from this repository. In Coolify, assign the
gateway service the domain `https://gateway.example.com:8080`; `8080` selects
the internal container port and is not exposed publicly. Set:

```env
PUBLIC_BASE_URL=https://gateway.example.com
TRUST_PROXY=true
POSTGRES_PASSWORD=replace-with-a-random-password
ADMIN_TOKEN=replace-with-a-random-token
NODE_TOKEN=replace-with-a-random-token
```

Coolify's proxy terminates TLS and reaches the container over its private
network, so the Compose file does not publish a gateway host port. Models and
checkpoints are not needed on the public server. With Cloudflare proxying
enabled, use Full (strict) SSL and keep the origin reachable only through the
Coolify proxy.

For local development:

```bash
cp .env.example .env
make up
```

The local override publishes the gateway at `http://127.0.0.1:8080`.

## Run a private model node

On a machine that has the checkpoints:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r backend-node/requirements.txt

GATEWAY_URL=https://gateway.example.com \
NODE_TOKEN=the-same-node-token \
NODE_NAME=my-mac \
NODE_TRANSPORT=pull \
python3 backend-node/worker.py
```

Required local runtime files:

```text
checkpoints/backtalk-assistant/model.pt
checkpoints/backtalk-assistant-v2/model.pt
checkpoints/backtalk-storyteller/model.pt
tokenizers/backtalk-tokenizer/tokenizer.json
```

Checkpoints are intentionally excluded from Git. Their long-term object-storage
distribution strategy is not fixed yet; existing local files continue to work.

For an optional node container on a machine that already has the checkpoints:

```bash
make up-node
```

## Models

| API ID | Purpose |
|---|---|
| `backtalk-assistant` | General Q&A and conversation |
| `backtalk-storyteller` | Structured short stories |

Optimizer-free runtime checkpoints are produced with:

```bash
python3 scripts/export_web_models.py
```

## Training

```bash
pip install -r requirements.txt
python3 train/cli.py --help
```

Training code lives in `train/`, configurations in `configs/`, and reproducible
dataset/training utilities in `scripts/`. Generated data, logs, packages, and
model weights are excluded from Git.

## License

MIT
