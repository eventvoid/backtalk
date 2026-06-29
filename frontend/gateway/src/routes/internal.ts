// Node-facing routes (protected by the node token): registration + heartbeat.
import type { FastifyInstance } from "fastify";
import { requireNode } from "../auth.js";
import { heartbeat, registerNode } from "../nodes.js";
import { one, query } from "../db.js";

export function registerInternal(app: FastifyInstance): void {
  app.post("/internal/nodes/register", async (req, reply) => {
    if (!requireNode(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    if (!b.name) return reply.code(400).send({ error: "name required" });
    const transport = b.transport === "push" ? "push" : "pull";
    if (transport === "push" && !b.url) {
      return reply.code(400).send({ error: "push nodes require url" });
    }
    const node = await registerNode({
      name: String(b.name),
      url: b.url ? String(b.url) : null,
      transport,
      models: b.models ?? [],
      max_concurrency: Number(b.max_concurrency ?? 0),
      throughput: b.throughput == null ? null : Number(b.throughput),
      system: (b.system as Record<string, unknown>) ?? {},
    });
    return { ok: true, node_id: node.id };
  });

  app.post("/internal/nodes/heartbeat", async (req, reply) => {
    if (!requireNode(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    if (!b.name) return reply.code(400).send({ error: "name required" });
    const ok = await heartbeat({
      name: String(b.name),
      throughput: b.throughput != null ? Number(b.throughput) : undefined,
      system: (b.system as Record<string, unknown>) ?? undefined,
      models: b.models ?? undefined,
    });
    return { ok };
  });

  // Long-poll for work. Pull nodes initiate every connection, so they work
  // behind NAT and do not need a public hostname or an open inbound port.
  app.post("/internal/nodes/jobs/next", async (req, reply) => {
    if (!requireNode(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const name = String(b.name ?? "");
    if (!name) return reply.code(400).send({ error: "name required" });
    await heartbeat({
      name,
      throughput: b.throughput != null ? Number(b.throughput) : undefined,
      system: (b.system as Record<string, unknown>) ?? undefined,
    });
    const waitMs = Math.max(0, Math.min(Number(b.wait_ms ?? 30000), 35000));
    const deadline = Date.now() + waitMs;

    while (true) {
      const job = await one<{ id: string; payload: unknown }>(
        `with next_job as (
           select j.id
             from node_jobs j join nodes n on n.id = j.node_id
            where n.name = $1 and n.transport = 'pull' and j.status = 'queued'
            order by j.created_at
            for update of j skip locked
            limit 1
         )
         update node_jobs j
            set status = 'running', started_at = now()
           from next_job
          where j.id = next_job.id
         returning j.id, j.payload`,
        [name],
      );
      if (job) return { job };
      if (Date.now() >= deadline) return { job: null };
      await new Promise((resolve) => setTimeout(resolve, 250));
    }
  });

  app.post("/internal/nodes/jobs/:id/complete", async (req, reply) => {
    if (!requireNode(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const name = String(b.name ?? "");
    const id = String((req.params as { id?: string }).id ?? "");
    if (!name || !id) return reply.code(400).send({ error: "name and job id required" });

    const isError = b.error != null;
    const rows = await query<{ id: string }>(
      `update node_jobs j
          set status = $3,
              result = $4,
              error = $5,
              error_status = $6,
              finished_at = now()
         from nodes n
        where j.id = $1 and n.name = $2 and j.node_id = n.id
          and j.status = 'running'
        returning j.id`,
      [
        id,
        name,
        isError ? "error" : "done",
        isError ? null : JSON.stringify(b.result ?? {}),
        isError ? String(b.error) : null,
        isError ? Number(b.error_status ?? 500) : null,
      ],
    );
    if (!rows.length) return reply.code(409).send({ error: "job is not running or not owned by node" });
    await heartbeat({
      name,
      throughput: b.throughput != null ? Number(b.throughput) : undefined,
      system: (b.system as Record<string, unknown>) ?? undefined,
    });
    return { ok: true };
  });

  app.post("/internal/nodes/jobs/:id/events", async (req, reply) => {
    if (!requireNode(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const name = String(b.name ?? "");
    const id = String((req.params as { id?: string }).id ?? "");
    const events = Array.isArray(b.events) ? b.events : [];
    if (!name || !id || !events.length) {
      return reply.code(400).send({ error: "name, job id and events required" });
    }
    const owner = await one<{ id: string }>(
      `select j.id from node_jobs j join nodes n on n.id=j.node_id
        where j.id=$1 and n.name=$2 and j.status='running'`,
      [id, name],
    );
    if (!owner) return reply.code(409).send({ error: "job is not running or not owned by node" });
    await query(
      `insert into node_job_events (job_id, event)
       select $1, value from jsonb_array_elements($2::jsonb)`,
      [id, JSON.stringify(events)],
    );
    await query(
      "update nodes set status='online', last_heartbeat=now() where name=$1",
      [name],
    );
    return { ok: true };
  });
}
