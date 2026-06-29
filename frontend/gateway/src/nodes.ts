// Backend node registry + load-balancer. Nodes accept unlimited concurrency;
// the gateway prioritises them by load (active_requests) and throughput
// (tokens/sec). active_requests is owned by the gateway (bumpActive), so the
// heartbeat never overwrites it.
import { config } from "./config.js";
import { one, query } from "./db.js";
import { log } from "./log.js";

export interface NodeRow {
  id: string;
  name: string;
  url: string | null;
  transport: "push" | "pull";
  status: string;
  models: Array<{ id: string; family?: string; kind?: string }>;
  max_concurrency: number;
  active_requests: number;
  load: number;
  throughput: number | null;
  system: Record<string, unknown>;
  last_heartbeat: string;
}

export async function registerNode(input: {
  name: string;
  url?: string | null;
  transport?: "push" | "pull";
  models: unknown;
  max_concurrency: number;
  throughput?: number | null;
  system?: Record<string, unknown>;
}): Promise<NodeRow> {
  const row = await one<NodeRow>(
    `insert into nodes (name, url, transport, models, max_concurrency, throughput, system, status, active_requests, last_heartbeat)
       values ($1,$2,$3,$4,$5,$6,$7,'online',0,now())
     on conflict (name) do update set
       url = excluded.url, transport = excluded.transport, models = excluded.models,
       max_concurrency = excluded.max_concurrency, throughput = excluded.throughput,
       system = excluded.system, status = 'online', last_heartbeat = now()
     returning *`,
    [input.name, input.url ?? null, input.transport ?? "pull",
     JSON.stringify(input.models ?? []), input.max_concurrency,
     input.throughput ?? null, JSON.stringify(input.system ?? {})],
  );
  log.info("node registered", {
    name: input.name,
    transport: input.transport ?? "pull",
    url: input.url ?? null,
  });
  return row as NodeRow;
}

export async function heartbeat(input: {
  name: string;
  throughput?: number;
  system?: Record<string, unknown>;
  models?: unknown;
}): Promise<boolean> {
  const res = await query(
    `update nodes set throughput = coalesce($2, throughput),
        system = coalesce($3, system), models = coalesce($4, models),
        status = 'online', last_heartbeat = now()
      where name = $1`,
    [
      input.name,
      input.throughput ?? null,
      input.system ? JSON.stringify(input.system) : null,
      input.models ? JSON.stringify(input.models) : null,
    ],
  );
  return Array.isArray(res);
}

// Mark nodes that stopped heartbeating as offline.
export async function sweepOffline(): Promise<void> {
  await query(
    `update nodes set status = 'offline'
      where status <> 'offline' and last_heartbeat < now() - ($1::int * interval '1 millisecond')`,
    [config.nodeOfflineMs],
  );
}

// Pick the best online node for `model`: lowest estimated backlog
// (active_requests / tokens-per-second), breaking ties toward faster nodes.
// No capacity cap — returns null only when no online node serves the model.
export async function pickNode(model: string): Promise<NodeRow | null> {
  const rows = await query<NodeRow>(
    `select * from nodes
      where status = 'online'
        and last_heartbeat > now() - ($1::int * interval '1 millisecond')
        and exists (
          select 1 from jsonb_array_elements(models) m
          where m->>'id' = $2 or m->>'family' = $2
        )
      order by (active_requests::float / greatest(throughput, 1)) asc,
               throughput desc nulls last
      limit 1`,
    [config.nodeOfflineMs, model],
  );
  return rows[0] ?? null;
}

export async function listNodes(): Promise<NodeRow[]> {
  return query<NodeRow>("select * from nodes order by name");
}

// Union of concrete models advertised by online nodes (alias entries excluded).
export async function availableModels(): Promise<Array<Record<string, unknown>>> {
  const rows = await query<{ models: Array<Record<string, unknown>> }>(
    `select models from nodes
      where status = 'online' and last_heartbeat > now() - ($1::int * interval '1 millisecond')`,
    [config.nodeOfflineMs],
  );
  const byId = new Map<string, Record<string, unknown>>();
  for (const r of rows) {
    for (const m of r.models ?? []) {
      if (m && typeof m.id === "string" && !m.alias) byId.set(m.id, m);
    }
  }
  return [...byId.values()];
}

export async function bumpActive(nodeId: string, delta: number): Promise<void> {
  await query(
    `update nodes set active_requests = greatest(0, active_requests + $2) where id = $1`,
    [nodeId, delta],
  );
}

// Send a generation job to a node. Throws on network/timeout/HTTP error.
export async function callNode(
  node: NodeRow,
  body: unknown,
): Promise<{ output: string; raw_output: string; input: string; model_input: string; tokens: number; latency_ms: number }> {
  if (node.transport === "pull") return callPullNode(node, body);
  if (!node.url) throw new Error(`push node ${node.name} has no URL`);
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), config.nodeRequestTimeoutMs);
  try {
    const res = await fetch(`${node.url}/internal/generate`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-node-token": config.nodeToken },
      body: JSON.stringify(body),
      signal: ctrl.signal,
    });
    if (!res.ok) {
      const text = await res.text().catch(() => "");
      let detail = text;
      try { detail = (JSON.parse(text).detail as string) ?? text; } catch { /* keep text */ }
      const err = new Error(detail || `node responded ${res.status}`) as Error & { status?: number };
      err.status = res.status;
      throw err;
    }
    return (await res.json()) as never;
  } finally {
    clearTimeout(timer);
  }
}

type NodeResult = {
  output: string;
  raw_output: string;
  input: string;
  model_input: string;
  tokens: number;
  latency_ms: number;
};

async function callPullNode(node: NodeRow, body: unknown): Promise<NodeResult> {
  const job = await enqueuePullJob(node, body);
  const deadline = Date.now() + config.nodeRequestTimeoutMs;
  while (Date.now() < deadline) {
    const row = await pullJobStatus(job.id);
    if (!row) throw new Error("node job disappeared");
    if (row.status === "done" && row.result) {
      await deletePullJob(job.id);
      return row.result;
    }
    if (row.status === "error") {
      const err = new Error(row.error || "node generation failed") as Error & { status?: number };
      err.status = row.error_status ?? 500;
      await deletePullJob(job.id);
      throw err;
    }
    if (row.status === "cancelled") throw new Error("node job cancelled");
    await new Promise((resolve) => setTimeout(resolve, 200));
  }
  await cancelPullJob(job.id);
  throw new Error(`node job timed out after ${config.nodeRequestTimeoutMs}ms`);
}

async function enqueuePullJob(node: NodeRow, body: unknown): Promise<{ id: string }> {
  const job = await one<{ id: string }>(
    `insert into node_jobs (node_id, payload) values ($1,$2) returning id`,
    [node.id, JSON.stringify(body)],
  );
  if (!job) throw new Error("could not queue node job");
  return job;
}

function pullJobStatus(jobId: string) {
  return one<{
    status: string;
    result: NodeResult | null;
    error: string | null;
    error_status: number | null;
  }>(
    "select status, result, error, error_status from node_jobs where id = $1",
    [jobId],
  );
}

async function cancelPullJob(jobId: string): Promise<void> {
  await query(
    "update node_jobs set status='cancelled', finished_at=now() where id=$1 and status in ('queued','running')",
    [jobId],
  );
}

async function deletePullJob(jobId: string): Promise<void> {
  await query("delete from node_jobs where id=$1", [jobId]);
}

// Streaming variant: returns the raw Response so the caller can read NDJSON
// events. Caller must consume the body; `signal` controls cancellation.
export async function callNodeStream(node: NodeRow, body: unknown, signal: AbortSignal): Promise<Response> {
  if (node.transport === "pull") {
    const job = await enqueuePullJob(node, body);
    const encoder = new TextEncoder();
    const stream = new ReadableStream<Uint8Array>({
      async start(controller) {
        let lastEventId = "0";
        let errorSeen = false;
        try {
          while (true) {
            if (signal.aborted) {
              await cancelPullJob(job.id);
              controller.close();
              return;
            }
            const events = await query<{ id: string; event: Record<string, unknown> }>(
              `select id::text, event from node_job_events
                where job_id=$1 and id>$2::bigint order by id`,
              [job.id, lastEventId],
            );
            for (const item of events) {
              lastEventId = item.id;
              if (item.event.event === "error") errorSeen = true;
              controller.enqueue(encoder.encode(JSON.stringify(item.event) + "\n"));
            }

            const status = await pullJobStatus(job.id);
            if (!status) throw new Error("node job disappeared");
            if (status.status === "done") {
              await deletePullJob(job.id);
              controller.close();
              return;
            }
            if (status.status === "error" || status.status === "cancelled") {
              if (!errorSeen) {
                controller.enqueue(encoder.encode(JSON.stringify({
                  event: "error",
                  detail: status.error || `node job ${status.status}`,
                }) + "\n"));
              }
              if (status.status === "error") await deletePullJob(job.id);
              controller.close();
              return;
            }
            await new Promise((resolve) => setTimeout(resolve, 50));
          }
        } catch (err) {
          controller.error(err);
        }
      },
      async cancel() {
        await cancelPullJob(job.id);
      },
    });
    return new Response(stream, { headers: { "content-type": "application/x-ndjson" } });
  }
  if (!node.url) throw new Error(`push node ${node.name} has no URL`);
  const res = await fetch(`${node.url}/internal/generate/stream`, {
    method: "POST",
    headers: { "content-type": "application/json", "x-node-token": config.nodeToken },
    body: JSON.stringify(body),
    signal,
  });
  if (!res.ok || !res.body) {
    const text = await res.text().catch(() => "");
    let detail = text;
    try { detail = (JSON.parse(text).detail as string) ?? text; } catch { /* keep text */ }
    const err = new Error(detail || `node responded ${res.status}`) as Error & { status?: number };
    err.status = res.status;
    throw err;
  }
  return res;
}
