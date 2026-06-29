// Admin/dashboard API (requires the admin token). Realtime overview (SSE),
// nodes, usage, limit events, collected dataset + export, API-key management.
import type { FastifyInstance, FastifyReply } from "fastify";
import { createApiKey, requireAdmin } from "../auth.js";
import { listNodes } from "../nodes.js";
import { exportJsonl } from "../dataset.js";
import { one, query } from "../db.js";
import { log } from "../log.js";
import type { Mode } from "../config.js";

// One dashboard snapshot. Feedback is reported as "% of eligible requests the
// user actually voted on" (only web mode offers feedback).
async function overview(): Promise<Record<string, unknown>> {
  const nodes = await listNodes();
  const totals = await one<{ requests: number; errors: number; rejected: number }>(
    `select count(*)::int requests,
            count(*) filter (where status='error')::int errors,
            count(*) filter (where status in ('rejected','busy'))::int rejected
       from requests`,
  );
  const offered = await one<{ n: number }>(
    `select count(*)::int n from requests where status='ok' and mode = 'web'`,
  );
  const voted = await one<{ n: number }>(
    `select count(distinct f.request_id)::int n from feedback f
       join requests r on r.id = f.request_id
      where r.status='ok' and r.mode = 'web'`,
  );
  const breakdownRows = await query<{ choice: string; n: number }>(
    "select choice, count(*)::int n from feedback group by choice",
  );
  const breakdown: Record<string, number> = { A: 0, B: 0, yes: 0, no: 0, none: 0 };
  for (const r of breakdownRows) breakdown[r.choice] = r.n;
  const limitEvents = await query(
    "select kind, mode, detail, created_at from limit_events order by created_at desc limit 20",
  );
  const off = offered?.n ?? 0;
  const v = voted?.n ?? 0;
  return {
    nodes,
    totals: { requests: totals?.requests ?? 0, errors: totals?.errors ?? 0, rejected: totals?.rejected ?? 0 },
    feedback: { offered: off, voted: v, voted_pct: off ? Math.round((100 * v) / off) : 0, breakdown },
    limit_events: limitEvents,
  };
}

export function registerAdmin(app: FastifyInstance): void {
  app.get("/admin/overview", async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    return overview();
  });

  // Realtime dashboard via SSE (auth by header — the client reads with fetch).
  app.get("/admin/stream", async (req, reply: FastifyReply) => {
    if (!requireAdmin(req, reply)) return;
    reply.hijack();
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    let alive = true;
    reply.raw.on("close", () => { alive = false; });
    while (alive) {
      try {
        const snap = await overview();
        reply.raw.write(`event: overview\ndata: ${JSON.stringify(snap)}\n\n`);
      } catch (err) {
        log.error("overview stream failed", { error: String(err) });
      }
      await new Promise((r) => setTimeout(r, 3000));
    }
    reply.raw.end();
  });

  app.get("/admin/dataset", async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const q = req.query as Record<string, string>;
    const where: string[] = ["true"];
    const params: unknown[] = [];
    if (q.mode) { params.push(q.mode); where.push(`r.mode = $${params.length}`); }
    if (q.model) { params.push(q.model); where.push(`r.model = $${params.length}`); }
    if (q.status) { params.push(q.status); where.push(`r.status = $${params.length}`); }
    params.push(Math.min(Number(q.limit ?? 5), 1000)); // default 5
    const rows = await query(
      `select r.id, r.status, r.mode, r.model, n.name as node, r.latency_ms, r.created_at, r.input,
              exists(select 1 from feedback f where f.request_id = r.id) as voted
         from requests r left join nodes n on n.id = r.node_id
        where ${where.join(" and ")}
        order by r.created_at desc limit $${params.length}`,
      params,
    );
    return { data: rows };
  });

  app.get("/admin/dataset/export", async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const q = req.query as Record<string, string>;
    const jsonl = await exportJsonl({
      mode: q.mode,
      model: q.model,
      limit: Number(q.limit ?? 10000),
    });
    reply.header("content-type", "application/x-ndjson");
    reply.header("content-disposition", 'attachment; filename="backtalk-dataset.jsonl"');
    return jsonl;
  });

  app.post("/admin/users", async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const name = String(b.name ?? "").trim();
    if (!name) return reply.code(400).send({ error: "name required" });
    const u = await one<{ id: string }>(
      "insert into users (name, role) values ($1, $2) returning id",
      [name, String(b.role ?? "user")],
    );
    return { user_id: u?.id };
  });

  app.post("/admin/keys", async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const userId = String(b.user_id ?? "");
    const mode = (b.mode === "native" ? "native" : "openai") as Mode;
    if (!userId) return reply.code(400).send({ error: "user_id required" });
    const raw = await createApiKey(userId, mode, b.name ? String(b.name) : null);
    return { api_key: raw, mode }; // returned ONCE; only the hash is stored
  });

  app.get("/admin/keys", async (req, reply) => {
    if (!requireAdmin(req, reply)) return;
    const rows = await query(
      `select k.id, k.key_prefix, k.mode, k.active, k.created_at, u.name as user, u.rating
         from api_keys k join users u on u.id = k.user_id order by k.created_at desc`,
    );
    return { data: rows };
  });
}
