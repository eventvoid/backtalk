// Browser UI backend. The browser holds no API key; these run as the built-in
// "web" user. Ask + Stories support feedback (A/B for two answers, Yes/No for one).
import type { FastifyInstance, FastifyReply } from "fastify";
import { runGeneration, runGenerationStream, type GenContext } from "../generate.js";
import { availableModels } from "../nodes.js";
import { logFeedback } from "../dataset.js";
import { noteFeedback } from "../users.js";
import { one } from "../db.js";
import { log } from "../log.js";
import { config } from "../config.js";
import { cleanParams, clientIp, normChoice, sendError } from "./util.js";

// web mode is the site only: a browser fetch sends an Origin matching the host.
function sameOrigin(req: import("fastify").FastifyRequest): boolean {
  const origin = req.headers["origin"];
  const expectedHost = config.publicBaseUrl
    ? new URL(config.publicBaseUrl).host
    : req.headers["host"];
  if (!origin || !expectedHost) return false;
  try { return new URL(String(origin)).host === expectedHost; } catch { return false; }
}

let webUserId: string | null = null;
async function webUser(): Promise<string | null> {
  if (!webUserId) {
    const u = await one<{ id: string }>("select id from users where role = 'web' limit 1");
    webUserId = u?.id ?? null;
  }
  return webUserId;
}

async function buildContext(req: import("fastify").FastifyRequest): Promise<GenContext | { error: string }> {
  const b = (req.body ?? {}) as Record<string, unknown>;
  const model = String(b.model ?? "backtalk-assistant");
  const isStory = !!b.story;
  const input = isStory
    ? { story: b.story as Record<string, string> }
    : { prompt: String(b.prompt ?? "") };
  if (!isStory && !(input.prompt ?? "").trim()) return { error: "prompt is required" };
  return {
    mode: "web", model, identity: "web:" + clientIp(req),
    apiKeyId: null, userId: await webUser(), allowFeedback: true, input, params: cleanParams(b),
  };
}

export function registerWeb(app: FastifyInstance): void {
  app.get("/web/models", async () => ({ data: await availableModels() }));

  // Streaming (SSE): used by the UI so answers appear as they generate.
  app.post("/web/generate/stream", async (req, reply: FastifyReply) => {
    if (!sameOrigin(req)) return reply.code(403).send({ error: "web mode is only available on the site" });
    const ctx = await buildContext(req);
    if ("error" in ctx) return reply.code(400).send({ error: ctx.error });

    reply.hijack();
    reply.raw.writeHead(200, {
      "content-type": "text/event-stream",
      "cache-control": "no-cache",
      connection: "keep-alive",
      "x-accel-buffering": "no",
    });
    const emit = (event: string, data: Record<string, unknown>) =>
      new Promise<void>((resolve) => {
        reply.raw.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`, () => resolve());
      });
    try {
      await runGenerationStream(ctx, emit);
    } catch (err) {
      log.error("web stream failed", { error: String(err) });
      reply.raw.write(`event: error\ndata: ${JSON.stringify({ message: "Internal error, try again later." })}\n\n`);
    } finally {
      reply.raw.end();
    }
  });

  // Non-streaming fallback (kept for simple clients / debugging).
  app.post("/web/generate", async (req, reply) => {
    if (!sameOrigin(req)) return reply.code(403).send({ error: "web mode is only available on the site" });
    const ctx = await buildContext(req);
    if ("error" in ctx) return reply.code(400).send({ error: ctx.error });
    try {
      const r = await runGeneration(ctx);
      return {
        request_id: r.requestId, feedback: r.feedback,
        answers: r.answers.map((a) => ({ variant: a.variant, output: a.output })),
        latency_ms: r.latencyMs,
      };
    } catch (err) {
      return sendError(reply, err);
    }
  });

  app.post("/web/feedback", async (req, reply) => {
    const b = (req.body ?? {}) as Record<string, unknown>;
    const requestId = String(b.request_id ?? "");
    const ok = await logFeedback(requestId, normChoice(b.choice));
    if (!ok) return reply.code(404).send({ error: "unknown request" });
    const row = await one<{ user_id: string | null }>(
      "select user_id from requests where id = $1",
      [requestId],
    );
    await noteFeedback(row?.user_id ?? null);
    return { ok: true };
  });
}
