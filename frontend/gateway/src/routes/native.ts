// Native API. A key is optional (anonymous callers are rate-limited by IP);
// larger limits + several concurrent requests. No feedback (that's web-only).
import type { FastifyInstance } from "fastify";
import { resolveIdentity } from "../auth.js";
import { runGeneration } from "../generate.js";
import { availableModels } from "../nodes.js";
import { cleanParams, sendError } from "./util.js";

export function registerNative(app: FastifyInstance): void {
  app.get("/api/v1/models", async () => ({ data: await availableModels() }));

  app.post("/api/v1/generate", async (req, reply) => {
    const who = await resolveIdentity(req, reply, "native");
    if (!who) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const model = String(b.model ?? "backtalk-assistant");
    const isStory = !!b.story;
    const input = isStory
      ? { story: b.story as Record<string, string> }
      : { prompt: String(b.prompt ?? "") };
    if (!isStory && !(input.prompt ?? "").trim()) {
      return reply.code(400).send({ error: "prompt is required" });
    }
    try {
      const r = await runGeneration({
        mode: "native", model, identity: who.identity,
        apiKeyId: who.apiKeyId, userId: who.userId, allowFeedback: false, input, params: cleanParams(b),
      });
      return {
        request_id: r.requestId,
        model,
        output: r.answers[0]?.output ?? "",
        tokens: r.answers[0]?.tokens ?? null,
        latency_ms: r.latencyMs,
      };
    } catch (err) {
      return sendError(reply, err);
    }
  });
}
