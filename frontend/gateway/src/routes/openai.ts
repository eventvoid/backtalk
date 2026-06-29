// OpenAI-compatible API (requires an openai-mode key). Stricter limits, no
// feedback. Point any OpenAI SDK at <gateway>/v1.
import { randomBytes } from "node:crypto";
import type { FastifyInstance, FastifyReply } from "fastify";
import { resolveIdentity } from "../auth.js";
import { runGeneration } from "../generate.js";
import { availableModels } from "../nodes.js";
import { cleanParams, sendError } from "./util.js";

function parseStory(content: string): Record<string, string> {
  const text = (content ?? "").trim();
  if (text.startsWith("{")) {
    try {
      const obj = JSON.parse(text);
      if (obj && typeof obj === "object") {
        const out: Record<string, string> = {};
        for (const [k, v] of Object.entries(obj)) out[k] = String(v);
        return out;
      }
    } catch { /* fall through */ }
  }
  const out: Record<string, string> = {};
  for (const line of text.split("\n")) {
    const i = line.indexOf(":");
    if (i > 0) out[line.slice(0, i).trim()] = line.slice(i + 1).trim();
  }
  return out;
}

function sse(reply: FastifyReply, id: string, model: string, answer: string): void {
  reply.hijack(); // we write to reply.raw directly; stop Fastify from sending
  reply.raw.writeHead(200, {
    "content-type": "text/event-stream",
    "cache-control": "no-cache",
    connection: "keep-alive",
  });
  const created = Math.floor(Date.now() / 1000);
  const chunk = (delta: Record<string, unknown>, finish: string | null) =>
    `data: ${JSON.stringify({
      id, object: "chat.completion.chunk", created, model,
      choices: [{ index: 0, delta, finish_reason: finish }],
    })}\n\n`;
  reply.raw.write(chunk({ role: "assistant" }, null));
  // BackTalk generates the underlying sequence in reverse, so a readable
  // left-to-right delta does not exist until generation finishes. Send the
  // real final answer once instead of simulating token streaming word by word.
  reply.raw.write(chunk({ content: answer }, null));
  reply.raw.write(chunk({}, "stop"));
  reply.raw.write("data: [DONE]\n\n");
  reply.raw.end();
}

export function registerOpenAI(app: FastifyInstance): void {
  app.get("/v1/models", async () => {
    const created = Math.floor(Date.now() / 1000);
    const models = await availableModels();
    const ids = [...models.map((m) => String(m.id)), "backtalk-assistant", "backtalk-storyteller"];
    return { object: "list", data: ids.map((id) => ({ id, object: "model", created, owned_by: "backtalk" })) };
  });

  app.post("/v1/chat/completions", async (req, reply) => {
    const who = await resolveIdentity(req, reply, "openai", true);
    if (!who) return;
    const b = (req.body ?? {}) as Record<string, unknown>;
    const model = String(b.model ?? "backtalk-assistant");
    const messages = Array.isArray(b.messages) ? (b.messages as Array<Record<string, unknown>>) : [];
    const lastUser = [...messages].reverse().find((m) => m.role === "user");
    if (!lastUser) {
      return reply.code(400).send({ error: { message: "messages must contain a user message", type: "invalid_request_error" } });
    }
    const content = String(lastUser.content ?? "");
    const isStory = /storyteller|stories/.test(model);
    const input = isStory ? { story: parseStory(content) } : { prompt: content };

    try {
      const r = await runGeneration({
        mode: "openai", model, identity: who.identity,
        apiKeyId: who.apiKeyId, userId: who.userId, allowFeedback: false, input,
        params: cleanParams(b),
      });
      const answer = r.answers[0]?.output ?? "";
      const id = "chatcmpl-" + randomBytes(12).toString("hex");
      if (b.stream === true) {
        sse(reply, id, model, answer);
        return reply;
      }
      const promptTokens = Math.max(1, content.split(/\s+/).length);
      const completionTokens = r.answers[0]?.tokens ?? answer.split(/\s+/).length;
      return {
        id, object: "chat.completion", created: Math.floor(Date.now() / 1000), model,
        choices: [{ index: 0, message: { role: "assistant", content: answer }, finish_reason: "stop" }],
        usage: { prompt_tokens: promptTokens, completion_tokens: completionTokens, total_tokens: promptTokens + completionTokens },
      };
    } catch (err) {
      return sendError(reply, err, true);
    }
  });
}
