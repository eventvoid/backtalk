import type { FastifyReply, FastifyRequest } from "fastify";
import { GatewayError } from "../generate.js";
import { log } from "../log.js";

export function clientIp(req: FastifyRequest): string {
  return req.ip;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));

// Accept native (max_new_tokens) and OpenAI (max_tokens) param names; clamp to range.
export function cleanParams(body: Record<string, unknown>): Record<string, unknown> {
  const out: Record<string, unknown> = {};
  if (body.temperature != null) out.temperature = clamp(Number(body.temperature), 0.05, 2);
  if (body.top_k != null) out.top_k = clamp(Math.trunc(Number(body.top_k)), 1, 256);
  const maxTok = body.max_new_tokens ?? body.max_tokens;
  if (maxTok != null) out.max_new_tokens = clamp(Math.trunc(Number(maxTok)), 8, 1000);
  if (body.seed != null && body.seed !== "") out.seed = Math.trunc(Number(body.seed));
  return out;
}

// A/B for two-answer feedback, yes/no for single-answer feedback, else none.
export function normChoice(c: unknown): string {
  const s = String(c ?? "").toLowerCase();
  if (s === "a") return "A";
  if (s === "b") return "B";
  if (s === "yes") return "yes";
  if (s === "no") return "no";
  return "none";
}

// Map any thrown error to a safe client response (no internals leaked).
export function sendError(reply: FastifyReply, err: unknown, openai = false): FastifyReply {
  if (err instanceof GatewayError) {
    return reply.code(err.status).send(
      openai
        ? { error: { message: err.message, type: err.kind ?? "error", code: null } }
        : { error: err.message, kind: err.kind },
    );
  }
  log.error("unhandled route error", { error: String(err) });
  const msg = "Internal error, try again later.";
  return reply.code(500).send(openai ? { error: { message: msg, type: "api_error", code: null } } : { error: msg });
}
