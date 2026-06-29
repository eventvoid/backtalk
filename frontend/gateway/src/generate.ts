// Core orchestration: enforce limits -> pick a node -> run 1 or 2 generations
// -> log request/answers -> return. Shared by the web, native and openai routes.
import { config, limitsFor, type Mode } from "./config.js";
import { bumpActive, callNode, callNodeStream, pickNode, type NodeRow } from "./nodes.js";
import { reserve, release, recordLimitEvent } from "./limits.js";
import { finishRequest, logAnswers, logRequest, type LoggedAnswer } from "./dataset.js";
import { notePrompted } from "./users.js";
import { query } from "./db.js";
import { log } from "./log.js";

export type Emit = (event: string, data: Record<string, unknown>) => void | Promise<void>;

async function readNodeNDJSON(res: Response, onEvent: (m: Record<string, unknown>) => void): Promise<void> {
  const reader = res.body!.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    buf += decoder.decode(value, { stream: true });
    let nl;
    while ((nl = buf.indexOf("\n")) >= 0) {
      const line = buf.slice(0, nl).trim();
      buf = buf.slice(nl + 1);
      if (!line) continue;
      let message: Record<string, unknown>;
      try {
        message = JSON.parse(line) as Record<string, unknown>;
      } catch {
        continue;
      }
      onEvent(message);
    }
  }
}

export class GatewayError extends Error {
  constructor(public status: number, message: string, public kind?: string) {
    super(message);
  }
}

export interface GenContext {
  mode: Mode;
  model: string;
  identity: string;       // rate-limit identity (api key id, or "web:<ip>")
  apiKeyId: string | null;
  userId: string | null;
  allowFeedback: boolean; // native/web assistant requests only
  input: { prompt?: string; story?: Record<string, string> };
  params: Record<string, unknown>;
}

export interface GenResult {
  requestId: string;
  answers: LoggedAnswer[];
  feedback: boolean;
  node: NodeRow;
  input: string;
  modelInput: string;
  latencyMs: number;
}

export async function runGeneration(ctx: GenContext): Promise<GenResult> {
  const limits = limitsFor(ctx.mode);

  // 1) limits
  const reserved = reserve(ctx.identity, limits);
  if (!reserved.ok) {
    await recordLimitEvent(ctx.apiKeyId, ctx.mode, reserved.kind, reserved.detail);
    await logRequest({
      apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
      nodeId: null, input: ctx.input, params: ctx.params, status: "rejected",
      feedback: false, latencyMs: null, error: `${reserved.kind}: ${reserved.detail}`,
    });
    throw new GatewayError(429, `Rate limit reached (${reserved.detail}). Try again later.`, reserved.kind);
  }

  try {
    // 2) routing
    const node = await pickNode(ctx.model);
    if (!node) {
      await logRequest({
        apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
        nodeId: null, input: ctx.input, params: ctx.params, status: "busy",
        feedback: false, latencyMs: null, error: "no available node",
      });
      throw new GatewayError(503, "Server is busy, try again later.", "busy");
    }

    // 3) feedback decision (only assistant prompts, native/web)
    const feedback = ctx.allowFeedback && !!ctx.input.prompt && Math.random() < config.feedbackRate;
    const n = feedback ? 2 : 1;

    await bumpActive(node.id, 1);
    const started = Date.now();
    const answers: LoggedAnswer[] = [];
    let inputText = "";
    let modelInput = "";
    try {
      for (let i = 0; i < n; i++) {
        const seed =
          typeof ctx.params.seed === "number" ? (ctx.params.seed as number) + i : undefined;
        const r = await callNode(node, {
          model: ctx.model,
          prompt: ctx.input.prompt,
          story: ctx.input.story,
          params: { ...ctx.params, ...(seed !== undefined ? { seed } : {}) },
        });
        answers.push({
          variant: i === 0 ? "A" : "B",
          output: r.output ?? "",
          raw_output: r.raw_output,
          tokens: r.tokens,
        });
        inputText = r.input ?? inputText;
        modelInput = r.model_input ?? modelInput;
      }
    } catch (err) {
      const status = (err as { status?: number }).status;
      if (status === 400) {
        // Client/validation error from the node — node is healthy, surface as 400.
        await logRequest({
          apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
          nodeId: node.id, input: ctx.input, params: ctx.params, status: "error",
          feedback: false, latencyMs: Date.now() - started, error: "invalid request",
        });
        throw new GatewayError(400, (err as Error).message || "invalid request", "invalid_request");
      }
      // Node failed mid-job: take it offline so it isn't picked again, log, surface generic error.
      await query("update nodes set status = 'offline' where id = $1", [node.id]);
      log.error("node call failed", { node: node.name, error: String(err) });
      await logRequest({
        apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
        nodeId: node.id, input: ctx.input, params: ctx.params, status: "error",
        feedback: false, latencyMs: Date.now() - started, error: "node error",
      });
      throw new GatewayError(502, "Generation failed, try again later.", "node_error");
    } finally {
      await bumpActive(node.id, -1);
    }

    const latencyMs = Date.now() - started;
    const requestId = await logRequest({
      apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
      nodeId: node.id, input: ctx.input, params: ctx.params, status: "ok",
      feedback, latencyMs, error: null,
    });
    await logAnswers(requestId, answers);

    return { requestId, answers, feedback, node, input: inputText, modelInput, latencyMs };
  } finally {
    release(ctx.identity);
  }
}

// Streaming orchestration: real model events are forwarded as they arrive.
// Feedback requests stream A and B separately instead of waiting for both.
// `emit(event, data)` writes one SSE event to the client.
export async function runGenerationStream(ctx: GenContext, emit: Emit): Promise<void> {
  const limits = limitsFor(ctx.mode);
  const reserved = reserve(ctx.identity, limits);
  if (!reserved.ok) {
    await recordLimitEvent(ctx.apiKeyId, ctx.mode, reserved.kind, reserved.detail);
    await logRequest({
      apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
      nodeId: null, input: ctx.input, params: ctx.params, status: "rejected",
      feedback: false, latencyMs: null, error: `${reserved.kind}: ${reserved.detail}`,
    });
    await emit("error", { message: `Rate limit reached (${reserved.detail}). Try again later.` });
    return;
  }

  try {
    const node = await pickNode(ctx.model);
    if (!node) {
      await logRequest({
        apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
        nodeId: null, input: ctx.input, params: ctx.params, status: "busy",
        feedback: false, latencyMs: null, error: "no available node",
      });
      await emit("error", { message: "Server is busy, try again later." });
      return;
    }

    const feedback = ctx.allowFeedback && !!ctx.input.prompt && Math.random() < config.feedbackRate;
    const requestId = await logRequest({
      apiKeyId: ctx.apiKeyId, userId: ctx.userId, mode: ctx.mode, model: ctx.model,
      nodeId: node.id, input: ctx.input, params: ctx.params, status: "pending",
      feedback, latencyMs: null, error: null,
    });
    await emit("start", { request_id: requestId, feedback, mode: ctx.mode, model: ctx.model });
    if (feedback) await notePrompted(ctx.userId);

    await bumpActive(node.id, 1);
    const started = Date.now();
    const answers: LoggedAnswer[] = [];
    const ctrl = new AbortController();
    let timer: ReturnType<typeof setTimeout> | null = null;
    const refreshNodeTimeout = () => {
      if (timer) clearTimeout(timer);
      timer = setTimeout(() => ctrl.abort(), config.nodeRequestTimeoutMs);
    };
    try {
      const count = feedback ? 2 : 1;
      for (let i = 0; i < count; i++) {
        const variant = (i === 0 ? "A" : "B") as "A" | "B";
        const seed = typeof ctx.params.seed === "number" ? (ctx.params.seed as number) + i : undefined;
        if (feedback) await emit("variant_start", { variant });
        refreshNodeTimeout();
        const res = await callNodeStream(node, {
          model: ctx.model,
          prompt: ctx.input.prompt,
          story: ctx.input.story,
          params: { ...ctx.params, ...(seed !== undefined ? { seed } : {}) },
        }, ctrl.signal);
        let output = "";
        let raw = "";
        let tokens = 0;
        let liveRaw = "";
        let resultReceived = false;
        let nodeError = "";
        // The model writes in reverse: stream the live (reversed) text so the
        // user sees it being written, then send the flipped readable answer.
        await readNodeNDJSON(res, (m) => {
          refreshNodeTimeout();
          if (m.event === "chunk") {
            if (m.delta != null) {
              const delta = String(m.delta);
              const reset = Boolean(m.reset);
              liveRaw = reset ? delta : liveRaw + delta;
              void emit("stream", { variant, delta, reset, tokens: Number(m.tokens) || 0 });
            } else {
              // Backward compatibility while an older pull node is upgrading.
              liveRaw = String(m.raw ?? "");
              void emit("stream", { variant, text: liveRaw, tokens: Number(m.tokens) || 0 });
            }
            tokens = Number(m.tokens) || tokens;
          }
          else if (m.event === "result") {
            resultReceived = true;
            output = String(m.output ?? "");
            raw = String(m.raw_output ?? "");
            tokens = Number(m.tokens) || tokens;
          } else if (m.event === "error") {
            nodeError = String(m.detail ?? "generation failed");
          }
        });
        if (nodeError || !resultReceived) {
          throw new Error(nodeError || "node stream ended before a result");
        }
        answers.push({ variant, output, raw_output: raw, tokens });
        await emit("final", { variant, output });
      }
    } catch (err) {
      const status = (err as { status?: number }).status;
      if (status === 400) {
        await finishRequest(requestId, "error", Date.now() - started, node.id, "invalid request");
        await emit("error", { message: (err as Error).message || "invalid request" });
        return;
      }
      await query("update nodes set status = 'offline' where id = $1", [node.id]);
      log.error("node stream failed", { node: node.name, error: String(err) });
      await finishRequest(requestId, "error", Date.now() - started, node.id, "node error");
      await emit("error", { message: "Generation failed, try again later." });
      return;
    } finally {
      if (timer) clearTimeout(timer);
      await bumpActive(node.id, -1);
    }

    const latencyMs = Date.now() - started;
    await finishRequest(requestId, "ok", latencyMs, node.id);
    await logAnswers(requestId, answers);
    await emit("done", { request_id: requestId, feedback, latency_ms: latencyMs });
  } finally {
    release(ctx.identity);
  }
}
