// Central config, all from env. Secrets never get logged (see log.ts).
import "dotenv/config";

function int(name: string, def: number): number {
  const v = process.env[name];
  const n = v ? Number(v) : NaN;
  return Number.isFinite(n) ? n : def;
}
function float(name: string, def: number): number {
  const v = process.env[name];
  const n = v ? Number(v) : NaN;
  return Number.isFinite(n) ? n : def;
}
function publicBaseUrl(): string {
  const value = (process.env.PUBLIC_BASE_URL ?? "").trim();
  if (!value) return "";
  let parsed: URL;
  try {
    parsed = new URL(value);
  } catch {
    throw new Error("PUBLIC_BASE_URL must be an absolute http(s) URL");
  }
  if (!["http:", "https:"].includes(parsed.protocol) || parsed.username || parsed.password) {
    throw new Error("PUBLIC_BASE_URL must be an absolute http(s) URL without credentials");
  }
  if (parsed.pathname !== "/" || parsed.search || parsed.hash) {
    throw new Error("PUBLIC_BASE_URL must contain only the origin, without a path, query, or hash");
  }
  return parsed.origin;
}
function trustProxy(): boolean | string {
  const value = (process.env.TRUST_PROXY ?? "").trim();
  if (!value || value === "false") return false;
  if (value === "true") return true;
  return value;
}

// Per-mode default limits. Override per API key via api_keys.limits (jsonb).
export interface Limits {
  active: number; // max concurrent requests
  rps: number;
  rpm: number;
  rph: number;
  rpd: number;
}

export const config = {
  host: process.env.HOST ?? "0.0.0.0",
  port: int("PORT", 8080),
  trustProxy: trustProxy(),
  // External origin used in documentation and generated public links. Keep
  // empty in local development to fall back to the browser's current origin.
  publicBaseUrl: publicBaseUrl(),
  databaseUrl: process.env.DATABASE_URL ?? "postgres://backtalk:backtalk@localhost:5432/backtalk",

  // Secrets (env only).
  adminToken: process.env.ADMIN_TOKEN ?? "",
  nodeToken: process.env.NODE_TOKEN ?? "",

  // Node health.
  nodeOfflineMs: int("NODE_OFFLINE_MS", 60000), // safely above the pull long-poll interval
  // Idle timeout, refreshed by every event during streamed generation. The
  // Larger default accommodates one-time accelerator graph compilation.
  nodeRequestTimeoutMs: int("NODE_REQUEST_TIMEOUT_MS", 300000),

  // Feedback: fraction of eligible (native/web) requests that produce 2 answers.
  feedbackRate: float("FEEDBACK_RATE", 0.3),
  // Drop a user's feedback rating toward this when they keep ignoring prompts.
  feedbackIgnorePenalty: float("FEEDBACK_IGNORE_PENALTY", 0.02),

  // Per-caller limits, ordered by intended use: the anonymous website is the
  // strictest, native API is moderate, and OpenAI-compatible clients get the
  // highest tier. These limits do not control backend-node concurrency.
  limits: {
    web: {
      active: int("WEB_ACTIVE", 1),
      rps: int("WEB_RPS", 1),
      rpm: int("WEB_RPM", 10),
      rph: int("WEB_RPH", 100),
      rpd: int("WEB_RPD", 500),
    } as Limits,
    openai: {
      active: int("OPENAI_ACTIVE", 4),
      rps: int("OPENAI_RPS", 4),
      rpm: int("OPENAI_RPM", 120),
      rph: int("OPENAI_RPH", 3000),
      rpd: int("OPENAI_RPD", 30000),
    } as Limits,
    native: {
      active: int("NATIVE_ACTIVE", 2),
      rps: int("NATIVE_RPS", 2),
      rpm: int("NATIVE_RPM", 30),
      rph: int("NATIVE_RPH", 500),
      rpd: int("NATIVE_RPD", 5000),
    } as Limits,
  },
};

export type Mode = "openai" | "native" | "web";

export function limitsFor(mode: Mode): Limits {
  return config.limits[mode];
}
