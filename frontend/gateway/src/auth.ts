// Auth: admin token (dashboard), node token (node registration/heartbeat),
// API keys (programmatic API). Keys are stored only as sha256 hashes.
import { createHash, randomBytes, timingSafeEqual } from "node:crypto";
import type { FastifyReply, FastifyRequest } from "fastify";
import { config, type Mode } from "./config.js";
import { one, query } from "./db.js";

export interface ApiKeyRow {
  id: string;
  user_id: string;
  mode: Mode;
  active: boolean;
  limits: Record<string, number>;
  rating: number;
}

export function hashKey(raw: string): string {
  return createHash("sha256").update(raw).digest("hex");
}

function safeEqual(a: string, b: string): boolean {
  const ab = Buffer.from(a);
  const bb = Buffer.from(b);
  if (ab.length !== bb.length || ab.length === 0) return false;
  return timingSafeEqual(ab, bb);
}

function bearer(req: FastifyRequest): string | null {
  const h = req.headers["authorization"];
  if (typeof h === "string" && h.toLowerCase().startsWith("bearer ")) return h.slice(7).trim();
  const x = req.headers["x-api-key"];
  return typeof x === "string" ? x.trim() : null;
}

export function requireAdmin(req: FastifyRequest, reply: FastifyReply): boolean {
  const tok = req.headers["x-admin-token"];
  if (!config.adminToken || typeof tok !== "string" || !safeEqual(tok, config.adminToken)) {
    reply.code(401).send({ error: "admin token required" });
    return false;
  }
  return true;
}

export function requireNode(req: FastifyRequest, reply: FastifyReply): boolean {
  const tok = req.headers["x-node-token"];
  if (!config.nodeToken || typeof tok !== "string" || !safeEqual(tok, config.nodeToken)) {
    reply.code(401).send({ error: "node token required" });
    return false;
  }
  return true;
}

export interface Identity { identity: string; apiKeyId: string | null; userId: string | null; }

function clientIp(req: FastifyRequest): string {
  return req.ip;
}

// Resolve a caller for an API mode. A key is OPTIONAL: with no key the caller is
// anonymous (rate-limited by IP); a provided key must be valid and match the
// mode. Returns null only when a key was provided but is invalid (reply sent).
export async function resolveIdentity(
  req: FastifyRequest,
  reply: FastifyReply,
  mode: Mode,
  openai = false,
): Promise<Identity | null> {
  const raw = bearer(req);
  if (!raw) {
    return { identity: `${mode}:${clientIp(req)}`, apiKeyId: null, userId: null };
  }
  const row = await one<ApiKeyRow>(
    `select k.id, k.user_id, k.mode, k.active, k.limits, u.rating
       from api_keys k join users u on u.id = k.user_id where k.key_hash = $1`,
    [hashKey(raw)],
  );
  const fail = (code: number, message: string) => {
    reply.code(code).send(openai ? { error: { message, type: "auth_error" } } : { error: message });
    return null;
  };
  if (!row || !row.active) return fail(401, "invalid API key");
  if (row.mode !== mode) return fail(403, `this key is for the ${row.mode} API`);
  return { identity: row.id, apiKeyId: row.id, userId: row.user_id };
}

// Create + store an API key, returning the raw key ONCE (never stored raw).
export async function createApiKey(userId: string, mode: Mode, name: string | null): Promise<string> {
  const raw = `bt_${mode}_${randomBytes(24).toString("base64url")}`;
  const prefix = raw.slice(0, `bt_${mode}_`.length + 4);
  await query(
    `insert into api_keys (user_id, name, key_prefix, key_hash, mode) values ($1,$2,$3,$4,$5)`,
    [userId, name, prefix, hashKey(raw), mode],
  );
  return raw;
}
