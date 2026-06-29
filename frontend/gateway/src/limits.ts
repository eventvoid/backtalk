// In-memory rate + concurrency limiter, keyed by an identity string (api key id
// or "web:<ip>"). Single-instance MVP; swap for Redis to scale horizontally.
import type { Limits } from "./config.js";
import { query } from "./db.js";

interface Bucket { times: number[]; active: number; }
const buckets = new Map<string, Bucket>();

const WINDOWS: Array<{ key: keyof Limits; ms: number }> = [
  { key: "rps", ms: 1000 },
  { key: "rpm", ms: 60_000 },
  { key: "rph", ms: 3_600_000 },
  { key: "rpd", ms: 86_400_000 },
];

export type LimitResult = { ok: true } | { ok: false; kind: "active" | "rate"; detail: string };

export function reserve(identity: string, limits: Limits): LimitResult {
  const now = Date.now();
  const b = buckets.get(identity) ?? { times: [], active: 0 };
  buckets.set(identity, b);
  // prune to the largest window (1 day)
  b.times = b.times.filter((t) => now - t < 86_400_000);

  if (b.active >= limits.active) {
    return { ok: false, kind: "active", detail: `max ${limits.active} concurrent request(s)` };
  }
  for (const w of WINDOWS) {
    const limit = limits[w.key];
    if (limit > 0 && b.times.filter((t) => now - t < w.ms).length >= limit) {
      return { ok: false, kind: "rate", detail: `limit ${limit}/${w.key}` };
    }
  }
  b.times.push(now);
  b.active += 1;
  return { ok: true };
}

export function release(identity: string): void {
  const b = buckets.get(identity);
  if (b) b.active = Math.max(0, b.active - 1);
}

export async function recordLimitEvent(
  apiKeyId: string | null,
  mode: string,
  kind: string,
  detail: string,
): Promise<void> {
  await query(
    `insert into limit_events (api_key_id, mode, kind, detail) values ($1,$2,$3,$4)`,
    [apiKeyId, mode, kind, detail],
  );
}
