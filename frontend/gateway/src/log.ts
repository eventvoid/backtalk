// Minimal structured logger with secret redaction. Never logs tokens/keys.
const REDACT_KEYS = [
  "authorization", "x-node-token", "x-admin-token", "x-api-key",
  "node_token", "api_key", "apikey", "key", "token", "admin_token", "password",
];

function redact(value: unknown): unknown {
  if (Array.isArray(value)) return value.map(redact);
  if (value && typeof value === "object") {
    const out: Record<string, unknown> = {};
    for (const [k, v] of Object.entries(value as Record<string, unknown>)) {
      out[k] = REDACT_KEYS.includes(k.toLowerCase()) ? "[redacted]" : redact(v);
    }
    return out;
  }
  return value;
}

function emit(level: string, msg: string, extra?: Record<string, unknown>) {
  const line: Record<string, unknown> = { t: new Date().toISOString(), level, msg };
  if (extra) Object.assign(line, redact(extra) as object);
  // eslint-disable-next-line no-console
  console.log(JSON.stringify(line));
}

export const log = {
  info: (msg: string, extra?: Record<string, unknown>) => emit("info", msg, extra),
  warn: (msg: string, extra?: Record<string, unknown>) => emit("warn", msg, extra),
  error: (msg: string, extra?: Record<string, unknown>) => emit("error", msg, extra),
};
