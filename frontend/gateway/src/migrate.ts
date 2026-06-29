// Simple forward-only migration runner. Applies migrations/*.sql in name order,
// tracked in schema_migrations. `reset` drops the public schema first.
import { readFileSync, readdirSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { pool } from "./db.js";
import { log } from "./log.js";

const here = dirname(fileURLToPath(import.meta.url));
const migrationsDir = join(here, "..", "migrations");

export async function migrate(): Promise<void> {
  await pool.query(
    `create table if not exists schema_migrations (
       version text primary key, applied_at timestamptz not null default now())`,
  );
  const applied = new Set(
    (await pool.query<{ version: string }>("select version from schema_migrations")).rows.map(
      (r) => r.version,
    ),
  );
  const files = readdirSync(migrationsDir).filter((f) => f.endsWith(".sql")).sort();
  for (const file of files) {
    if (applied.has(file)) continue;
    const sql = readFileSync(join(migrationsDir, file), "utf8");
    const client = await pool.connect();
    try {
      await client.query("begin");
      await client.query(sql);
      await client.query("insert into schema_migrations(version) values ($1)", [file]);
      await client.query("commit");
      log.info("migration applied", { file });
    } catch (err) {
      await client.query("rollback");
      throw err;
    } finally {
      client.release();
    }
  }
}

async function reset(): Promise<void> {
  await pool.query("drop schema public cascade; create schema public;");
  log.info("schema reset");
  await migrate();
}

const cmd = process.argv[2] ?? "up";
if (process.argv[1] && /migrate\.(ts|js)$/.test(process.argv[1])) {
  (cmd === "reset" ? reset() : migrate())
    .then(() => pool.end())
    .then(() => process.exit(0))
    .catch((err) => {
      log.error("migration failed", { error: String(err) });
      process.exit(1);
    });
}
