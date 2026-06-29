import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import Fastify from "fastify";
import cors from "@fastify/cors";
import fastifyStatic from "@fastify/static";
import { config } from "./config.js";
import { log } from "./log.js";
import { migrate } from "./migrate.js";
import { pool } from "./db.js";
import { listNodes, sweepOffline } from "./nodes.js";
import { registerInternal } from "./routes/internal.js";
import { registerWeb } from "./routes/web.js";
import { registerNative } from "./routes/native.js";
import { registerOpenAI } from "./routes/openai.js";
import { registerAdmin } from "./routes/admin.js";

const here = dirname(fileURLToPath(import.meta.url));
const publicDir = join(here, "..", "public");

async function main(): Promise<void> {
  if (!config.adminToken || !config.nodeToken) {
    log.warn("ADMIN_TOKEN and/or NODE_TOKEN are empty — set them in .env for any real use");
  }

  await migrate();
  // The gateway owns active_requests (via bumpActive); clear any stale counts
  // left from a previous run so routing starts from a clean slate.
  await pool.query("update nodes set active_requests = 0").catch(() => {});

  const app = Fastify({ logger: false, bodyLimit: 1_000_000, trustProxy: config.trustProxy });
  await app.register(cors, { origin: true, methods: ["GET", "POST"] });
  // Serve /app.css, /app.js, etc. Page routes below are explicit (index disabled
  // so there's no route clash on "/").
  await app.register(fastifyStatic, { root: publicDir, prefix: "/", index: false });

  app.get("/health", async () => {
    const nodes = await listNodes();
    const online = nodes.filter((n) => n.status === "online").length;
    return { status: "ok", nodes_total: nodes.length, nodes_online: online };
  });
  app.get("/runtime-config.js", (_req, reply) => {
    const runtimeConfig = JSON.stringify({ publicBaseUrl: config.publicBaseUrl || null });
    return reply
      .header("cache-control", "no-store")
      .type("application/javascript")
      .send(`window.BACKTALK_CONFIG=${runtimeConfig};`);
  });

  // Clean page routes (assets like /app.css are served by @fastify/static).
  app.get("/", (_req, reply) => reply.sendFile("index.html"));
  app.get("/stories", (_req, reply) => reply.sendFile("stories.html"));
  app.get("/docs", (_req, reply) => reply.sendFile("docs.html"));
  app.get("/dashboard", (_req, reply) => reply.sendFile("dashboard.html"));

  registerInternal(app);
  registerWeb(app);
  registerNative(app);
  registerOpenAI(app);
  registerAdmin(app);

  const timer = setInterval(() => {
    sweepOffline().catch((err) => log.error("offline sweep failed", { error: String(err) }));
  }, 10_000);

  const shutdown = async () => {
    clearInterval(timer);
    await app.close();
    await pool.end();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);

  await app.listen({ host: config.host, port: config.port });
  log.info("gateway listening", { host: config.host, port: config.port });
}

main().catch((err) => {
  log.error("gateway failed to start", { error: String(err) });
  process.exit(1);
});
