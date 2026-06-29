// Request/response/feedback logging + dataset export (JSONL) for future training.
import { one, query } from "./db.js";

export interface LoggedAnswer { variant: "A" | "B"; output: string; raw_output?: string; tokens?: number; }

export async function logRequest(input: {
  apiKeyId: string | null;
  userId: string | null;
  mode: string;
  model: string;
  nodeId: string | null;
  input: unknown;
  params: unknown;
  status: string;
  feedback: boolean;
  latencyMs: number | null;
  error: string | null;
}): Promise<string> {
  const row = await one<{ id: string }>(
    `insert into requests (api_key_id, user_id, mode, model, node_id, input, params, status, feedback, latency_ms, error)
       values ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11) returning id`,
    [
      input.apiKeyId, input.userId, input.mode, input.model, input.nodeId,
      JSON.stringify(input.input), JSON.stringify(input.params),
      input.status, input.feedback, input.latencyMs, input.error,
    ],
  );
  return (row as { id: string }).id;
}

export async function finishRequest(
  id: string,
  status: string,
  latencyMs: number | null,
  nodeId: string | null,
  error: string | null = null,
): Promise<void> {
  await query(
    `update requests set status = $2, latency_ms = $3, node_id = coalesce($4, node_id), error = $5 where id = $1`,
    [id, status, latencyMs, nodeId, error],
  );
}

export async function logAnswers(requestId: string, answers: LoggedAnswer[]): Promise<void> {
  for (const a of answers) {
    await query(
      `insert into responses (request_id, variant, output, raw_output, tokens) values ($1,$2,$3,$4,$5)`,
      [requestId, a.variant, a.output, a.raw_output ?? null, a.tokens ?? null],
    );
  }
}

export async function logFeedback(requestId: string, choice: string): Promise<boolean> {
  const req = await one<{ id: string; feedback: boolean }>(
    "select id, feedback from requests where id = $1",
    [requestId],
  );
  if (!req) return false;
  await query("insert into feedback (request_id, choice) values ($1,$2)", [requestId, choice]);
  return true;
}

export async function feedbackStats(): Promise<Record<string, number>> {
  const rows = await query<{ choice: string; n: string }>(
    "select choice, count(*)::text as n from feedback group by choice",
  );
  const out: Record<string, number> = { A: 0, B: 0, none: 0 };
  for (const r of rows) out[r.choice] = Number(r.n);
  return out;
}

// Export collected data as JSONL: one line per request with its answers + choice.
export async function exportJsonl(filters: { mode?: string; model?: string; limit?: number }): Promise<string> {
  const where: string[] = ["r.status = 'ok'"];
  const params: unknown[] = [];
  if (filters.mode) { params.push(filters.mode); where.push(`r.mode = $${params.length}`); }
  if (filters.model) { params.push(filters.model); where.push(`r.model = $${params.length}`); }
  params.push(Math.min(filters.limit ?? 10000, 100000));
  const limitIdx = params.length;

  const rows = await query<{
    id: string; mode: string; model: string; input: unknown; params: unknown;
    latency_ms: number | null; created_at: string; feedback: boolean;
    answers: Array<{
      variant: "A" | "B";
      output: string;
      raw_output: string | null;
      tokens: number | null;
    }>;
    choice: string | null;
  }>(
    `select r.id, r.mode, r.model, r.input, r.params, r.latency_ms, r.created_at, r.feedback,
            coalesce(json_agg(json_build_object(
              'variant', resp.variant,
              'output', resp.output,
              'raw_output', resp.raw_output,
              'tokens', resp.tokens
            ) order by resp.created_at)
                     filter (where resp.id is not null), '[]') as answers,
            (select choice from feedback f where f.request_id = r.id order by created_at desc limit 1) as choice
       from requests r
       left join responses resp on resp.request_id = r.id
      where ${where.join(" and ")}
      group by r.id
      order by r.created_at desc
      limit $${limitIdx}`,
    params,
  );
  return rows
    .map((r) => {
      const selectedVariant =
        r.choice === "A" || r.choice === "B" ? r.choice :
        r.choice === "yes" ? "A" : null;
      const selectedAnswer =
        selectedVariant ? r.answers.find((a) => a.variant === selectedVariant)?.output ?? null : null;
      return JSON.stringify({
        id: r.id, mode: r.mode, model: r.model, input: r.input, params: r.params,
        answers: r.answers,
        voted: r.choice !== null,
        user_choice: r.choice,
        selected: r.choice,
        selected_variant: selectedVariant,
        selected_answer: selectedAnswer,
        feedback_offered: r.feedback,
        feedback: r.feedback,
        latency_ms: r.latency_ms, timestamp: r.created_at,
      });
    })
    .join("\n");
}
