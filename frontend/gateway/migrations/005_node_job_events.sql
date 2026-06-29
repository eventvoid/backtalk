-- Ordered, durable transport for real generation events from pull nodes.
create table if not exists node_job_events (
  id         bigserial primary key,
  job_id     uuid not null references node_jobs(id) on delete cascade,
  event      jsonb not null,
  created_at timestamptz not null default now()
);
create index if not exists node_job_events_job_idx on node_job_events(job_id, id);
