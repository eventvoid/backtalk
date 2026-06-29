-- Pull nodes need only outbound HTTPS access to the gateway. The gateway
-- queues work; a node polls, runs it locally, then posts the result.
alter table nodes alter column url drop not null;
alter table nodes add column if not exists transport text not null default 'push';

create table if not exists node_jobs (
  id          uuid primary key default gen_random_uuid(),
  node_id     uuid not null references nodes(id) on delete cascade,
  payload     jsonb not null,
  status      text not null default 'queued', -- queued | running | done | error | cancelled
  result      jsonb,
  error       text,
  error_status integer,
  created_at  timestamptz not null default now(),
  started_at  timestamptz,
  finished_at timestamptz
);
create index if not exists node_jobs_poll_idx
  on node_jobs(node_id, created_at) where status = 'queued';
