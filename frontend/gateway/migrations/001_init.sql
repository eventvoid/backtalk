-- BackTalk gateway schema. gen_random_uuid() is core in PostgreSQL 13+.

create table if not exists users (
  id              uuid primary key default gen_random_uuid(),
  name            text not null,
  role            text not null default 'user',   -- user | admin | web
  rating          real not null default 1.0,      -- routing priority / feedback rating
  feedback_prompts integer not null default 0,
  feedback_given  integer not null default 0,
  created_at      timestamptz not null default now()
);

create table if not exists api_keys (
  id          uuid primary key default gen_random_uuid(),
  user_id     uuid references users(id) on delete cascade,
  name        text,
  key_prefix  text not null,                       -- shown in dashboard, e.g. "bt_native_ab12"
  key_hash    text not null unique,                -- sha256(full key); raw key is never stored
  mode        text not null default 'openai',      -- openai | native
  active      boolean not null default true,
  limits      jsonb not null default '{}'::jsonb,  -- optional per-key overrides
  created_at  timestamptz not null default now()
);
create index if not exists api_keys_hash_idx on api_keys(key_hash);

create table if not exists nodes (
  id              uuid primary key default gen_random_uuid(),
  name            text unique not null,
  url             text not null,                   -- gateway -> node base url
  status          text not null default 'online',  -- online | offline
  models          jsonb not null default '[]'::jsonb,
  max_concurrency integer not null default 1,
  active_requests integer not null default 0,
  load            real not null default 0,
  system          jsonb not null default '{}'::jsonb, -- cpu/ram/gpu/vram
  last_heartbeat  timestamptz not null default now(),
  created_at      timestamptz not null default now()
);

create table if not exists requests (
  id          uuid primary key default gen_random_uuid(),
  api_key_id  uuid references api_keys(id) on delete set null,
  user_id     uuid references users(id) on delete set null,
  mode        text not null,                       -- web | openai | native
  model       text not null,
  node_id     uuid references nodes(id) on delete set null,
  input       jsonb not null,                      -- {prompt} | {story}
  params      jsonb not null default '{}'::jsonb,
  status      text not null default 'ok',          -- ok | error | rejected | busy
  feedback    boolean not null default false,      -- produced 2 answers for A/B?
  latency_ms  integer,
  error       text,
  created_at  timestamptz not null default now()
);
create index if not exists requests_created_idx on requests(created_at desc);
create index if not exists requests_key_idx on requests(api_key_id);

create table if not exists responses (
  id          uuid primary key default gen_random_uuid(),
  request_id  uuid references requests(id) on delete cascade,
  variant     text not null default 'A',           -- A | B
  output      text not null,
  raw_output  text,
  tokens      integer,
  created_at  timestamptz not null default now()
);
create index if not exists responses_request_idx on responses(request_id);

create table if not exists feedback (
  id          uuid primary key default gen_random_uuid(),
  request_id  uuid references requests(id) on delete cascade,
  choice      text not null,                       -- A | B | none
  created_at  timestamptz not null default now()
);

create table if not exists limit_events (
  id          uuid primary key default gen_random_uuid(),
  api_key_id  uuid references api_keys(id) on delete set null,
  mode        text,
  kind        text not null,                       -- rate | active | busy
  detail      text,
  created_at  timestamptz not null default now()
);
create index if not exists limit_events_created_idx on limit_events(created_at desc);

-- The browser UI runs as this built-in user (no API key in the browser).
insert into users (name, role)
select 'web', 'web'
where not exists (select 1 from users where role = 'web');
