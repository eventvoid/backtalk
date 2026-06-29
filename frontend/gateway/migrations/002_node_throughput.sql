-- Nodes accept unlimited concurrency now; the gateway prioritises them by load
-- (active_requests) and throughput (tokens/sec) instead of a hard slot cap.
alter table nodes add column if not exists throughput real not null default 0;
