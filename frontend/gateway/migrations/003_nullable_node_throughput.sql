-- Throughput is unknown until a node has completed its first generation.
alter table nodes alter column throughput drop not null;
alter table nodes alter column throughput drop default;
update nodes set throughput = null where throughput = 0;
