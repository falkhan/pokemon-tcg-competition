#!/usr/bin/env zsh
# M8 run watcher — `watch -n 30 ./scripts/m8_watch.sh` in a spare terminal.
# Shows the live progress of the current M8 measurement/training runs.
T=/tmp/claude-1000/-home-falkhan-Documents-python-projects-pokemon-tcg-competition/f97ed68e-26e2-4468-906f-cbcf9ec7e2ff/tasks

echo "== load =="
uptime | sed 's/.*load/load/'

echo
echo "== M8.3 leg A: PPO probe (8 iters, evals @2/4/6/8; ~2h total) =="
tail -4 "$T/bc1f29ose.output" 2>/dev/null | cut -c1-110 || echo "(no output yet)"

echo
echo "== M8.2 leg B: solver-teacher probe (800 games -> 4 shards, then train) =="
n=$(ls data/bc_v2_solver/shard_*.npz 2>/dev/null | wc -l)
echo "collection: $((n * 200))/800 games ($n/4 shards on disk)"
tail -2 "$T/bepqi3sdt.output" 2>/dev/null | cut -c1-110

echo
echo "== queued: M8.4(b) sims ladder {16,32,64} (starts when leg A frees cores) =="
