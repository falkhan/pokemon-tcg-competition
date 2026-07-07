"""Self-play PPO training loop (ARCHITECTURE.md §4.1, §5.2).

Start from the BC checkpoint (bc.py). Reference implementation to crib from:
CleanRL's single-file ppo.py — https://docs.cleanrl.dev/rl-algorithms/ppo/ —
adapted for the variable-length option action space (logits over presented
options instead of a fixed Discrete head).

Loop sketch:
    for iteration in range(N):
        collect GAMES_PER_ITER games vs an opponent pool
            (current policy, past checkpoints, rule-based, random)
        rewards: terminal ±1  +  0.1 * prize-delta shaping per step
        advantages: GAE(lambda) using the value head
        PPO clipped update over minibatches of decisions
        every EVAL_EVERY: evaluate vs best checkpoint; promote only if
            win rate >= 0.55 over >= 200 games; dump replays; log TensorBoard

Throughput note: one battle per process (cg DLL global state) — use
multiprocessing.Pool for game collection, one env per worker.
"""

# TODO: Trajectory dataclass (state_ctx, options, action, logprob, value, reward)
# TODO: collect(policy, opponent_pool, n_games) using rl.eval.RecordingAgent
# TODO: gae(trajectories, gamma=0.99, lam=0.95)
# TODO: ppo_update(model, optimizer, batch, clip=0.2, epochs=4)
# TODO: __main__: full loop with checkpointing + TensorBoard SummaryWriter
