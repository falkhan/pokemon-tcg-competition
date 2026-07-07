"""Behavior cloning warm start (ARCHITECTURE.md §4.1 step 1, milestone M1).

Plan:
 1. Wrap the rule-based sample agent (sample-agent/main.py) in
    rl.eval.RecordingAgent and self-play a few thousand games.
 2. For each recorded decision, encode (state, context, options) with
    rl.encoders and treat the agent's first pick as the class label.
 3. Train OptionScorer with cross-entropy over the presented options
    (mask padding to -inf), Adam, a few epochs.
 4. Save checkpoint -> starting point for ppo.py; export via policy.save_npz.

Success gate: >95% win rate vs the random agent (rl.eval.play_games).
"""

# TODO: collect_dataset(n_games) -> list[(state_ctx, options, label)]
# TODO: train(dataset, epochs=..) -> OptionScorer
# TODO: __main__ entry: collect, train, evaluate vs random, save checkpoint
