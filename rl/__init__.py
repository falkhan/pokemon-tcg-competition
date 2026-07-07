"""RL agent for the Kaggle Pokémon TCG AI Battle (cabt environment).

Package layout (see ARCHITECTURE.md for the full plan):

    encoders.py     obs dict -> fixed-width numpy state vector, option -> feature vector
    policy.py       torch option-scoring network + value head
    bc.py           behavior cloning warm start from the rule-based sample agent
    ppo.py          self-play PPO training loop
    deck_search.py  deck legality + mutation-bandit deck search
    eval.py         opponent-pool evaluation, win-rate metrics, replay dumps

Design rule: encoders are numpy-only so the Kaggle submission can reuse them
without a torch dependency (weights are exported to .npz).
"""
