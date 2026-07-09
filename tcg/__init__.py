"""The readable refactor of the whole agent codebase for the Kaggle Pokémon
TCG AI Battle (cabt): the rule-based pilot AND the training/shipping side that
previously lived in ``rl/`` (see docs/M6.md).

Package layout, by dependency tier:

Pure Python + the ``cg`` engine only (can ship inside a submission bundle):

    models.py     Card / Attack dataclasses built from the engine's card database
    library.py    the card database: CARDS / ATTACKS lookup tables + card-id sets
    constants.py  every tuned score tier and game constant, with the rationale
    combat.py     damage math: can_afford() / best_damage()
    pilot.py      make_generic_pilot() — scores every legal option each turn
    decks.py      deck-file loading (decks/*.csv, deck.csv)
    teachers.py   isolated loader for the sample-agent rule experts

+ numpy (and polars where noted) — training-side data plumbing:

    selfplay.py      parallel PPO self-play collection + the .npz shard writer
    value_training.py supervised value-head training (torch inside functions)
    encoders.py      observation/option -> vector encoders (polars for the
                     card-feature parquet; numpy-only after import)
    deck_search.py   deck legality + mutation-bandit deck search (polars)

+ torch — networks and training loops:

    network.py          OptionScorer policy/value net + save_npz export
    behavior_cloning.py BC collection + training from the rule teacher
    ppo.py              self-play PPO (GAE + clipped update + promotion gate)
    search.py           determinized MCTS + the hybrid rule/value agent

+ kaggle_environments — running and shipping:

    evaluation.py    play_games / option_type_report + replay pages
    shipping.py      submission export + pre-submission gates

This package intentionally does not import its submodules here: building the
card tables requires the ``cg`` engine, and ``import tcg`` alone should stay
side-effect free.

The ``rl/`` package still exists unchanged and remains the live path for the
submission pipeline (``rl.export`` / ``rl.gate`` / ``submission_rules``);
switching consumers over to ``tcg`` is gated on real-engine gate games and is
the remaining follow-up (docs/M6.md).
"""
