import random

# Example agent that selects cards at random.
def agent(obs_dict: dict) -> list[int]:
    return random.sample(
        list(range(len(obs_dict["select"]["option"]))), obs_dict["select"]["maxCount"]
    )