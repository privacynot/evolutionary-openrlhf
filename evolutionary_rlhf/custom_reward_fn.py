def create_custom_reward_fn(individual):
    """Returns a reward function compatible with OpenRLHF's PPO trainer."""
    def reward_fn(samples, **kwargs):
        # samples: list of dicts with 'context' and 'response'
        return individual.R.mean(dim=-1).tolist()  # placeholder
    return reward_fn
