def test_diffusion_init():
    from evolutionary_rlhf.diffusion_reward import DiffusionRewardModel
    config = {'hidden_dim': 256, 'diffusion_steps': 100, 'condition_dim': 1024, 'max_len': 64, 'learning_rate': 1e-4}
    model = DiffusionRewardModel(config)
    assert model is not None
