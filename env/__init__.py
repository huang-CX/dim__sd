from gym.envs.registration import register

register(
    id='Exhx5-v0',
    entry_point='env.exhx5_env:Exhx5Env'
)
register(
    id='Exhx5WalkMod-v0',
    entry_point='env.exhx5_walkingmodule_env:Exhx5WalkEnv',
    max_episode_steps=500
)
register(
    id='Exhx5Adj-v0',
    entry_point='env.exhx5_adjustable_env:Exhx5AdjEnv',
    max_episode_steps=1000
)
