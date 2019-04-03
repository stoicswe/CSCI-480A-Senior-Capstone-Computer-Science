from gym.envs.registration import register

register(
    id='GridworldEnv-v0',
    entry_point='gridworld_env.envs:GridWorld',
)