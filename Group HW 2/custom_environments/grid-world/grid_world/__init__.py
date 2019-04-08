from gym.envs.registration import register

register(
    id='GridworldEnv-v0',
    entry_point='grid_world.envs:GridWorld',
)