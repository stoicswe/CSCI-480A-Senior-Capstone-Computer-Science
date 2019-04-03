from gym.envs.registration import register

register(
    id='ChainEnv-v0',
    entry_point='chain_env.envs:ChainEnv',
)