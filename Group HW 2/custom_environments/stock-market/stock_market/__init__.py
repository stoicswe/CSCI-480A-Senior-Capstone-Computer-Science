from gym.envs.registration import register

register(
    id='StockMarketEnv-v0',
    entry_point='stock_market.envs:StockMarket',
)