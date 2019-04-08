from agent import Agent
import gym
import stock_market
import sys
from keras.callbacks import TensorBoard, EarlyStopping

try:

	stock_name = "GSPC"
	window_size = 10
	episode_count = 1000
	#data = getStockDataVec(stock_name)

	env = gym.make('StockMarketEnv-v0')
	env.setup(stock_name, window_size)


	agent = Agent(window_size)
	batch_size = 32

	for e in range(episode_count + 1):
		print ("Episode " + str(e) + "/" + str(episode_count))
		state = env.getState(0, window_size + 1)

		total_profit = 0
		agent.inventory = []

		for t in range(env.getLength()):
			action = agent.act(state)

			# sit
			next_state = env.getState(t + 1, window_size + 1)
			reward = 0

			if action == 1: # buy
				agent.inventory.append(env.getStock(t))
				print ("Buy: " + env.formatPrice(env.getStock(t)))

			elif action == 2 and len(agent.inventory) > 0: # sell
				bought_price = agent.inventory.pop(0)
				reward = max(env.getStock(t) - bought_price, 0)
				total_profit += env.getStock(t) - bought_price
				print ("Sell: " + env.formatPrice(env.getStock(t)) + " | Profit: " + env.formatPrice(env.getStock(t) - bought_price))

			done = True if t == env.getLength() - 1 else False
			agent.memory.append((state, action, reward, next_state, done))
			state = next_state

			if done:
				print ("--------------------------------")
				print ("Total Profit: " + env.formatPrice(total_profit))
				print ("--------------------------------")

			if len(agent.memory) > batch_size:
				agent.expReplay(batch_size)

		if e % 10 == 0:
			agent.model.save("models/model_ep" + str(e))
except Exception as e:
	print("Error occured: {0}".format(e))
finally:
	exit()