from collections import namedtuple
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand
import math

class StockMarket(gym.Env):
    etadata = {'render.modes': ['human']}

    def getStockDataVec(self, key):
        vec = []
        lines = open("data/" + key + ".csv", "r").read().splitlines()

        for line in lines[1:]:
            vec.append(float(line.split(",")[4]))

        return vec

    def __init__(self):
        self.stock_name = ""
        self.window_size = 0
        self.data = []
        self.l = 0
        self.batch_size = 32
    
    def setup(self, stock_name: str, window_size: int):
        self.stock_name = stock_name
        self.window_size = window_size
        self.data = self.getStockDataVec(stock_name)
        self.l = len(self.data) - 1
        self.batch_size = 32
    
    def getLength(self):
        return self.l
    
    def getStock(self, id):
        return self.data[id]
    
    def step(self, t, n):
        return self.getState(t, n)
    
    def sigmoid(self,x):
        try:
            if x < 0:
                return 1 - 1 / (1 + math.exp(x))
            return 1 / (1 + math.exp(-x))
        except OverflowError as err:
            print("Overflow err: {0} - Val of x: {1}".format(err, x))
        except ZeroDivisionError:
            print("division by zero!")
        except Exception as err:
            print("Error in sigmoid: " + err)
    
    def formatPrice(self, n):
	    return str(("-$" if n < 0 else "$") + "{0:.2f}".format(abs(n)))
    
    def getState(self, t, n):
	    d = t - n + 1
	    block = self.data[d:t + 1] if d >= 0 else -d * [self.data[0]] + self.data[0:t + 1] # pad with t0
	    res = []
	    for i in range(n - 1):
		    res.append(self.sigmoid(block[i + 1] - block[i]))

	    return np.array([res])
    
    def reset(self):
        return
    
    def render(self, mode='human', close=False):
        return