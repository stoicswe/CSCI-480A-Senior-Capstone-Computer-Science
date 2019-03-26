import gym
from gym import error, spaces, utils
from gym.utils import seeding

class FooEnv(gym.Env):
  metadata = {'render.modes': ['human']}

  def __init__(self, reward_distribution):
    self.states = 10
    self.start_pos = 5
    self.current_pos = self.start_pos
    self.rewards = [0.1, 0, 0, 0, 0, 0, 0, 0, 0, 1.0]

    def step(self, action):
        if (action == 0):
            if (self.current_pos == 0):
                return self.current_pos, self.rewards[0], True
            else:
                self.current_pos = self.current_pos - 1
                return self.current_pos, self.rewards[self.current_pos], False
        
        if (action == 1):
            if (self.current_pos == (len(self.rewards) - 1)):
                return self.current_pos, self.rewards[self.current_pos], True
            else:
                self.current_pos = self.current_pos + 1
                return self.current_pos, self.rewards[self.current_pos], False

    def reset(self):
        self.current_pos = self.start_pos

    def sampling(self):
        if (rand.random() < 0.5):
            return 0
        else:
            return 1

  def render(self, mode='human', close=False):
     