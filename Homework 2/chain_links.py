import numpy as np

class Chain:
    def __init__(self, n, start_state, reward_left, reward_right):
        self.rewards = np.zeros(n)
        self.start_pos = start_state
        self.current_pos = start_state
        self.reward_left = reward_left
        self.reward_right = reward_right
        self.rewards[0] = reward_left
        self.rewards[n-1] = reward_right
    
    def reset(self):
        self.current_pos = self.start_pos
    
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
    
    def get_rewards(self):
        return self.rewards
    
    def get_pos(self):
        return self.current_pos

    def get_observation_space(self):
        return len(self.rewards)