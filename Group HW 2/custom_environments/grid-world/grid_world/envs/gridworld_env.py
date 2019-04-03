from collections import namedtuple
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, height: int, width: int, random_action_p= 0.1, risky_p_loss=0.15):
        #do setup
        # Actions:
            # left = 0
            # right = 1
            # up = 2
            # down = 3
        self.ACTIONS = [0,1,2,3]
        self.height = height
        self.width = width
        self.FALL_REWARD = -40
        self.GOAL_REWARD = 100
        self.random_action_p = random_action_p
        self.risky_p_loss = risky_p_loss
        self.risky_goal_states = {}
        self.initial_state = [self.height-1,0]
        self.goal_states = {[self.height-1, self.width-1]}
        self.cliff_states = set()
        self.current_state = self.initial_state

        if height != 1:
            for x in range(width):
                for y in range(height):
                    s = [x,y]
                    p_cliff = 0.1 * (y / height)**2 * bool(x > 1 and y > 0 and x < width-2 and y < height-1)
                    if s == self.initial_state or s in self.goal_states:
                        continue
                    
                    if np.random.random() < p_cliff:
                        self.cliff_states.add(s)
        
    
    def step(self, action):
        #given an action, move to a state, then generate the reward and return the new state
        new_s = self.target_state(self.current_state, action)
        r = 0
        done = False
            
        if(self.current_state in self.cliff_states):
            r = self.FALL_REWARD
            new_s = self.initial_state
        elif(self.current_state in self.goal_states):
            r = self.GOAL_REWARD
            done = True
        else:
            r = -1
        
        return new_s, r, done
    
    def reset(self):
        self.current_state = self.initial_state
        return self.current_state
    
    def render(self, mode='human', close=False):
        # no reason to render
        return
    
    def states(self):
        """ iterator over all possible states """
        for y in range(self.height):
            for x in range(self.width):
                s = [y, x]
                if s in self.cliff_states:
                    continue
                yield s
    
    def target_state(self, s, a):
        """ Return the next deterministic state """
        x = s[0]
        y = s[1]
        if a == 0:
            return [y, max(x - 1, 0)]
        if a == 1:
            return [y, min(x + 1, self.width - 1)]
        if a == 2:
            return [max(y - 1, 0), x]
        if a == 3:
            return [min(y + 1, self.height - 1), x]

    def transitions(self, s):
        if s in self.goal_states:
            return [[s, 1.0, 0] for a in self.ACTIONS]

        if s in self.risky_goal_states:
            goal = next(iter(self.goal_states))
            return [[[goal, self.risky_p_loss, -50], [goal, 1-self.risky_p_loss, 100]] for a in self.ACTIONS]

        transitions_full = []
        for a in self.ACTIONS:
            transitions_actions = []

            # over all *random* actions
            for a_ in self.ACTIONS:
                s_ = self.target_state(s, a_)
                if s_ in self.cliff_states:
                    r = self.FALL_REWARD
                    # s_ = self.initial_state
                    s_ = next(iter(self.goal_states))
                else:
                    r = -1
                p = 1.0 - self.random_action_p if a_ == a else self.random_action_p / 3
                if p != 0:
                    transitions_actions.append([s_, p, r])
            transitions_full.append(transitions_actions)

        return transitions_full