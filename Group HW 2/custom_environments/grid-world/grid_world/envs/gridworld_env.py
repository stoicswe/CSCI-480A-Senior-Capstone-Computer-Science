from collections import namedtuple
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding
import random as rand

State = namedtuple('State', ['y', 'x'])
Transition = namedtuple('Transition', ['state', 'prob', 'reward'])

class GridWorld(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        #do setup
        # Actions:
            # left = 0
            # right = 1
            # up = 2
            # down = 3
        self.ACTIONS = [0,1,2,3]
        self.ACTION_LEFT = 0
        self.ACTION_RIGHT = 1
        self.ACTION_UP = 2
        self.ACTION_DOWN = 3
        self.ACTIONS = [self.ACTION_LEFT, self.ACTION_RIGHT, self.ACTION_UP, self.ACTION_DOWN]
        self.height = 0
        self.width = 0
        self.FALL_REWARD = -40
        self.GOAL_REWARD = 100
        self.random_action_p = 0
        self.risky_p_loss = 0
        self.risky_goal_states = {}
        self.initial_state = []
        self.goal_states = {}
        self.cliff_states = set()
        self.current_state = self.initial_state
    
    def setup(self, height: int, width: int, random_action_p= 0.1, risky_p_loss=0.15):
        self.height = height
        self.width = width
        self.random_action_p = random_action_p
        self.risky_p_loss = risky_p_loss
        self.initial_state = State(self.height - 1, 0)
        self.goal_states = {State(self.height - 1, self.width - 1)}
        if height != 1:
            for x in range(width):
                for y in range(height):
                    s = State(y, x)
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
        x, y = s.x, s.y
        if a == self.ACTION_LEFT:
            return State(y, max(x - 1, 0))
        if a == self.ACTION_RIGHT:
            return State(y, min(x + 1, self.width - 1))
        if a == self.ACTION_UP:
            return State(max(y - 1, 0), x)
        if a == self.ACTION_DOWN:
            return State(min(y + 1, self.height - 1), x)

    def transitions(self, s):
        """
        returns a list of Transitions from the state s for each action, only non zero probabilities are given
        serves the lists for all actions at once
        """
        if s in self.goal_states:
            return [[Transition(state=s, prob=1.0, reward=0)] for a in self.ACTIONS]

        if s in self.risky_goal_states:
            goal = next(iter(self.goal_states))
            return [[Transition(state=goal, prob=self.risky_p_loss, reward=-50),
                     Transition(state=goal, prob=1-self.risky_p_loss, reward=100)] for a in self.ACTIONS]

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
                    transitions_actions.append(Transition(s_, p, r))
            transitions_full.append(transitions_actions)

        return transitions_full
    
    def sample_transition(self, s, a):
        """ Sample a single transition, duh. """
        trans = self.transitions(s)[a]
        state_probs = [tran.prob for tran in trans]
        return trans[np.random.choice(len(trans), p=state_probs)]