import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS = {
    "4x4": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFG"
    ],
    "8x8": [
        "SFFFFFFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFG"
    ],
}

class FrozenLakeEnv(discrete.DiscreteEnv):
    """
    Winter is here. You and your friends were tossing around a frisbee at the park
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following
        SFFF
        FHFH
        FFFH
        HFFG
    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom
    G : goal, where the frisbee is located
    The episode ends when you reach the goal or fall in a hole.
    You receive a reward of 1 if you reach the goal, and zero otherwise.
    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, desc=None, map_name="4x4",is_slippery=False):
        if desc is None and map_name is None:
            raise ValueError('Must provide either desc or map_name')
        elif desc is None:
            desc = MAPS[map_name]
        self.desc = desc = np.asarray(desc,dtype='c')
        self.nrow, self.ncol = nrow, ncol = desc.shape
        self.reward_range = (0, 1)

        # IRL Adjustments
        self.height=self.ncol; self.width=self.nrow

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_slippery:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = float(newletter == b'G')
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = float(newletter == b'G')
                            li.append((1.0, newstate, rew, done))

        super(FrozenLakeEnv, self).__init__(nS, nA, P, isd)
    
    def get_transition_mat(self):
        """
        get transition dynamics of the gridworld

        return:
            P_a         NxNxN_ACTIONS transition probabilities matrix - 
                        P_a[s0, s1, a] is the transition prob of 
                        landing at state s1 when taking action 
                        a at state s0
        """
        N_STATES = self.height*self.width
        N_ACTIONS = self.nA
        P_a = np.zeros((N_STATES, N_STATES, N_ACTIONS))

        for si in range(N_STATES):
            posi = self.idx2pos(si)
            for a in range(N_ACTIONS):
                probs = self.get_transition_states_and_probs(posi, a)

                for posj, prob in probs:
                    sj = self.pos2idx(posj)
                    # Prob of si to sj given action a
                    # P_a[si, sj, a] = prob
                    print(si, sj, a)
                    print(int(si), int(sj), int(a))
                    P_a[int(si), int(sj), int(a)] = prob
        return P_a
    def get_transition_states_and_probs(self, state, action):
        """
        get all the possible transition states and their probabilities with [action] on [state]
            args
                state     (y, x)
                action    int
        returns
            a list of (state, probability) pair
        """
        li=[]
        newstate = self.P[self.pos2idx(state)][action][0][1]
        possibleactions = self.P[newstate]
        for a in possibleactions:
            print(possibleactions[a])
            li.append((self.idx2pos(possibleactions[a][0][1]),1/len(possibleactions)))
        return li
        
    def idx2pos(self, idx):
        """
        input:
        1d idx
        returns:
        2d column-major position
        """
        return (idx % self.height, idx // self.height)
    def pos2idx(self, pos):
        """
        input:
        column-major 2d position
        returns:
        1d index
        """
        return pos[0] + pos[1] * self.height
        #return pos + self.height
    

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout

        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()
        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True)
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile